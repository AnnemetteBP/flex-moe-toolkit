from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from fake_test_models.fake_flex_olmo import FakeFlexOlmoModel
from flex_moe_toolkit.core.metrics import load_balance
from flex_moe_toolkit.core.routing_diagnostics import (
    compute_all_metrics,
    compute_coactivation,
    compute_entropy,
)
from scripts.flex_olmo.utils.evaluate_fake_flex_olmo_dataset import (
    DATASET_PATH,
    load_jsonl,
    text_to_fake_inputs,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "flex_olmo" / "paper_style"
MAX_SEQ_LEN = 32
TOP_K_VALUES = (1, 2, 4)
DOMAIN_SPECIALIZATION_TOP_K = 2


def build_prompt(example, choice_text):
    return f"Question: {example['question']}\nChoice: {choice_text}"


def tokenize_prompt(prompt: str) -> list[str]:
    tokens = prompt.replace("\n", " ").split()
    clipped = tokens[:MAX_SEQ_LEN]
    return clipped if clipped else ["<empty>"]


def run_prompt(model, prompt: str):
    tokens = tokenize_prompt(prompt)
    inputs = text_to_fake_inputs(
        prompt,
        hidden_size=model.config.hidden_size,
        sequence_length=len(tokens),
    )
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    router_logits_by_layer = tuple(layer.detach().cpu().float() for layer in outputs.router_logits)
    return tokens, inputs, outputs, router_logits_by_layer


def top1_usage_counts(router_logits: torch.Tensor) -> torch.Tensor:
    top1 = torch.argmax(router_logits, dim=-1)
    return torch.bincount(top1.reshape(-1), minlength=router_logits.shape[-1]).float()


def top1_token_assignments(router_logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(router_logits, dim=-1)


def token_pair_coactivation(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    topk = torch.topk(torch.softmax(router_logits, dim=-1), k=top_k, dim=-1).indices
    num_experts = router_logits.shape[-1]
    matrix = torch.zeros((num_experts, num_experts), dtype=torch.float32)
    for token_experts in topk.reshape(-1, top_k):
        experts = sorted(int(idx) for idx in token_experts.tolist())
        for left, right in combinations(experts, 2):
            matrix[left, right] += 1
            matrix[right, left] += 1
    return matrix


def combination_counter(router_logits: torch.Tensor, top_k: int) -> Counter:
    topk = torch.topk(torch.softmax(router_logits, dim=-1), k=top_k, dim=-1).indices
    counter = Counter()
    for token_experts in topk.reshape(-1, top_k).tolist():
        counter[tuple(sorted(int(idx) for idx in token_experts))] += 1
    return counter


def pseudo_choice_loss(choice_scores: list[float], correct_index: int) -> float:
    scores = torch.tensor(choice_scores, dtype=torch.float32)
    log_probs = torch.log_softmax(scores, dim=0)
    return float(-log_probs[correct_index].item())


def collect_analysis_records(model, examples):
    records = []

    for example in examples:
        choice_payloads = []
        for choice_idx, choice_text in enumerate(example["choices"]):
            prompt = build_prompt(example, choice_text)
            tokens, _inputs, outputs, router_logits_by_layer = run_prompt(model, prompt)
            score = float(outputs.last_hidden_state[:, -1, :].mean().item())
            choice_payloads.append(
                {
                    "choice_idx": choice_idx,
                    "choice_text": choice_text,
                    "prompt": prompt,
                    "tokens": tokens,
                    "score": score,
                    "router_logits_by_layer": router_logits_by_layer,
                }
            )

        scores = [payload["score"] for payload in choice_payloads]
        predicted_payload = max(choice_payloads, key=lambda item: item["score"])
        correct_payload = choice_payloads[example["correct_choice_idx"]]
        correct_first_layer = correct_payload["router_logits_by_layer"][0]
        entropy_mean, entropy_per_token = compute_entropy(correct_first_layer)
        max_confidence = torch.softmax(correct_first_layer, dim=-1).max(dim=-1).values

        records.append(
            {
                "example_id": example["id"],
                "benchmark": example["benchmark"],
                "language": example["language"],
                "correct_choice_idx": example["correct_choice_idx"],
                "predicted_choice_idx": predicted_payload["choice_idx"],
                "is_correct": predicted_payload["choice_idx"] == example["correct_choice_idx"],
                "choice_scores": scores,
                "pseudo_loss": pseudo_choice_loss(scores, example["correct_choice_idx"]),
                "correct_tokens": correct_payload["tokens"],
                "correct_router_logits_by_layer": correct_payload["router_logits_by_layer"],
                "entropy_per_token": entropy_per_token.detach().cpu(),
                "entropy_mean": float(entropy_mean.item()),
                "routing_confidence_per_token": max_confidence.detach().cpu(),
                "routing_confidence_mean": float(max_confidence.mean().item()),
            }
        )

    return records


def plot_routing_probability_distributions(records):
    num_layers = len(records[0]["correct_router_logits_by_layer"])
    for layer_idx in range(num_layers):
        flattened_probs = []
        for record in records:
            probs = torch.softmax(record["correct_router_logits_by_layer"][layer_idx], dim=-1)
            flattened_probs.extend(probs.reshape(-1).tolist())

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(flattened_probs, bins=30, color="#2f6db2", edgecolor="black")
        ax.set_xlabel("Routing probability")
        ax.set_ylabel("Count")
        ax.set_title(f"Routing Probability Distribution - Layer {layer_idx}")
        fig.savefig(OUTPUT_DIR / f"routing_prob_layer_{layer_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_usage_distribution(records):
    num_experts = records[0]["correct_router_logits_by_layer"][0].shape[-1]
    usage_counts = torch.zeros(num_experts, dtype=torch.float32)
    for record in records:
        usage_counts += top1_usage_counts(record["correct_router_logits_by_layer"][0])

    usage = usage_counts / usage_counts.sum()
    mean_usage = float(usage.mean().item())
    std_usage = float(usage.std(unbiased=False).item())
    cv = std_usage / mean_usage if mean_usage > 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(num_experts), usage.tolist(), color="#2f6db2")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Normalized usage")
    ax.set_title(f"Expert Usage Distribution (CV={cv:.4f})")
    fig.savefig(OUTPUT_DIR / "usage_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return usage, cv


def plot_entropy_views(records):
    flattened_entropy = []
    max_length = max(record["entropy_per_token"].shape[-1] for record in records)
    padded = []

    for record in records:
        entropy_values = record["entropy_per_token"].reshape(-1).tolist()
        flattened_entropy.extend(entropy_values)
        row = [math.nan] * max_length
        for idx, value in enumerate(entropy_values):
            row[idx] = value
        padded.append(row)

    entropy_curve = np.nanmean(np.array(padded, dtype=float), axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(flattened_entropy, bins=20, color="#2f6db2", edgecolor="black")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Token count")
    ax.set_title("Entropy Histogram")
    fig.savefig(OUTPUT_DIR / "entropy_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(entropy_curve)), entropy_curve, color="#2f6db2")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Mean entropy")
    ax.set_title("Entropy over Sequence")
    fig.savefig(OUTPUT_DIR / "entropy_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return float(np.nanmean(entropy_curve))


def plot_corrected_coactivation(records):
    num_layers = len(records[0]["correct_router_logits_by_layer"])
    num_experts = records[0]["correct_router_logits_by_layer"][0].shape[-1]
    layerwise_matrices = [
        torch.zeros((num_experts, num_experts), dtype=torch.float32)
        for _ in range(num_layers)
    ]
    for record in records:
        for layer_idx, layer_router_logits in enumerate(record["correct_router_logits_by_layer"]):
            layerwise_matrices[layer_idx] += token_pair_coactivation(layer_router_logits, top_k=2)

    matrix = layerwise_matrices[0].clone()
    for layer_matrix in layerwise_matrices[1:]:
        matrix += layer_matrix

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix.numpy(), cmap="viridis", ax=ax)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Expert")
    ax.set_title("Corrected Token-Level Coactivation Across All Layers")
    fig.savefig(OUTPUT_DIR / "coactivation_heatmap_fixed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    for layer_idx, layer_matrix in enumerate(layerwise_matrices):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(layer_matrix.numpy(), cmap="viridis", ax=ax)
        ax.set_xlabel("Expert")
        ax.set_ylabel("Expert")
        ax.set_title(f"Corrected Token-Level Coactivation Layer {layer_idx}")
        fig.savefig(
            OUTPUT_DIR / f"coactivation_heatmap_fixed_layer_{layer_idx}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

    return matrix, layerwise_matrices


def save_expert_token_specialization(records):
    token_counters = defaultdict(Counter)
    for record in records:
        assignments = top1_token_assignments(record["correct_router_logits_by_layer"][0]).reshape(-1).tolist()
        tokens = record["correct_tokens"]
        for token, expert_idx in zip(tokens, assignments):
            normalized_token = token.lower().strip(".,:;!?()[]{}\"'")
            if normalized_token:
                token_counters[int(expert_idx)][normalized_token] += 1

    payload = {
        f"expert_{expert_idx}": counter.most_common(10)
        for expert_idx, counter in sorted(token_counters.items())
    }
    with (OUTPUT_DIR / "expert_token_specialization.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload


def compute_domain_specialization(records, top_k=DOMAIN_SPECIALIZATION_TOP_K):
    num_layers = len(records[0]["correct_router_logits_by_layer"])
    num_experts = records[0]["correct_router_logits_by_layer"][0].shape[-1]
    domains = sorted({record["benchmark"] for record in records})
    domain_to_idx = {domain: idx for idx, domain in enumerate(domains)}
    num_domains = len(domains)
    specialization_by_layer = []

    for layer_idx in range(num_layers):
        counts_d_e = torch.zeros((num_domains, num_experts), dtype=torch.float32)
        counts_d = torch.zeros(num_domains, dtype=torch.float32)

        for record in records:
            router_logits = record["correct_router_logits_by_layer"][layer_idx]
            probs = torch.softmax(router_logits, dim=-1)
            topk_experts = torch.topk(probs, k=min(top_k, num_experts), dim=-1).indices
            domain_idx = domain_to_idx[record["benchmark"]]
            domain_labels = torch.full(
                topk_experts.shape[:2],
                domain_idx,
                dtype=torch.long,
            )

            for batch_idx in range(topk_experts.shape[0]):
                for token_idx in range(topk_experts.shape[1]):
                    token_domain = int(domain_labels[batch_idx, token_idx].item())
                    experts = topk_experts[batch_idx, token_idx]
                    for expert_idx in experts.tolist():
                        counts_d_e[token_domain, int(expert_idx)] += 1.0
                    counts_d[token_domain] += 1.0

        spec = counts_d_e / (counts_d.unsqueeze(-1) + 1e-9)
        specialization_by_layer.append(spec)

    return domains, specialization_by_layer


def plot_domain_specialization(records, top_k=DOMAIN_SPECIALIZATION_TOP_K):
    domains, specialization_by_layer = compute_domain_specialization(records, top_k=top_k)
    num_layers = len(specialization_by_layer)
    num_experts = specialization_by_layer[0].shape[-1]
    uniform = float(top_k) / float(num_experts)

    for layer_idx, spec in enumerate(specialization_by_layer):
        fig, ax = plt.subplots(figsize=(8, 4))
        expert_axis = np.arange(num_experts)
        for domain_idx, domain in enumerate(domains):
            ax.plot(expert_axis, spec[domain_idx].tolist(), marker="o", linewidth=1.5, label=domain)
        ax.axhline(uniform, color="#b24c2f", linestyle="--", linewidth=1.5, label="uniform")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Domain specialization")
        ax.set_title(f"Domain Specialization - Layer {layer_idx}")
        ax.set_xticks(list(range(num_experts)))
        ax.legend(frameon=False)
        fig.savefig(OUTPUT_DIR / f"domain_specialization_layer_{layer_idx}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    mean_specialization = np.array(
        [[float(spec[domain_idx].mean().item()) for spec in specialization_by_layer] for domain_idx in range(len(domains))]
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    layer_axis = np.arange(num_layers)
    for domain_idx, domain in enumerate(domains):
        ax.plot(layer_axis, mean_specialization[domain_idx], marker="o", linewidth=1.5, label=domain)
    ax.axhline(uniform, color="#b24c2f", linestyle="--", linewidth=1.5, label="uniform")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean domain specialization")
    ax.set_title("Domain Specialization by Layer")
    ax.set_xticks(list(range(num_layers)))
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "domain_specialization_layerwise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "top_k": top_k,
        "uniform_reference": uniform,
        "aggregate_by_layer": {
            f"layer_{layer_idx}": {
                domain: specialization_by_layer[layer_idx][domain_idx].tolist()
                for domain_idx, domain in enumerate(domains)
            }
            for layer_idx in range(num_layers)
        },
        "mean_by_domain_over_layers": {
            domain: mean_specialization[domain_idx].tolist()
            for domain_idx, domain in enumerate(domains)
        },
    }


def bucket_performance_by_entropy(records):
    ordered = sorted(records, key=lambda item: item["entropy_mean"])
    buckets = np.array_split(np.arange(len(ordered)), 4)
    summary = []
    for bucket_id, indices in enumerate(buckets):
        if len(indices) == 0:
            continue
        losses = [ordered[idx]["pseudo_loss"] for idx in indices]
        entropies = [ordered[idx]["entropy_mean"] for idx in indices]
        summary.append(
            {
                "bucket": bucket_id,
                "mean_entropy": float(np.mean(entropies)),
                "mean_loss": float(np.mean(losses)),
            }
        )
    return summary


def evaluate_performance_vs_k(examples):
    metrics_by_k = []
    for top_k in TOP_K_VALUES:
        torch.manual_seed(17)
        model = FakeFlexOlmoModel(num_experts=7, num_experts_per_tok=top_k)
        records = collect_analysis_records(model, examples)
        accuracy = float(np.mean([record["is_correct"] for record in records]))
        mean_loss = float(np.mean([record["pseudo_loss"] for record in records]))
        metrics_by_k.append({"top_k": top_k, "accuracy": accuracy, "loss": mean_loss})

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot([item["top_k"] for item in metrics_by_k], [item["loss"] for item in metrics_by_k], marker="o", color="#b24c2f")
    ax1.set_xlabel("top-k active experts")
    ax1.set_ylabel("Pseudo-loss", color="#b24c2f")
    ax1.tick_params(axis="y", labelcolor="#b24c2f")

    ax2 = ax1.twinx()
    ax2.plot([item["top_k"] for item in metrics_by_k], [item["accuracy"] for item in metrics_by_k], marker="s", color="#2f6db2")
    ax2.set_ylabel("Accuracy", color="#2f6db2")
    ax2.tick_params(axis="y", labelcolor="#2f6db2")
    ax1.set_title("Performance vs Number of Active Experts")
    fig.savefig(OUTPUT_DIR / "performance_vs_k.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics_by_k


def compute_routing_stability(records):
    by_domain = defaultdict(list)
    for record in records:
        by_domain[record["benchmark"]].append(record)

    stability = {}
    for domain, domain_records in by_domain.items():
        if len(domain_records) < 2:
            stability[domain] = 1.0
            continue

        pair_scores = []
        for left_record, right_record in combinations(domain_records, 2):
            left = top1_token_assignments(left_record["correct_router_logits_by_layer"][0]).reshape(-1)
            right = top1_token_assignments(right_record["correct_router_logits_by_layer"][0]).reshape(-1)
            length = min(left.shape[0], right.shape[0])
            agreement = (left[:length] == right[:length]).float().mean().item()
            pair_scores.append(agreement)
        stability[domain] = float(np.mean(pair_scores))

    return stability


def save_metrics_json(records, usage, load_balance_cv, corrected_coactivation, layerwise_coactivation, domain_specialization, expert_tokens, entropy_curve_mean, performance_vs_k, routing_stability):
    num_experts = records[0]["correct_router_logits_by_layer"][0].shape[-1]
    first_layer_logits = torch.cat(
        [record["correct_router_logits_by_layer"][0].reshape(-1, num_experts) for record in records],
        dim=0,
    ).unsqueeze(0)
    metrics = compute_all_metrics(first_layer_logits, top_k=2)
    top_expert_dominance = float(usage.max().item())
    bucketed_loss = bucket_performance_by_entropy(records)

    payload = {
        "summary": {
            "entropy_mean": float(metrics["entropy_mean"].item()),
            "entropy_curve_mean": entropy_curve_mean,
            "load_balance_cv": load_balance_cv,
            "offdiag_ratio": float(metrics["offdiag_ratio"]),
            "top_expert_dominance": top_expert_dominance,
            "load_balance_score": float(load_balance(torch.softmax(first_layer_logits, dim=-1))),
        },
        "usage": usage.tolist(),
        "coactivation_matrix": corrected_coactivation.tolist(),
        "layerwise_coactivation_matrices": [
            layer_matrix.tolist() for layer_matrix in layerwise_coactivation
        ],
        "domain_specialization": domain_specialization,
        "expert_token_specialization": expert_tokens,
        "performance_vs_k": performance_vs_k,
        "entropy_loss_buckets": bucketed_loss,
        "routing_stability": routing_stability,
    }

    with (OUTPUT_DIR / "paper_style_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    examples = load_jsonl(DATASET_PATH)
    model = FakeFlexOlmoModel(num_experts=7)
    records = collect_analysis_records(model, examples)

    plot_routing_probability_distributions(records)
    usage, load_balance_cv = plot_usage_distribution(records)
    entropy_curve_mean = plot_entropy_views(records)
    corrected_coactivation, layerwise_coactivation = plot_corrected_coactivation(records)
    expert_tokens = save_expert_token_specialization(records)
    domain_specialization = plot_domain_specialization(records)
    performance_vs_k = evaluate_performance_vs_k(examples)
    routing_stability = compute_routing_stability(records)
    save_metrics_json(
        records=records,
        usage=usage,
        load_balance_cv=load_balance_cv,
        corrected_coactivation=corrected_coactivation,
        layerwise_coactivation=layerwise_coactivation,
        domain_specialization=domain_specialization,
        expert_tokens=expert_tokens,
        entropy_curve_mean=entropy_curve_mean,
        performance_vs_k=performance_vs_k,
        routing_stability=routing_stability,
    )


if __name__ == "__main__":
    main()


TARGET_LAYER_ID = 0
VOCAB_SPECIALIZATION_TOP_K = 2


def _as_layerwise_router_logits(router_logits):
    if isinstance(router_logits, (list, tuple)):
        return torch.stack(tuple(router_logits), dim=0)
    if router_logits.ndim == 3:
        return router_logits.unsqueeze(0)
    return router_logits


def compute_vocabulary_specialization(router_logits, input_ids, top_k=VOCAB_SPECIALIZATION_TOP_K):
    router_logits = _as_layerwise_router_logits(router_logits)
    num_layers, _batch_size, _seq_len, num_experts = router_logits.shape
    vocab_size = int(input_ids.max().item()) + 1 if input_ids.numel() else 1

    specialization_per_layer = []
    specialization_per_expert_per_layer = []
    specialization_matrix_per_layer = []

    for layer_idx in range(num_layers):
        router_logits_layer = router_logits[layer_idx]
        probabilities = torch.softmax(router_logits_layer, dim=-1)
        topk_indices = torch.topk(probabilities, k=min(top_k, num_experts), dim=-1).indices

        token_expert_counts = torch.zeros((vocab_size, num_experts), dtype=torch.float32)
        token_totals = torch.zeros(vocab_size, dtype=torch.float32)

        flat_input_ids = input_ids.reshape(-1)
        flat_topk_indices = topk_indices.reshape(-1, topk_indices.shape[-1])

        for token_id, expert_ids in zip(flat_input_ids.tolist(), flat_topk_indices.tolist()):
            token_totals[token_id] += float(len(expert_ids))
            for expert_idx in expert_ids:
                token_expert_counts[token_id, int(expert_idx)] += 1.0

        valid_token_mask = token_totals > 0
        specialization_matrix = torch.zeros_like(token_expert_counts)
        specialization_matrix[valid_token_mask] = (
            token_expert_counts[valid_token_mask]
            / token_totals[valid_token_mask].unsqueeze(-1)
        )

        specialization_by_expert = specialization_matrix[valid_token_mask].mean(dim=0)
        specialization_per_expert_per_layer.append(specialization_by_expert.tolist())
        specialization_per_layer.append(float(specialization_by_expert.mean().item()))
        specialization_matrix_per_layer.append(specialization_matrix.tolist())

    return {
        "specialization_per_layer": specialization_per_layer,
        "specialization_per_expert_per_layer": specialization_per_expert_per_layer,
        "specialization_matrix_per_layer": specialization_matrix_per_layer,
    }


def _build_vocab_from_records(records):
    token_to_id = {}
    for record in records:
        for token in record["correct_tokens"]:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
    return token_to_id


def _plot_vocabulary_specialization(results):
    layer_values = results["specialization_per_layer"]
    expert_values = results["specialization_per_expert_per_layer"]
    target_layer_id = min(TARGET_LAYER_ID, len(expert_values) - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(layer_values)), layer_values, marker="o", color="#2f6db2")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average specialization")
    ax.set_title("Vocabulary Specialization by Layer")
    fig.savefig(OUTPUT_DIR / "vocab_specialization_layerwise.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(expert_values[target_layer_id])), expert_values[target_layer_id], color="#2f6db2")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Specialization")
    ax.set_title(f"Vocabulary Specialization by Expert - Layer {target_layer_id}")
    fig.savefig(
        OUTPUT_DIR / f"vocab_specialization_expert_layer_{target_layer_id}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def run_vocabulary_specialization_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    examples = load_jsonl(DATASET_PATH)
    model = FakeFlexOlmoModel(num_experts=7)
    records = collect_analysis_records(model, examples)

    token_to_id = _build_vocab_from_records(records)
    num_layers = len(records[0]["correct_router_logits_by_layer"])
    input_id_batches = []
    layer_router_logits_batches = [[] for _ in range(num_layers)]

    for record in records:
        input_ids = torch.tensor(
            [[token_to_id[token] for token in record["correct_tokens"]]],
            dtype=torch.long,
        )
        input_id_batches.append(input_ids)
        for layer_idx, layer_router_logits in enumerate(record["correct_router_logits_by_layer"]):
            layer_router_logits_batches[layer_idx].append(layer_router_logits)

    stacked_input_ids = torch.cat(input_id_batches, dim=1) if input_id_batches else torch.zeros((1, 0), dtype=torch.long)
    stacked_router_logits = torch.stack(
        [torch.cat(layer_batches, dim=1) for layer_batches in layer_router_logits_batches],
        dim=0,
    )

    results = compute_vocabulary_specialization(
        stacked_router_logits,
        stacked_input_ids,
        top_k=VOCAB_SPECIALIZATION_TOP_K,
    )
    _plot_vocabulary_specialization(results)
    return results
