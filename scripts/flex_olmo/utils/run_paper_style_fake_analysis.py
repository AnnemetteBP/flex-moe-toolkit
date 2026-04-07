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


def plot_domain_expert_usage(records):
    num_layers = len(records[0]["correct_router_logits_by_layer"])
    num_experts = records[0]["correct_router_logits_by_layer"][0].shape[-1]
    domains = sorted({record["benchmark"] for record in records})
    matrix = np.zeros((len(domains), num_experts), dtype=float)
    layerwise_matrices = np.zeros((num_layers, len(domains), num_experts), dtype=float)

    for row_idx, domain in enumerate(domains):
        usage_counts = torch.zeros(num_experts, dtype=torch.float32)
        layerwise_usage_counts = [torch.zeros(num_experts, dtype=torch.float32) for _ in range(num_layers)]
        for record in records:
            if record["benchmark"] == domain:
                usage_counts += top1_usage_counts(record["correct_router_logits_by_layer"][0])
                for layer_idx, layer_router_logits in enumerate(record["correct_router_logits_by_layer"]):
                    layerwise_usage_counts[layer_idx] += top1_usage_counts(layer_router_logits)
        usage = usage_counts / usage_counts.sum()
        matrix[row_idx] = usage.numpy()
        for layer_idx, layer_usage_counts in enumerate(layerwise_usage_counts):
            layer_usage = layer_usage_counts / layer_usage_counts.sum()
            layerwise_matrices[layer_idx, row_idx] = layer_usage.numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(matrix, cmap="viridis", ax=ax, yticklabels=domains)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Domain")
    ax.set_title("Domain Expert Usage")
    fig.savefig(OUTPUT_DIR / "domain_expert_usage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    expert_axis = np.arange(num_experts, dtype=float)
    fig, axes = plt.subplots(num_layers, 1, figsize=(9, 2.5 * num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]

    ridge_scale = 8.0
    ridge_offset = 1.0
    colors = sns.color_palette("viridis", n_colors=len(domains))

    for layer_idx, ax in enumerate(axes):
        for domain_idx, domain in enumerate(domains):
            baseline = domain_idx * ridge_offset
            values = layerwise_matrices[layer_idx, domain_idx] * ridge_scale
            ax.fill_between(
                expert_axis,
                baseline,
                baseline + values,
                color=colors[domain_idx],
                alpha=0.6,
            )
            ax.plot(expert_axis, baseline + values, color=colors[domain_idx], linewidth=1.5)

        ax.set_yticks([idx * ridge_offset for idx in range(len(domains))])
        ax.set_yticklabels(domains)
        ax.set_ylabel(f"Layer {layer_idx}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.2)

    axes[0].set_title("Layer-wise Domain Expert Usage Ridge Plot")
    axes[-1].set_xlabel("Expert")
    axes[-1].set_xticks(list(range(num_experts)))
    fig.savefig(OUTPUT_DIR / "domain_expert_usage_layerwise_ridge.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "aggregate": {domain: matrix[idx].tolist() for idx, domain in enumerate(domains)},
        "layerwise": {
            f"layer_{layer_idx}": {
                domain: layerwise_matrices[layer_idx, domain_idx].tolist()
                for domain_idx, domain in enumerate(domains)
            }
            for layer_idx in range(num_layers)
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


def save_metrics_json(records, usage, load_balance_cv, corrected_coactivation, layerwise_coactivation, domain_usage, expert_tokens, entropy_curve_mean, performance_vs_k, routing_stability):
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
        "domain_expert_usage": domain_usage,
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
    domain_usage = plot_domain_expert_usage(records)
    performance_vs_k = evaluate_performance_vs_k(examples)
    routing_stability = compute_routing_stability(records)
    save_metrics_json(
        records=records,
        usage=usage,
        load_balance_cv=load_balance_cv,
        corrected_coactivation=corrected_coactivation,
        layerwise_coactivation=layerwise_coactivation,
        domain_usage=domain_usage,
        expert_tokens=expert_tokens,
        entropy_curve_mean=entropy_curve_mean,
        performance_vs_k=performance_vs_k,
        routing_stability=routing_stability,
    )


if __name__ == "__main__":
    main()
