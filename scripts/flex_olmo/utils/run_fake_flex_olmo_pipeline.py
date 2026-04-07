from __future__ import annotations

import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

from fake_test_models.fake_flex_olmo import FakeFlexOlmoModel, build_fake_inputs
from flex_moe_toolkit.adapters.flex_olmo import FlexOlmoAdapter, iter_flex_olmo_layers
from flex_moe_toolkit.core.metrics import load_balance, routing_entropy
from flex_moe_toolkit.pipelines.flex_olmo import (
    analyze_flex_olmo_routing,
    restricted_expert_mode,
)
from flex_moe_toolkit.prev_analysis.plots import plot_expert_combination_upset
from flex_moe_toolkit.utils.hooks import register_router_hook
from flex_moe_toolkit.utils.jsonl import write_jsonl
from flex_moe_toolkit.utils.router_activity import (
    build_layerwise_upset_data_from_router_logits,
    count_token_expert_combinations_by_layer_from_router_logits,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "flex_olmo"
COMBINED_PATH = OUTPUT_ROOT / "combined_flex" / "fake_flex_olmo_results.jsonl"
EXPERTS_PATH = OUTPUT_ROOT / "flex_experts" / "fake_flex_olmo_results.jsonl"
LAYERWISE_UPSET_PATH = OUTPUT_ROOT / "combined_flex" / "fake_flex_olmo_layerwise_upset.png"


def architecture_record(model):
    layers = list(iter_flex_olmo_layers(model))
    return {
        "record_type": "architecture",
        "scope": "combined_flex",
        "num_layers": len(layers),
        "num_experts": model.config.num_experts,
        "num_experts_per_tok": model.config.num_experts_per_tok,
        "hidden_size": model.config.hidden_size,
        "layer_paths": [f"layers.{idx}.mlp" for idx, _layer in enumerate(layers)],
        "router_paths": [f"layers.{idx}.mlp.gate" for idx, _layer in enumerate(layers)],
        "expert_container_paths": [
            f"layers.{idx}.mlp.experts" for idx, _layer in enumerate(layers)
        ],
    }


def hook_record(model, inputs):
    captured = []

    def hook_fn(_module, _args, output):
        probs = output[0] if isinstance(output, tuple) else output
        captured.append(
            {
                "shape": list(probs.shape),
                "mean_probability": float(probs.mean().item()),
                "max_probability": float(probs.max().item()),
            }
        )

    handles = register_router_hook(model, hook_fn)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {
        "record_type": "hooks",
        "scope": "combined_flex",
        "num_hook_calls": len(captured),
        "captures": captured,
    }


def adapter_record(model, inputs):
    adapter = FlexOlmoAdapter()
    router_logits = adapter.get_router_logits(model, inputs)
    router_probs = adapter.get_router_probs(model, inputs)
    routing = analyze_flex_olmo_routing(model, inputs)

    return {
        "record_type": "adapters",
        "scope": "combined_flex",
        "adapter": adapter.__class__.__name__,
        "num_router_layers": len(router_logits),
        "router_logit_shapes": [list(layer.shape) for layer in router_logits],
        "router_prob_shapes": [list(layer.shape) for layer in router_probs],
        "topk_experts": routing["topk_experts"],
    }


def metrics_record(model, inputs):
    adapter = FlexOlmoAdapter()
    router_logits = adapter.get_router_logits(model, inputs)
    router_probs = adapter.get_router_probs(model, inputs)
    routing = analyze_flex_olmo_routing(model, inputs)

    return {
        "record_type": "metrics",
        "scope": "combined_flex",
        "entropy": routing_entropy(router_logits),
        "load_balance": load_balance(router_probs),
        "expert_usage": routing["expert_usage"],
        "layer_expert_matrix": routing["layer_expert_matrix"],
    }


def layerwise_upset_record_and_plot(model, inputs):
    adapter = FlexOlmoAdapter()
    router_logits = adapter.get_router_logits(model, inputs)
    sample_labels = [f"sample_{idx}" for idx in range(inputs["input_ids"].shape[0])]
    upset_data = build_layerwise_upset_data_from_router_logits(
        router_logits_by_layer=router_logits,
        top_k=model.config.num_experts_per_tok,
        sample_labels=sample_labels,
    )
    per_layer_combination_counts = count_token_expert_combinations_by_layer_from_router_logits(
        router_logits_by_layer=router_logits,
        top_k=model.config.num_experts_per_tok,
    )
    plot_paths = []
    serialized_counts = {}

    for layer_idx, combination_counts in enumerate(per_layer_combination_counts):
        plot_path = OUTPUT_ROOT / "combined_flex" / f"fake_flex_olmo_layer_{layer_idx}_token_upset.png"
        plot_expert_combination_upset(
            combination_counts=combination_counts,
            path=plot_path,
            title=f"Fake FlexOlmo Token-Level Expert Combinations (Layer {layer_idx})",
        )
        plot_paths.append(str(plot_path))
        serialized_counts[f"layer_{layer_idx}"] = {
            ",".join(str(expert) for expert in combo): count
            for combo, count in sorted(combination_counts.items())
        }

    return {
        "record_type": "layerwise_upset",
        "scope": "combined_flex",
        "num_entries": len(upset_data),
        "upset_data": upset_data,
        "token_level_combination_counts_by_layer": serialized_counts,
        "plot_paths": plot_paths,
    }


def expert_records(model, inputs):
    records = []

    for expert_idx in range(model.config.num_experts):
        with restricted_expert_mode(model, allowed_experts=[expert_idx]):
            routing = analyze_flex_olmo_routing(model, inputs, top_k=1)

        records.append(
            {
                "record_type": "expert_metrics",
                "scope": "flex_experts",
                "expert_idx": expert_idx,
                "entropy": routing["entropy"],
                "load_balance": routing["load_balance"],
                "expert_usage": routing["expert_usage"],
                "layer_expert_matrix": routing["layer_expert_matrix"],
                "topk_experts": routing["topk_experts"],
            }
        )

    return records


def main():
    model = FakeFlexOlmoModel()
    inputs = build_fake_inputs(hidden_size=model.config.hidden_size)

    combined_records = [
        architecture_record(model),
        hook_record(model, inputs),
        adapter_record(model, inputs),
        metrics_record(model, inputs),
        layerwise_upset_record_and_plot(model, inputs),
    ]
    expert_mode_records = expert_records(model, inputs)

    write_jsonl(combined_records, COMBINED_PATH)
    write_jsonl(expert_mode_records, EXPERTS_PATH)

    print(f"Wrote {len(combined_records)} combined-model records to {COMBINED_PATH}")
    print(f"Wrote {len(expert_mode_records)} expert records to {EXPERTS_PATH}")
    print(f"Saved layerwise token-level upset examples under {OUTPUT_ROOT / 'combined_flex'}")


if __name__ == "__main__":
    main()
