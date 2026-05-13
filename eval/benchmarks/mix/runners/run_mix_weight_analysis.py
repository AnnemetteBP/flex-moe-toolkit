from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import FlexOlmoForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.pipelines.flex_olmo_weights import analyze_flex_olmo_weights
from eval.benchmarks.mix.runners.run_mix_analysis import (
    load_allowed_model_names,
    parse_dtype,
    resolve_device,
)
from eval.benchmarks.mix.runners.run_mix_router_direction import parse_decoder_layers


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "weight_analysis" / "a4"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact FlexOlmo weight analysis for router rows and expert MLP weights."
    )
    parser.add_argument("--model-path", help="Explicit path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--selected-layers", default="early_mid_late_last")
    parser.add_argument("--public-expert-idx", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")
    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")
    return str(Path(args.model_root) / args.model_name)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_serializable(obj):
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)
    model_name = args.model_name or Path(model_path).name
    device = resolve_device(args.device)

    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=parse_dtype(args.dtype))
    model.to(device)
    model.eval()

    analysis = analyze_flex_olmo_weights(model, public_expert_idx=args.public_expert_idx)
    selected_layers = parse_decoder_layers(args.selected_layers, int(model.config.num_hidden_layers))
    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    matrix_arrays: dict[str, np.ndarray] = {}
    summary_rows: list[dict] = []
    for layer_record in analysis["layer_weight_analysis"]:
        layer_idx = int(layer_record["layer_idx"])
        summary_rows.append(
            {
                "model_name": model_name,
                "layer": layer_idx,
                "router_mean_offdiag_similarity": float(
                    layer_record["router_similarity_summary"]["mean_offdiag_similarity"]
                ),
                "router_max_offdiag_similarity": float(
                    layer_record["router_similarity_summary"]["max_offdiag_similarity"]
                ),
                "router_mean_norm": float(np.mean(layer_record["router_weight_norms"])),
                "router_mean_abs": float(np.mean(layer_record["router_weight_mean_abs"])),
                "router_mean_public_distance": float(np.mean(layer_record["router_weight_public_distance"])),
                "gate_up_mean_offdiag_similarity": float(
                    layer_record["gate_up_proj_similarity_summary"]["mean_offdiag_similarity"]
                ),
                "gate_up_max_offdiag_similarity": float(
                    layer_record["gate_up_proj_similarity_summary"]["max_offdiag_similarity"]
                ),
                "gate_up_mean_norm": float(np.mean(layer_record["gate_up_proj_norms"])),
                "gate_up_mean_abs": float(np.mean(layer_record["gate_up_proj_mean_abs"])),
                "gate_up_mean_public_distance": float(np.mean(layer_record["gate_up_proj_public_distance"])),
                "down_proj_mean_offdiag_similarity": (
                    float(layer_record["down_proj_similarity_summary"]["mean_offdiag_similarity"])
                    if "down_proj_similarity_summary" in layer_record
                    else np.nan
                ),
                "down_proj_max_offdiag_similarity": (
                    float(layer_record["down_proj_similarity_summary"]["max_offdiag_similarity"])
                    if "down_proj_similarity_summary" in layer_record
                    else np.nan
                ),
                "down_proj_mean_norm": (
                    float(np.mean(layer_record["down_proj_norms"])) if "down_proj_norms" in layer_record else np.nan
                ),
                "down_proj_mean_abs": (
                    float(np.mean(layer_record["down_proj_mean_abs"]))
                    if "down_proj_mean_abs" in layer_record
                    else np.nan
                ),
                "down_proj_mean_public_distance": (
                    float(np.mean(layer_record["down_proj_public_distance"]))
                    if "down_proj_public_distance" in layer_record
                    else np.nan
                ),
            }
        )

        matrix_arrays[f"layer_{layer_idx}_router_similarity"] = np.asarray(layer_record["router_similarity"], dtype=np.float32)
        matrix_arrays[f"layer_{layer_idx}_router_norms"] = np.asarray(layer_record["router_weight_norms"], dtype=np.float32)
        matrix_arrays[f"layer_{layer_idx}_gate_up_similarity"] = np.asarray(
            layer_record["gate_up_proj_similarity"], dtype=np.float32
        )
        matrix_arrays[f"layer_{layer_idx}_gate_up_public_distance"] = np.asarray(
            layer_record["gate_up_proj_public_distance"], dtype=np.float32
        )
        if "down_proj_similarity" in layer_record:
            matrix_arrays[f"layer_{layer_idx}_down_proj_similarity"] = np.asarray(
                layer_record["down_proj_similarity"], dtype=np.float32
            )
            matrix_arrays[f"layer_{layer_idx}_down_proj_public_distance"] = np.asarray(
                layer_record["down_proj_public_distance"], dtype=np.float32
            )

    write_csv(model_output_root / "weight_analysis_summary.csv", summary_rows)
    np.savez_compressed(model_output_root / "weight_analysis_matrices.npz", **matrix_arrays)

    run_manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "selected_layers": selected_layers,
        "public_expert_idx": args.public_expert_idx,
        "num_layers": analysis["num_layers"],
        "num_experts": analysis["num_experts"],
        "weight_analysis_summary_path": str(model_output_root / "weight_analysis_summary.csv"),
        "weight_analysis_matrices_path": str(model_output_root / "weight_analysis_matrices.npz"),
        "aggregate_summary": analysis["summary"],
    }
    (model_output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (model_output_root / "weight_analysis.json").write_text(
        json.dumps(to_serializable(analysis), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote weight analysis to {model_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
