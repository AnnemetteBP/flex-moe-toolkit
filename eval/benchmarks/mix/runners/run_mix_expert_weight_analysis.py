from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from eval.benchmarks.mix.runners.run_mix_analysis import (
    load_allowed_model_names,
    parse_hidden_state_layers,
    resolve_device,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval_results" / "mix" / "expert_weight_analysis" / "a4"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "model_paths" / "all_models.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact standalone-expert weight analysis for selected checkpoints."
    )
    parser.add_argument("--model-path", help="Explicit model path or HF identifier.")
    parser.add_argument("--model-name", help="Model name from model_paths/all_models.txt.")
    parser.add_argument("--model-root", help="Directory containing model folders on UCloud.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--selected-layers", default="early_mid_late_last")
    parser.add_argument("--fingerprint-size", type=int, default=4096)
    parser.add_argument("--fingerprint-seed", type=int, default=17)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    return getattr(torch, dtype_name)


def resolve_model_path(args: argparse.Namespace) -> str:
    if args.model_path:
        return args.model_path
    if not args.model_name or not args.model_root:
        raise ValueError("Provide either --model-path or both --model-name and --model-root.")
    allowed_names = load_allowed_model_names(args.model_registry)
    if args.model_name not in allowed_names:
        raise ValueError(f"Model name `{args.model_name}` was not found in {args.model_registry}.")
    return str(Path(args.model_root) / args.model_name)


def resolved_model_name(args: argparse.Namespace, model_path: str) -> str:
    if args.model_name:
        return args.model_name
    return Path(model_path).name


def iter_decoder_layers(model):
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        raise ValueError("Model does not expose decoder layers via `.layers` or `.model.layers`.")
    return list(layers)


def extract_weight(module_or_tensor) -> torch.Tensor:
    if module_or_tensor is None:
        raise ValueError("Cannot extract weight from `None`.")
    if isinstance(module_or_tensor, torch.Tensor):
        return module_or_tensor.detach().cpu().float()
    if hasattr(module_or_tensor, "weight"):
        return module_or_tensor.weight.detach().cpu().float()
    raise ValueError(f"Unsupported weight source of type {type(module_or_tensor).__name__}.")


def stable_seed(component_name: str, shape: tuple[int, ...], base_seed: int) -> int:
    payload = f"{component_name}|{shape}|{base_seed}".encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()
    return int(digest[:8], 16)


def sampled_fingerprint(weights: torch.Tensor, component_name: str, sample_size: int, base_seed: int) -> np.ndarray:
    flat = weights.detach().cpu().flatten().float().numpy()
    if flat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    take = min(sample_size, flat.size)
    rng = np.random.default_rng(stable_seed(component_name, tuple(weights.shape), base_seed))
    if take == flat.size:
        indices = np.arange(flat.size)
    else:
        indices = np.sort(rng.choice(flat.size, size=take, replace=False))
    return flat[indices].astype(np.float32)


def summary_stats(weights: torch.Tensor) -> dict[str, float]:
    flat = weights.detach().cpu().flatten().float()
    return {
        "l2_norm": float(torch.linalg.vector_norm(flat, ord=2).item()),
        "mean_abs": float(flat.abs().mean().item()),
        "std": float(flat.std(unbiased=False).item()),
    }


def collect_component_weights(model, selected_layers: list[int]) -> dict[str, torch.Tensor]:
    components: dict[str, torch.Tensor] = {}
    embedding_module = model.get_input_embeddings()
    components["embedding"] = extract_weight(embedding_module)

    layers = iter_decoder_layers(model)
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        if not hasattr(layer, "mlp"):
            continue
        mlp = layer.mlp
        if hasattr(mlp, "gate_up_proj"):
            components[f"layer_{layer_idx}_gate_up_proj"] = extract_weight(mlp.gate_up_proj)
        if hasattr(mlp, "up_proj"):
            components[f"layer_{layer_idx}_up_proj"] = extract_weight(mlp.up_proj)
        if hasattr(mlp, "gate_proj"):
            components[f"layer_{layer_idx}_gate_proj"] = extract_weight(mlp.gate_proj)
        if hasattr(mlp, "down_proj"):
            components[f"layer_{layer_idx}_down_proj"] = extract_weight(mlp.down_proj)
    return components


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args)
    model_name = resolved_model_name(args, model_path)
    device = resolve_device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=parse_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError("Model config must define `num_hidden_layers` for standalone expert weight analysis.")
    selected_layers = parse_hidden_state_layers(args.selected_layers, num_hidden_layers=num_hidden_layers)
    model_output_root = Path(args.output_root) / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    component_weights = collect_component_weights(model, selected_layers)
    summary_rows: list[dict[str, Any]] = []
    fingerprint_arrays: dict[str, np.ndarray] = {}

    for component_name, weights in sorted(component_weights.items()):
        stats = summary_stats(weights)
        fingerprint_arrays[component_name] = sampled_fingerprint(
            weights,
            component_name=component_name,
            sample_size=args.fingerprint_size,
            base_seed=args.fingerprint_seed,
        )
        summary_rows.append(
            {
                "model_name": model_name,
                "component_name": component_name,
                "shape": "x".join(str(dim) for dim in weights.shape),
                "numel": int(weights.numel()),
                "l2_norm": stats["l2_norm"],
                "mean_abs": stats["mean_abs"],
                "std": stats["std"],
                "fingerprint_size": int(fingerprint_arrays[component_name].shape[0]),
            }
        )

    write_csv(model_output_root / "expert_weight_summary.csv", summary_rows)
    np.savez_compressed(model_output_root / "expert_weight_fingerprints.npz", **fingerprint_arrays)
    manifest = {
        "model_name": model_name,
        "model_path": model_path,
        "selected_layers": selected_layers,
        "fingerprint_size": args.fingerprint_size,
        "fingerprint_seed": args.fingerprint_seed,
        "summary_path": str(model_output_root / "expert_weight_summary.csv"),
        "fingerprints_path": str(model_output_root / "expert_weight_fingerprints.npz"),
        "components": sorted(component_weights.keys()),
    }
    (model_output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote standalone expert weight analysis to {model_output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
