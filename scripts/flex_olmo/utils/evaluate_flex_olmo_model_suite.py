from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import re
import sys

import torch
from transformers import FlexOlmoForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

from flex_moe_toolkit.core.routing_diagnostics import compute_offdiagonal_ratio
from flex_moe_toolkit.pipelines.flex_olmo_eval import (
    FlexOlmoEvalRunSpec,
    evaluate_dataset_across_runs,
    load_jsonl_records,
    save_dataset_run_outputs,
    sanitize_name,
)
from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes
from flex_moe_toolkit.utils.jsonl import to_jsonable, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run router-focused evaluation across many FlexOlmo checkpoints and "
            "write a consolidated JSONL summary for UCloud batch jobs."
        )
    )
    parser.add_argument(
        "--model-path",
        dest="model_paths",
        action="append",
        default=[],
        help="Path or HF identifier for a FlexOlmo checkpoint. Repeat for multiple models.",
    )
    parser.add_argument(
        "--model-path-file",
        help="Optional text file listing one model path / HF identifier per line.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer path or HF identifier. Defaults to each model path.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=[],
        help="Path to a JSONL dataset. Repeat for multiple datasets.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Optional directory containing JSONL eval datasets. All `*.jsonl` files will be evaluated.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "flex_olmo" / "router_suite"),
        help="Directory where per-model/per-dataset outputs will be written.",
    )
    parser.add_argument(
        "--summary-jsonl",
        help="Optional explicit path for the consolidated router suite summary JSONL.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: `auto`, `cpu`, `cuda`, or an explicit device like `cuda:0`.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype used when loading each checkpoint.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokenized length per scored prompt+continuation sequence.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Optional cap per dataset for smoke-testing before a full UCloud sweep.",
    )
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def collect_model_paths(args) -> list[str]:
    model_paths = list(args.model_paths)

    if args.model_path_file:
        for line in Path(args.model_path_file).read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                model_paths.append(candidate)

    seen = set()
    unique_model_paths = []
    for model_path in model_paths:
        if model_path in seen:
            continue
        seen.add(model_path)
        unique_model_paths.append(model_path)

    if not unique_model_paths:
        raise ValueError("Provide at least one `--model-path` or a `--model-path-file`.")

    return unique_model_paths


def collect_dataset_paths(args) -> list[Path]:
    dataset_paths = [Path(path) for path in args.datasets]
    if args.dataset_dir:
        dataset_paths.extend(sorted(Path(args.dataset_dir).glob("*.jsonl")))

    unique_paths = []
    seen = set()
    for path in dataset_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("Provide at least one `--dataset` or a `--dataset-dir` containing `*.jsonl` files.")

    return unique_paths


def infer_model_metadata(model_path: str) -> dict[str, object]:
    name = Path(model_path).name or model_path
    lower_name = name.lower()

    active_experts = None
    match = re.search(r"(?:^|[_-])a(\d+)(?:$|[_-])", lower_name)
    if match:
        active_experts = int(match.group(1))

    scale_b = None
    match = re.search(r"(\d+)b", lower_name)
    if match:
        scale_b = int(match.group(1))

    if re.search(r"(?:^|[_-])rt(?:$|[_-])", lower_name):
        variant = "rt"
    elif re.search(r"(?:^|[_-])v2(?:$|[_-])", lower_name):
        variant = "v2"
    elif re.search(r"(?:^|[_-])v1(?:$|[_-])", lower_name):
        variant = "v1"
    else:
        variant = "base"

    return {
        "model_name": name,
        "model_label": sanitize_name(name),
        "active_experts_variant": active_experts,
        "training_variant": variant,
        "scale_b": scale_b,
        "is_router_tuned": variant == "rt",
    }


def load_model_and_tokenizer(model_path: str, tokenizer_path: str | None, device: torch.device, dtype_name: str):
    dtype = parse_dtype(dtype_name)
    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer_with_known_fixes(tokenizer_path or model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must define either `pad_token_id` or `eos_token_id` so batched scoring can pad inputs."
            )
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_as_configured_run_spec(model) -> FlexOlmoEvalRunSpec:
    return FlexOlmoEvalRunSpec(
        label="as_configured",
        allowed_experts=tuple(range(int(model.config.num_experts))),
    )


def build_suite_summary_record(
    model_path: str,
    dataset_path: Path,
    output_root: Path,
    run_payload: dict,
    metadata: dict[str, object],
    max_examples: int | None,
) -> dict[str, object]:
    records = run_payload["records"]
    summaries = run_payload["summaries"]
    routing_aggregate = run_payload["routing_aggregate"]
    summary = summaries[0] if summaries else {}

    layer_coactivation_matrices = routing_aggregate["layer_coactivation_matrices"]
    layer_offdiag_ratios = [
        compute_offdiagonal_ratio(matrix)
        for matrix in layer_coactivation_matrices
        if matrix is not None
    ]
    mean_load_balance = (
        sum(float(record["load_balance"]) for record in records) / len(records)
        if records
        else 0.0
    )
    mean_entropy = (
        sum(float(record["entropy"]) for record in records) / len(records)
        if records
        else 0.0
    )

    return {
        "record_type": "router_suite_summary",
        "model_path": model_path,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_name": sanitize_name(dataset_path.stem),
        "run_label": run_payload["run_spec"].label,
        "num_examples": len(records),
        "max_examples": max_examples,
        "num_experts": len(run_payload["run_spec"].allowed_experts),
        "router_entropy_mean": mean_entropy,
        "router_load_balance_mean": mean_load_balance,
        "router_offdiag_ratio": float(routing_aggregate["offdiag_ratio"]),
        "router_usage": routing_aggregate["usage"],
        "router_num_active_experts": int((routing_aggregate["usage"] > 0).sum().item()),
        "layer_offdiag_ratios": layer_offdiag_ratios,
        "accuracy": summary.get("accuracy"),
        "mean_layer_iou": summary.get("mean_layer_iou"),
        "activated_expert_combinations": summary.get("activated_expert_combinations", {}),
        "layer_pattern_counts": summary.get("layer_pattern_counts", {}),
        "records_path": str(
            output_root
            / metadata["model_label"]
            / sanitize_name(dataset_path.stem)
            / run_payload["run_spec"].label
            / "eval_records.jsonl"
        ),
        "routing_analysis_path": str(
            output_root
            / metadata["model_label"]
            / sanitize_name(dataset_path.stem)
            / run_payload["run_spec"].label
            / "routing_analysis.jsonl"
        ),
        **metadata,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    model_paths = collect_model_paths(args)
    dataset_paths = collect_dataset_paths(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_summary_jsonl = (
        Path(args.summary_jsonl)
        if args.summary_jsonl
        else output_root / "router_suite_summary.jsonl"
    )

    suite_manifest = {
        "device": str(device),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "max_examples": args.max_examples,
        "models": {},
        "datasets": [str(path.resolve()) for path in dataset_paths],
    }
    suite_summary_records = []

    for model_path in model_paths:
        metadata = infer_model_metadata(model_path)
        model_output_root = output_root / metadata["model_label"]
        model_output_root.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            tokenizer_path=args.tokenizer_path,
            device=device,
            dtype_name=args.dtype,
        )
        model_impl_path = str(inspect.getsourcefile(FlexOlmoForCausalLM))
        run_spec = build_as_configured_run_spec(model)

        print(
            f"Evaluating {model_path} on {device} with run `{run_spec.label}` "
            f"using FlexOlmo from {model_impl_path}"
        )

        suite_manifest["models"][model_path] = {
            "metadata": metadata,
            "model_impl_path": model_impl_path,
            "datasets": {},
        }

        for dataset_path in dataset_paths:
            examples = load_jsonl_records(dataset_path)
            if args.max_examples is not None:
                examples = examples[: args.max_examples]

            dataset_name = sanitize_name(dataset_path.stem)
            run_results = evaluate_dataset_across_runs(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                run_specs=[run_spec],
                max_length=args.max_length,
                device=device,
                context={
                    "model_path": model_path,
                    "model_name": metadata["model_name"],
                    "model_label": metadata["model_label"],
                    "active_experts_variant": metadata["active_experts_variant"],
                    "training_variant": metadata["training_variant"],
                    "scale_b": metadata["scale_b"],
                    "is_router_tuned": metadata["is_router_tuned"],
                },
            )

            manifest = save_dataset_run_outputs(
                output_root=model_output_root,
                dataset_name=dataset_name,
                run_results=run_results,
            )
            run_payload = run_results[run_spec.label]
            suite_summary_record = build_suite_summary_record(
                model_path=model_path,
                dataset_path=dataset_path,
                output_root=output_root,
                run_payload=run_payload,
                metadata=metadata,
                max_examples=args.max_examples,
            )
            suite_summary_records.append(suite_summary_record)

            suite_manifest["models"][model_path]["datasets"][dataset_name] = {
                "source_path": str(dataset_path.resolve()),
                "num_examples": len(examples),
                "manifest": manifest,
            }

            print(
                f"Finished model `{metadata['model_name']}` on dataset `{dataset_name}`; "
                f"router entropy={suite_summary_record['router_entropy_mean']:.4f}, "
                f"offdiag={suite_summary_record['router_offdiag_ratio']:.4f}"
            )

        del model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(suite_summary_records, suite_summary_jsonl)
    manifest_path = output_root / "router_suite_manifest.json"
    manifest_path.write_text(json.dumps(to_jsonable(suite_manifest), indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote consolidated router suite summary to {suite_summary_jsonl}")
    print(f"Wrote suite manifest to {manifest_path}")


if __name__ == "__main__":
    main()
