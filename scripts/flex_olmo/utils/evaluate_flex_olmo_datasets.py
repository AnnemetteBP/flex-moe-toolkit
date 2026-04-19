from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
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

from flex_moe_toolkit.pipelines.flex_olmo_eval import (
    build_run_specs,
    evaluate_dataset_across_runs,
    load_jsonl_records,
    save_dataset_run_outputs,
    sanitize_name,
)
from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a real FlexOlmo checkpoint across one or more JSONL eval datasets."
    )
    parser.add_argument("--model-path", required=True, help="Path or HF identifier for the FlexOlmo checkpoint.")
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer path or HF identifier. Defaults to --model-path.",
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
        default=str(PROJECT_ROOT / "outputs" / "flex_olmo" / "eval_runs"),
        help="Directory where per-dataset/per-run outputs will be written.",
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
        help="Torch dtype used when loading the checkpoint.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokenized length per scored prompt+continuation sequence.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Expert index treated as the public expert for `public_only` runs.",
    )
    parser.add_argument(
        "--combined-active-experts",
        default="2,4,7",
        help="Comma-separated active expert counts for combined runs.",
    )
    parser.add_argument(
        "--expert-order",
        help=(
            "Optional comma-separated expert ordering used for combined runs. "
            "Defaults to `0,1,2,...`."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Optional cap per dataset for smoke-testing before launching a full remote GPU run.",
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


def load_model_and_tokenizer(args, device: torch.device):
    dtype = parse_dtype(args.dtype)
    model = FlexOlmoForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer_with_known_fixes(args.tokenizer_path or args.model_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer must define either `pad_token_id` or `eos_token_id` so batched scoring can pad inputs."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    args = parse_args()
    device = resolve_device(args.device)
    dataset_paths = collect_dataset_paths(args)
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    expert_order = None
    if args.expert_order:
        expert_order = tuple(int(part.strip()) for part in args.expert_order.split(",") if part.strip())

    combined_counts = tuple(
        int(part.strip()) for part in args.combined_active_experts.split(",") if part.strip()
    )
    run_specs = build_run_specs(
        num_experts=model.config.num_experts,
        public_expert_idx=args.public_expert_idx,
        combined_active_counts=combined_counts,
        include_individual_experts=True,
        expert_order=expert_order,
    )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    overall_manifest = {
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path or args.model_path,
        "model_impl_path": str(inspect.getsourcefile(FlexOlmoForCausalLM)),
        "device": str(device),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "public_expert_idx": args.public_expert_idx,
        "run_labels": [run_spec.label for run_spec in run_specs],
        "datasets": {},
    }

    print(f"Using FlexOlmo implementation from {overall_manifest['model_impl_path']}")
    print(
        "Configured eval runs: "
        + ", ".join(run_spec.label for run_spec in run_specs)
    )

    for dataset_path in dataset_paths:
        examples = load_jsonl_records(dataset_path)
        if args.max_examples is not None:
            examples = examples[: args.max_examples]
        dataset_name = sanitize_name(dataset_path.stem)
        run_results = evaluate_dataset_across_runs(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            run_specs=run_specs,
            max_length=args.max_length,
            device=device,
            context={
                "model_path": args.model_path,
                "model_name": Path(args.model_path).name or args.model_path,
            },
        )
        manifest = save_dataset_run_outputs(
            output_root=output_root,
            dataset_name=dataset_name,
            run_results=run_results,
        )
        overall_manifest["datasets"][dataset_name] = {
            "source_path": str(dataset_path.resolve()),
            "num_examples": len(examples),
            "runs": manifest,
        }
        print(
            f"Finished dataset `{dataset_name}` from {dataset_path} on {device}; "
            f"saved {len(manifest)} run outputs under {output_root / dataset_name}"
        )

    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(overall_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote overall manifest to {manifest_path}")


if __name__ == "__main__":
    main()
