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

from flex_moe_toolkit.pipelines.flex_olmo_saturation import (
    compute_example_router_saturation,
    infer_model_metadata,
    summarize_router_saturation,
)
from flex_moe_toolkit.pipelines.flex_olmo_eval import load_jsonl_records, sanitize_name
from flex_moe_toolkit.utils import load_tokenizer_with_known_fixes
from flex_moe_toolkit.utils.jsonl import to_jsonable, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare FlexOlmo checkpoints against a final checkpoint using router saturation."
    )
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_paths",
        action="append",
        default=[],
        help="Checkpoint path or HF identifier to compare against the final checkpoint. Repeat for multiple checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-path-file",
        help="Optional text file listing one checkpoint path / HF identifier per line.",
    )
    parser.add_argument(
        "--final-checkpoint-path",
        required=True,
        help="Path or HF identifier for the final checkpoint T.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer path or HF identifier. Defaults to --final-checkpoint-path.",
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
        help="Optional directory containing JSONL eval datasets. All `*.jsonl` files will be compared.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "flex_olmo" / "router_saturation"),
        help="Directory where per-checkpoint saturation outputs will be written.",
    )
    parser.add_argument(
        "--summary-jsonl",
        help="Optional explicit path for the consolidated saturation summary JSONL.",
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
        help="Torch dtype used when loading checkpoints.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum tokenized prompt length per example.",
    )
    parser.add_argument(
        "--top-k",
        help=(
            "Optional top-k override. Provide a single integer like `8` or a comma-separated "
            "list like `1,2,4,8`. Defaults to the final checkpoint's configured routing k."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Optional cap per dataset for smoke-testing.",
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


def collect_checkpoint_paths(args) -> list[str]:
    checkpoint_paths = list(args.checkpoint_paths)
    if args.checkpoint_path_file:
        for line in Path(args.checkpoint_path_file).read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                checkpoint_paths.append(candidate)

    unique = []
    seen = set()
    for path in checkpoint_paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    if not unique:
        raise ValueError("Provide at least one `--checkpoint-path` or a `--checkpoint-path-file`.")
    return unique


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


def parse_top_k(top_k_arg):
    if top_k_arg is None:
        return None
    parts = [part.strip() for part in str(top_k_arg).split(",") if part.strip()]
    values = [int(part) for part in parts]
    if len(values) == 1:
        return values[0]
    return values


def load_model(model_path: str, device: torch.device, dtype_name: str):
    model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=parse_dtype(dtype_name))
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_paths = collect_checkpoint_paths(args)
    dataset_paths = collect_dataset_paths(args)
    top_k = parse_top_k(args.top_k)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_jsonl = (
        Path(args.summary_jsonl)
        if args.summary_jsonl
        else output_root / "router_saturation_summary.jsonl"
    )

    final_model = load_model(args.final_checkpoint_path, device=device, dtype_name=args.dtype)
    tokenizer = load_tokenizer_with_known_fixes(args.tokenizer_path or args.final_checkpoint_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest = {
        "final_checkpoint_path": args.final_checkpoint_path,
        "model_impl_path": str(inspect.getsourcefile(FlexOlmoForCausalLM)),
        "device": str(device),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "top_k": top_k,
        "datasets": [str(path.resolve()) for path in dataset_paths],
        "comparisons": {},
    }
    summary_records = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_model = load_model(checkpoint_path, device=device, dtype_name=args.dtype)
        checkpoint_metadata = infer_model_metadata(checkpoint_path)
        comparison_output_root = output_root / sanitize_name(checkpoint_metadata["model_name"])
        comparison_output_root.mkdir(parents=True, exist_ok=True)

        manifest["comparisons"][checkpoint_path] = {
            "metadata": checkpoint_metadata,
            "datasets": {},
        }

        for dataset_path in dataset_paths:
            examples = load_jsonl_records(dataset_path)
            if args.max_examples is not None:
                examples = examples[: args.max_examples]

            example_records = [
                {
                    **checkpoint_metadata,
                    "final_checkpoint_path": args.final_checkpoint_path,
                    **compute_example_router_saturation(
                        checkpoint_model=checkpoint_model,
                        final_model=final_model,
                        tokenizer=tokenizer,
                        example=example,
                        example_index=example_index,
                        device=device,
                        max_length=args.max_length,
                        top_k=top_k,
                    ),
                }
                for example_index, example in enumerate(examples)
            ]

            summary_record = {
                **checkpoint_metadata,
                "final_checkpoint_path": args.final_checkpoint_path,
                "dataset_path": str(dataset_path.resolve()),
                "dataset_name": sanitize_name(dataset_path.stem),
                **summarize_router_saturation(example_records),
            }
            summary_records.append(summary_record)

            dataset_dir = comparison_output_root / sanitize_name(dataset_path.stem)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            examples_path = dataset_dir / "router_saturation_examples.jsonl"
            summary_path = dataset_dir / "router_saturation_summary.jsonl"
            write_jsonl(example_records, examples_path)
            write_jsonl([summary_record], summary_path)

            manifest["comparisons"][checkpoint_path]["datasets"][sanitize_name(dataset_path.stem)] = {
                "source_path": str(dataset_path.resolve()),
                "num_examples": len(examples),
                "examples_path": str(examples_path),
                "summary_path": str(summary_path),
            }

        del checkpoint_model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(summary_records, summary_jsonl)
    manifest_path = output_root / "router_saturation_manifest.json"
    manifest_path.write_text(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
