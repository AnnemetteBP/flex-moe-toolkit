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

from flex_moe_toolkit.pipelines.flex_olmo_weights import analyze_flex_olmo_weights
from flex_moe_toolkit.utils.jsonl import to_jsonable, write_jsonl


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze static FlexOlmo router/expert weights across one or many checkpoints."
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
        "--output-jsonl",
        required=True,
        help="Path for the JSONL summary records.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional directory for detailed per-model JSON files.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device: `cpu`, `cuda`, or an explicit device like `cuda:0`.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Torch dtype used when loading each checkpoint.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Expert index treated as the public expert when computing distance profiles.",
    )
    return parser.parse_args()


def parse_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def collect_model_paths(args) -> list[str]:
    model_paths = list(args.model_paths)
    if args.model_path_file:
        for line in Path(args.model_path_file).read_text(encoding="utf-8").splitlines():
            candidate = line.strip()
            if candidate and not candidate.startswith("#"):
                model_paths.append(candidate)

    unique = []
    seen = set()
    for path in model_paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    if not unique:
        raise ValueError("Provide at least one `--model-path` or a `--model-path-file`.")
    return unique


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
        "model_path": model_path,
        "active_experts_variant": active_experts,
        "training_variant": variant,
        "scale_b": scale_b,
        "is_router_tuned": variant == "rt",
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = parse_dtype(args.dtype)
    model_paths = collect_model_paths(args)

    summary_records = []
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for model_path in model_paths:
        metadata = infer_model_metadata(model_path)
        model = FlexOlmoForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        model.to(device)
        model.eval()

        analysis = analyze_flex_olmo_weights(model, public_expert_idx=args.public_expert_idx)
        impl_path = str(inspect.getsourcefile(FlexOlmoForCausalLM))
        summary_record = {
            "record_type": "weight_analysis_summary",
            "model_impl_path": impl_path,
            **metadata,
            **analysis["summary"],
            "num_layers": analysis["num_layers"],
            "num_experts": analysis["num_experts"],
            "public_expert_idx": analysis["public_expert_idx"],
        }
        summary_records.append(summary_record)

        if output_dir is not None:
            detail_path = output_dir / f"{metadata['model_name']}.weights.json"
            detail_payload = {
                **metadata,
                "model_impl_path": impl_path,
                **analysis,
            }
            detail_path.write_text(json.dumps(to_jsonable(detail_payload), indent=2, sort_keys=True), encoding="utf-8")

        del model
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    write_jsonl(summary_records, args.output_jsonl)


if __name__ == "__main__":
    main()
