from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


HEAVY_RECORD_FIELDS = {
    "prompt_router_logits_by_layer",
    "prompt_router_probs_by_layer",
    "prompt_hidden_states_by_layer",
    "prompt_hidden_state_norms_by_layer",
    "ground_truth_output_hidden_states_by_layer",
    "ground_truth_output_hidden_state_norms_by_layer",
    "predicted_output_hidden_states_by_layer",
    "predicted_output_hidden_state_norms_by_layer",
    "ground_truth_router_token_summaries_by_layer",
    "predicted_router_token_summaries_by_layer",
    "prompt_router_token_summaries_by_layer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a slimmer copy of mix-suite results for local transfer by removing the heaviest "
            "fields from routing_records.jsonl while keeping manifests and summary files."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Source result directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Destination slim result directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output directory.",
    )
    return parser.parse_args()


def slim_record(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in HEAVY_RECORD_FIELDS
    }


def slim_jsonl_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    input_bytes = input_path.stat().st_size
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            dst.write(json.dumps(slim_record(record), ensure_ascii=False) + "\n")

    output_bytes = output_path.stat().st_size
    return input_bytes, output_bytes


def copy_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)
    return input_path.stat().st_size, output_path.stat().st_size


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    total_input_bytes = 0
    total_output_bytes = 0
    slimmed_files = 0
    copied_files = 0

    for input_path in sorted(input_root.rglob("*")):
        if not input_path.is_file():
            continue
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path

        if input_path.name == "routing_records.jsonl":
            in_bytes, out_bytes = slim_jsonl_file(input_path, output_path)
            slimmed_files += 1
        else:
            in_bytes, out_bytes = copy_file(input_path, output_path)
            copied_files += 1

        total_input_bytes += in_bytes
        total_output_bytes += out_bytes

    print(f"Input root:   {input_root}")
    print(f"Output root:  {output_root}")
    print(f"Slimmed files: {slimmed_files}")
    print(f"Copied files:  {copied_files}")
    print(f"Input size:    {total_input_bytes / (1024 ** 3):.2f} GiB")
    print(f"Output size:   {total_output_bytes / (1024 ** 3):.2f} GiB")
    if total_input_bytes > 0:
        reduction = 100.0 * (1.0 - (total_output_bytes / total_input_bytes))
        print(f"Reduction:     {reduction:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
