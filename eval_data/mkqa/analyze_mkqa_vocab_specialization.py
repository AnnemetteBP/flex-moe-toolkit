from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.utils.jsonl import write_jsonl


SOURCE_SPECS = {
    "prompt": ("input_token_ids", "prompt_topk_experts_by_layer"),
    "predicted": ("predicted_output_token_ids", "predicted_output_topk_experts_by_layer"),
    "ground_truth": ("ground_truth_output_token_ids", "ground_truth_output_topk_experts_by_layer"),
}


def normalize_layer_token_experts(layer_assignments) -> list[list[int]]:
    if not layer_assignments:
        return []

    first_item = layer_assignments[0]

    # Batched shape serialized from tensors: [batch][token][expert]
    if isinstance(first_item, list) and first_item and isinstance(first_item[0], list):
        token_assignments = first_item
    else:
        # Unbatched serialized shape: [token][expert] or [token]
        token_assignments = layer_assignments

    normalized = []
    for token_experts in token_assignments:
        if isinstance(token_experts, list):
            normalized.append([int(expert_idx) for expert_idx in token_experts])
        else:
            normalized.append([int(token_experts)])
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute vocabulary-specialization summaries from saved MKQA routing records."
    )
    parser.add_argument(
        "--routing-root",
        help="Dataset output directory containing per-run subdirectories with routing_records.jsonl files.",
    )
    parser.add_argument(
        "--routing-records-jsonl",
        action="append",
        default=[],
        help="Direct path to a routing_records.jsonl file. Repeat as needed.",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional tokenizer path for decoding token ids in the top-token summaries.",
    )
    parser.add_argument(
        "--sources",
        default="prompt,predicted,ground_truth",
        help="Comma-separated token sources to analyze.",
    )
    parser.add_argument(
        "--top-tokens",
        type=int,
        default=20,
        help="Number of top decoded tokens to save per expert/layer/source.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory for vocab-specialization outputs.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_routing_record_paths(args) -> list[Path]:
    paths = [Path(path) for path in args.routing_records_jsonl]
    if args.routing_root:
        paths.extend(sorted(Path(args.routing_root).glob("*/routing_records.jsonl")))
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("Provide either --routing-root or at least one --routing-records-jsonl path.")
    return unique


def decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    return tokenizer.decode([token_id], skip_special_tokens=False)


def normalized_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probabilities = [count / total for count in counter.values() if count > 0]
    entropy = -sum(prob * math.log(prob + 1e-12) for prob in probabilities)
    max_entropy = math.log(len(probabilities) + 1e-12)
    if max_entropy <= 0:
        return 0.0
    return entropy / max_entropy


def specialization_score(counter: Counter) -> float:
    return 1.0 - normalized_entropy(counter)


def collect_counts(records: list[dict], selected_sources: list[str]):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    token_totals = defaultdict(lambda: defaultdict(Counter))

    for record in records:
        for source in selected_sources:
            token_field, expert_field = SOURCE_SPECS[source]
            token_ids = record.get(token_field) or []
            experts_by_layer = record.get(expert_field) or []
            if not token_ids or not experts_by_layer:
                continue

            for layer_idx, layer_assignments in enumerate(experts_by_layer):
                token_assignments = normalize_layer_token_experts(layer_assignments)
                for token_id, token_experts in zip(token_ids, token_assignments):
                    for _expert_idx in token_experts:
                        token_totals[source][layer_idx][int(token_id)] += 1
                    for expert_idx in token_experts:
                        counts[source][layer_idx][int(expert_idx)][int(token_id)] += 1

    return counts, token_totals


def build_summary_records(
    records_path: Path,
    records: list[dict],
    counts,
    token_totals,
    tokenizer,
    top_tokens: int,
) -> list[dict]:
    summary_records = []
    if not records:
        return summary_records
    model_name = records[0]["model_name"]
    model_path = records[0]["model_path"]

    for source, per_layer in counts.items():
        for layer_idx, per_expert in sorted(per_layer.items()):
            expert_mean_specialization = {}
            expert_token_distribution_concentration = {}
            expert_top_tokens = {}
            layer_token_totals = token_totals[source][layer_idx]
            for expert_idx, token_counter in sorted(per_expert.items()):
                token_specialization_values = {
                    int(token_id): (
                        count / layer_token_totals[int(token_id)] if layer_token_totals[int(token_id)] else 0.0
                    )
                    for token_id, count in token_counter.items()
                }
                expert_mean_specialization[f"expert_{expert_idx}"] = (
                    sum(token_specialization_values.values()) / len(token_specialization_values)
                    if token_specialization_values
                    else 0.0
                )
                expert_token_distribution_concentration[f"expert_{expert_idx}"] = specialization_score(
                    token_counter
                )
                expert_top_tokens[f"expert_{expert_idx}"] = [
                    {
                        "token_id": token_id,
                        "decoded": decode_token(tokenizer, token_id),
                        "count": count,
                        "token_total_routes": int(layer_token_totals[int(token_id)]),
                        "specialization": token_specialization_values[int(token_id)],
                    }
                    for token_id, count in sorted(
                        token_counter.items(),
                        key=lambda item: (
                            token_specialization_values[int(item[0])],
                            item[1],
                        ),
                        reverse=True,
                    )[:top_tokens]
                ]

            summary_records.append(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "record_type": "vocab_specialization",
                    "routing_records_path": str(records_path),
                    "run_label": records_path.parent.name,
                    "source": source,
                    "layer_idx": layer_idx,
                    "num_experts": len(per_expert),
                    "token_totals": {str(token_id): int(count) for token_id, count in sorted(layer_token_totals.items())},
                    "mean_specialization_by_expert": expert_mean_specialization,
                    "expert_token_distribution_concentration": expert_token_distribution_concentration,
                    "mean_specialization": (
                        sum(expert_mean_specialization.values()) / len(expert_mean_specialization)
                        if expert_mean_specialization
                        else 0.0
                    ),
                    "top_tokens_by_expert": expert_top_tokens,
                }
            )

    return summary_records


def main():
    args = parse_args()
    selected_sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    unknown_sources = sorted(set(selected_sources) - set(SOURCE_SPECS))
    if unknown_sources:
        raise ValueError(f"Unknown source(s): {unknown_sources}. Available: {sorted(SOURCE_SPECS)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path) if args.tokenizer_path else None
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {"sources": selected_sources, "runs": {}}

    for records_path in resolve_routing_record_paths(args):
        records = load_jsonl(records_path)
        counts, token_totals = collect_counts(records, selected_sources=selected_sources)
        summary_records = build_summary_records(
            records_path=records_path,
            records=records,
            counts=counts,
            token_totals=token_totals,
            tokenizer=tokenizer,
            top_tokens=args.top_tokens,
        )

        run_dir = output_root / records_path.parent.name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "vocab_specialization_summary.jsonl"
        write_jsonl(summary_records, summary_path, sort_keys=False)

        manifest["runs"][records_path.parent.name] = {
            "routing_records_path": str(records_path),
            "summary_path": str(summary_path),
        }

    manifest_path = output_root / "vocab_specialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote vocabulary-specialization outputs to {output_root}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
