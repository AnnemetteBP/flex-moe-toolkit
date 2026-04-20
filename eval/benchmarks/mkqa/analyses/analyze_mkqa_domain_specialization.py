from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
        description="Compute domain/language specialization summaries from saved MKQA routing records."
    )
    parser.add_argument("--routing-root", help="Directory containing per-run routing_records.jsonl files.")
    parser.add_argument(
        "--routing-records-jsonl",
        action="append",
        default=[],
        help="Direct path to a routing_records.jsonl file. Repeat as needed.",
    )
    parser.add_argument(
        "--sources",
        default="prompt,predicted,ground_truth",
        help="Comma-separated token sources to analyze.",
    )
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
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


def build_domain_specialization_records(records: list[dict], selected_sources: list[str]) -> list[dict]:
    if not records:
        return []

    model_name = records[0]["model_name"]
    model_path = records[0]["model_path"]
    run_label = records[0]["run_label"]

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    domain_token_totals = defaultdict(Counter)

    for record in records:
        language = record["language"]
        for source in selected_sources:
            token_field, expert_field = SOURCE_SPECS[source]
            token_ids = record.get(token_field) or []
            experts_by_layer = record.get(expert_field) or []
            if not token_ids or not experts_by_layer:
                continue

            domain_token_totals[source][language] += len(token_ids)

            for layer_idx, layer_assignments in enumerate(experts_by_layer):
                token_assignments = normalize_layer_token_experts(layer_assignments)
                for token_experts in token_assignments:
                    for expert_idx in token_experts:
                        counts[source][layer_idx][int(expert_idx)][language] += 1

    summary_records = []
    for source, per_layer in counts.items():
        for layer_idx, per_expert in sorted(per_layer.items()):
            for expert_idx, language_counter in sorted(per_expert.items()):
                total_assignments = sum(language_counter.values())
                language_specialization = {
                    language: count / domain_token_totals[source][language]
                    if domain_token_totals[source][language]
                    else 0.0
                    for language, count in sorted(language_counter.items())
                }
                expert_conditioned_distribution = {
                    language: count / total_assignments if total_assignments else 0.0
                    for language, count in sorted(language_counter.items())
                }
                summary_records.append(
                    {
                        "model_name": model_name,
                        "model_path": model_path,
                        "record_type": "domain_specialization",
                        "run_label": run_label,
                        "source": source,
                        "layer_idx": layer_idx,
                        "expert_idx": expert_idx,
                        "assignment_count": total_assignments,
                        "language_token_totals": dict(sorted(domain_token_totals[source].items())),
                        "language_assignment_counts": dict(sorted(language_counter.items())),
                        "language_specialization": language_specialization,
                        "expert_conditioned_language_distribution": expert_conditioned_distribution,
                    }
                )
    return summary_records


def main():
    args = parse_args()
    selected_sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    unknown_sources = sorted(set(selected_sources) - set(SOURCE_SPECS))
    if unknown_sources:
        raise ValueError(f"Unknown source(s): {unknown_sources}. Available: {sorted(SOURCE_SPECS)}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": selected_sources, "runs": {}}

    for records_path in resolve_routing_record_paths(args):
        records = load_jsonl(records_path)
        summary_records = build_domain_specialization_records(records, selected_sources=selected_sources)
        run_dir = output_root / records_path.parent.name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "domain_specialization_summary.jsonl"
        write_jsonl(summary_records, summary_path, sort_keys=False)
        manifest["runs"][records_path.parent.name] = {
            "routing_records_path": str(records_path),
            "summary_path": str(summary_path),
        }

    manifest_path = output_root / "domain_specialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote domain-specialization outputs to {output_root}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
