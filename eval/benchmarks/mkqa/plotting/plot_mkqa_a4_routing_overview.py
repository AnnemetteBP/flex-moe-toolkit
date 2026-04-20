from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


MODEL_PATTERN = re.compile(r"^FlexOlmo-8x7B-1T-a4-(?P<size>\d+)B(?:-(?P<variant>.+))?$")
VARIANT_ORDER = {
    "": 0,
    "v1": 1,
    "v2": 2,
    "float32": 3,
    "rt": 4,
    "rt4": 5,
}
DEFAULT_ASSUMED_EXPERT_LABELS = {
    0: "public",
    1: "code",
    2: "creative",
    3: "math",
    4: "news",
    5: "pes2o",
    6: "reddit",
    7: "danish",
}


@dataclass(frozen=True)
class ModelMeta:
    name: str
    size_b: int
    variant: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an overview of routing-collapse indicators across A4 MKQA combined-native runs."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa/full/flexolmo/a4",
        help="MKQA combined-native results root containing routing/domain/vocab outputs.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where overview outputs will be written. Defaults to <results-root>/routing_overview.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional model names to include. Defaults to all complete A4 routing runs under the results root.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Expert index to label as the public expert. Defaults to 0, matching the FlexOlmo convention used elsewhere in this repo.",
    )
    parser.add_argument(
        "--expert-label",
        action="append",
        default=[],
        help=(
            "Optional explicit expert label mapping in the form `<idx>=<label>`. "
            "Can be passed multiple times, for example `--expert-label 0=public --expert-label 3=math`."
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("assumed", "minimal"),
        default="assumed",
        help=(
            "How to label expert ids in plots and tables. "
            "`assumed` uses the current working hypothesis "
            "`0=public, 1=code, 2=creative, 3=math, 4=news, 5=pes2o, 6=reddit, 7=danish`. "
            "`minimal` only labels the verified public expert and leaves the rest as `expert_<id>`."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_expert_labels(raw_labels: list[str], public_expert_idx: int, label_mode: str) -> dict[int, str]:
    if label_mode == "assumed":
        labels = dict(DEFAULT_ASSUMED_EXPERT_LABELS)
        if public_expert_idx != 0:
            labels = {
                (public_expert_idx if idx == 0 else idx): value
                for idx, value in labels.items()
            }
            labels[0] = f"expert_0"
    else:
        labels = {public_expert_idx: "public"}
    for raw_item in raw_labels:
        if "=" not in raw_item:
            raise ValueError(
                f"Invalid `--expert-label` value `{raw_item}`. Expected the form `<idx>=<label>`."
            )
        idx_str, label = raw_item.split("=", 1)
        labels[int(idx_str.strip())] = label.strip()
    return labels


def expert_label(expert_idx: int, labels: dict[int, str]) -> str:
    if expert_idx < 0:
        return "none"
    return labels.get(expert_idx, f"expert_{expert_idx}")


def parse_model_meta(model_name: str) -> ModelMeta:
    match = MODEL_PATTERN.match(model_name)
    if not match:
        raise ValueError(f"Unrecognized A4 model naming scheme: {model_name}")
    return ModelMeta(
        name=model_name,
        size_b=int(match.group("size")),
        variant=match.group("variant") or "",
    )


def variant_sort_key(variant: str) -> tuple[int, str]:
    return (VARIANT_ORDER.get(variant, 99), variant)


def iter_complete_models(results_root: Path) -> list[str]:
    routing_root = results_root / "routing"
    if not routing_root.exists():
        return []

    names: list[str] = []
    for model_dir in routing_root.iterdir():
        if not model_dir.is_dir():
            continue
        native_root = model_dir / "mkqa_en_da" / "native_full"
        if not (native_root / "routing_summary.jsonl").exists():
            continue
        if not (native_root / "routing_analysis.jsonl").exists():
            continue
        if not (native_root / "routing_records.jsonl").exists():
            continue
        names.append(model_dir.name)
    return sorted(names, key=lambda name: (parse_model_meta(name).size_b, variant_sort_key(parse_model_meta(name).variant)))


def effective_num_experts(counts: list[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    probs = np.array(counts, dtype=float) / total
    entropy = -np.sum([p * math.log(p) for p in probs if p > 0.0])
    return float(math.exp(entropy))


def dominant_share(counts: list[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    return float(max(counts) / total)


def experts_above_threshold(counts: list[float], threshold: float = 0.01) -> int:
    total = float(sum(counts))
    if total <= 0:
        return 0
    probs = np.array(counts, dtype=float) / total
    return int(np.sum(probs >= threshold))


def compute_layer_top1_share_matrix(routing_records: list[dict], num_experts: int) -> tuple[list[float], list[int]]:
    num_layers = len(routing_records[0]["prompt_router_token_summaries_by_layer"])
    dominant_shares: list[float] = []
    dominant_experts: list[int] = []

    for layer_idx in range(num_layers):
        counts = np.zeros(num_experts, dtype=float)
        for record in routing_records:
            top1_ids = record["prompt_router_token_summaries_by_layer"][layer_idx]["top1_expert_ids"]
            flat_ids = np.array(top1_ids, dtype=int).reshape(-1)
            for expert_idx in flat_ids.tolist():
                counts[int(expert_idx)] += 1.0
        total = float(counts.sum())
        if total <= 0:
            dominant_shares.append(0.0)
            dominant_experts.append(-1)
            continue
        dominant_experts.append(int(np.argmax(counts)))
        dominant_shares.append(float(np.max(counts) / total))

    return dominant_shares, dominant_experts


def compute_model_row(results_root: Path, model_name: str, expert_labels: dict[int, str]) -> dict:
    meta = parse_model_meta(model_name)
    native_root = results_root / "routing" / model_name / "mkqa_en_da" / "native_full"
    summary = load_jsonl(native_root / "routing_summary.jsonl")[0]
    analysis = load_jsonl(native_root / "routing_analysis.jsonl")[-1]
    routing_records = load_jsonl(native_root / "routing_records.jsonl")

    usage = analysis["usage"]
    num_experts = int(summary["available_experts"][-1]) + 1 if summary.get("available_experts") else len(usage)
    layer_dominant_shares, layer_dominant_experts = compute_layer_top1_share_matrix(routing_records, num_experts=num_experts)
    aggregate_matrix = np.array(analysis["coactivation_matrix"], dtype=float)
    offdiag_values = aggregate_matrix[~np.eye(aggregate_matrix.shape[0], dtype=bool)]

    return {
        "model_name": model_name,
        "size_b": meta.size_b,
        "variant": meta.variant or "base",
        "model_native_top_k": int(summary["model_native_top_k"]),
        "effective_top_k": int(summary["effective_top_k"]),
        "num_examples": int(summary["num_examples"]),
        "mean_entropy": float(summary["mean_entropy"]),
        "mean_load_balance": float(summary["mean_load_balance"]),
        "mean_layer_iou": float(summary["mean_layer_iou"]),
        "mean_top1_prob": float(analysis["mean_top1_prob"]),
        "mean_top1_top2_margin": float(analysis["mean_top1_top2_margin"]),
        "mean_top2_prob": float(analysis["mean_top2_prob"]),
        "mean_selected_expert_prob_mass": float(analysis["mean_selected_expert_prob_mass"]),
        "effective_num_experts": effective_num_experts(usage),
        "dominant_expert_share": dominant_share(usage),
        "dominant_expert_idx": int(np.argmax(np.array(usage, dtype=float))),
        "dominant_expert_label": expert_label(int(np.argmax(np.array(usage, dtype=float))), expert_labels),
        "experts_above_1pct": experts_above_threshold(usage, threshold=0.01),
        "mean_offdiag_coactivation": float(offdiag_values.mean()) if offdiag_values.size else 0.0,
        "layer_dominant_share_mean": float(np.mean(layer_dominant_shares)),
        "layer_dominant_share_max": float(np.max(layer_dominant_shares)),
        "layer_dominant_shares": layer_dominant_shares,
        "layer_dominant_experts": layer_dominant_experts,
        "layer_dominant_labels": [expert_label(expert_idx, expert_labels) for expert_idx in layer_dominant_experts],
        "usage": [float(value) for value in usage],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, list)
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scalar_rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_findings(rows: list[dict]) -> str:
    by_effective = sorted(rows, key=lambda row: row["effective_num_experts"])
    by_dominant = sorted(rows, key=lambda row: row["dominant_expert_share"], reverse=True)
    by_balance = sorted(rows, key=lambda row: row["mean_load_balance"], reverse=True)
    standard_rows = [row for row in rows if "-rt" not in row["model_name"]]
    standard_by_effective = sorted(standard_rows, key=lambda row: row["effective_num_experts"]) if standard_rows else []

    least_diverse = by_effective[0]
    most_dominant = by_dominant[0]
    best_balanced = by_balance[0]
    least_diverse_standard = standard_by_effective[0] if standard_by_effective else None
    most_diverse_standard = standard_by_effective[-1] if standard_by_effective else None

    median_effective = float(np.median([row["effective_num_experts"] for row in rows]))
    median_dominant = float(np.median([row["dominant_expert_share"] for row in rows]))

    lines = [
        "# A4 Routing Overview",
        "",
        (
            "- Expert labels use the current working hypothesis for native A4 checkpoints: "
            "`0=public, 1=code, 2=creative, 3=math, 4=news, 5=pes2o, 6=reddit, 7=danish`. "
            "Only `0=public` is verified directly in this repo; the rest are provisional labels."
        ),
        f"- Models analyzed: {len(rows)} complete native-full A4 runs.",
        f"- Median effective experts from aggregate usage: {median_effective:.3f}.",
        f"- Median dominant-expert share from aggregate usage: {median_dominant:.3%}.",
        (
            f"- Most collapsed by effective expert count: `{least_diverse['model_name']}` "
            f"({least_diverse['effective_num_experts']:.3f} effective experts, "
            f"{least_diverse['dominant_expert_share']:.3%} dominant share)."
        ),
        (
            f"- Highest dominant-expert share: `{most_dominant['model_name']}` "
            f"({most_dominant['dominant_expert_share']:.3%}, "
            f"{most_dominant['dominant_expert_label']} / id {most_dominant['dominant_expert_idx']})."
        ),
        (
            f"- Highest saved load-balance score: `{best_balanced['model_name']}` "
            f"({best_balanced['mean_load_balance']:.4f}), but it still routes "
            f"{best_balanced['dominant_expert_share']:.3%} of aggregate top-1 traffic to one expert."
        ),
    ]
    if least_diverse_standard and most_diverse_standard:
        lines.extend(
            [
                (
                    f"- Among the non-RT A4 checkpoints, effective expert count only ranges from "
                    f"{least_diverse_standard['effective_num_experts']:.3f} "
                    f"(`{least_diverse_standard['model_name']}`) to "
                    f"{most_diverse_standard['effective_num_experts']:.3f} "
                    f"(`{most_diverse_standard['model_name']}`), so the collapse pattern is very stable."
                ),
            ]
        )
    lines.extend(
        [
        (
            "- Interpretation: across the standard A4 checkpoints, routing remains strongly collapsed in aggregate "
            "because almost all prompt top-1 traffic goes to one expert, even when mean load-balance differences "
            "move a bit with scale or checkpoint variant."
        ),
        ]
    )
    return "\n".join(lines) + "\n"


def plot_metric_overview(rows: list[dict], output_path: Path) -> None:
    labels = [row["model_name"].replace("FlexOlmo-8x7B-1T-", "") for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    metrics = [
        ("effective_num_experts", "Effective Experts", "#246eb9"),
        ("dominant_expert_share", "Dominant Expert Share", "#c94f3d"),
        ("mean_load_balance", "Mean Load Balance", "#3d9970"),
        ("mean_top1_top2_margin", "Mean Top-1/Top-2 Margin", "#8e5ea2"),
    ]

    for ax, (field, title, color) in zip(axes, metrics):
        values = [row[field] for row in rows]
        ax.plot(x, values, marker="o", color=color, linewidth=2)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.25)
        if field == "dominant_expert_share":
            ax.set_ylim(0.0, 1.0)
        if field == "effective_num_experts":
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(8.0, color="black", linestyle=":", linewidth=1, alpha=0.5)

    axes[0].set_title("A4 Routing Collapse Indicators Across Checkpoints")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=55, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_layer_dominance_heatmaps(rows: list[dict], share_path: Path, expert_path: Path, expert_labels: dict[int, str]) -> None:
    share_matrix = np.array([row["layer_dominant_shares"] for row in rows], dtype=float)
    expert_matrix = np.array([row["layer_dominant_experts"] for row in rows], dtype=float)
    labels = [row["model_name"].replace("FlexOlmo-8x7B-1T-", "") for row in rows]

    fig, ax = plt.subplots(figsize=(16, max(6, 0.38 * len(rows) + 2)))
    sns.heatmap(share_matrix, cmap="mako", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Per-Layer Top-1 Dominant-Expert Share")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")
    ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()
    fig.savefig(share_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, max(6, 0.38 * len(rows) + 2)))
    sns.heatmap(expert_matrix, cmap="tab10", vmin=-0.5, vmax=7.5, ax=ax, cbar_kws={"ticks": list(range(8))})
    ax.set_title("Per-Layer Dominant Expert Identity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Model")
    ax.set_yticklabels(labels, rotation=0)
    colorbar = ax.collections[0].colorbar
    ticks = list(range(8))
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([expert_label(tick, expert_labels) for tick in ticks])
    fig.tight_layout()
    fig.savefig(expert_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "routing_overview"
    output_root.mkdir(parents=True, exist_ok=True)
    expert_labels = parse_expert_labels(
        args.expert_label,
        public_expert_idx=args.public_expert_idx,
        label_mode=args.label_mode,
    )

    model_names = args.model_name or iter_complete_models(results_root)
    if not model_names:
        raise SystemExit(f"No complete A4 routing runs found under {results_root}.")

    rows = [compute_model_row(results_root, model_name, expert_labels=expert_labels) for model_name in model_names]
    rows = sorted(rows, key=lambda row: (row["size_b"], variant_sort_key("" if row["variant"] == "base" else row["variant"])))

    write_csv(rows, output_root / "a4_routing_overview.csv")
    write_json(output_root / "a4_routing_overview.json", rows)
    (output_root / "README.md").write_text(summarize_findings(rows), encoding="utf-8")
    plot_metric_overview(rows, output_root / "a4_routing_collapse_metrics.png")
    plot_layer_dominance_heatmaps(
        rows,
        share_path=output_root / "a4_layer_dominant_share_heatmap.png",
        expert_path=output_root / "a4_layer_dominant_expert_heatmap.png",
        expert_labels=expert_labels,
    )

    print(f"Wrote A4 routing overview for {len(rows)} models to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
