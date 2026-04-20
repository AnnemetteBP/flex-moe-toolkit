from __future__ import annotations

import argparse
import csv
import os
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
DEFAULT_FOCUS_MODELS = [
    "FlexOlmo-8x7B-1T-a4-5B",
    "FlexOlmo-8x7B-1T-a4-25B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2",
    "FlexOlmo-8x7B-1T-a4-55B-v2-rt",
]
LANGUAGE_COLORS = {
    "overall": "#4c78a8",
    "en": "#54a24b",
    "da": "#f58518",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an overview plot for post-hoc A4 MKQA accuracy summaries."
    )
    parser.add_argument(
        "--accuracy-root",
        default="eval_results/mkqa/full/flexolmo/a4/accuracy",
        help="Directory containing mkqa_accuracy_overview.csv",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where the plots will be written. Defaults to <accuracy-root>/plots",
    )
    parser.add_argument(
        "--focus-model",
        action="append",
        default=[],
        help="Optional explicit focus models for the compact comparison panel.",
    )
    return parser.parse_args()


def model_sort_key(model_name: str) -> tuple[int, tuple[int, str]]:
    match = MODEL_PATTERN.match(model_name)
    if not match:
        return (999, (999, model_name))
    size_b = int(match.group("size"))
    variant = match.group("variant") or ""
    return (size_b, (VARIANT_ORDER.get(variant, 99), variant))


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in (
            "exact_match_accuracy",
            "relaxed_match_accuracy",
            "mean_token_f1",
            "en_exact_match_accuracy",
            "en_relaxed_match_accuracy",
            "da_exact_match_accuracy",
            "da_relaxed_match_accuracy",
            "en_mean_token_f1",
            "da_mean_token_f1",
        ):
            row[key] = float(row[key])
    return rows


def choose_focus_models(rows: list[dict], explicit_focus: list[str]) -> list[str]:
    available = {row["model_name"] for row in rows}
    if explicit_focus:
        return [name for name in explicit_focus if name in available]
    return [name for name in DEFAULT_FOCUS_MODELS if name in available]


def plot_full_ranking(rows: list[dict], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["relaxed_match_accuracy"], reverse=True)
    labels = [row["model_name"].replace("FlexOlmo-8x7B-1T-", "") for row in ordered]
    values = [row["relaxed_match_accuracy"] for row in ordered]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.34 * len(ordered) + 1.5)))
    ax.barh(np.arange(len(ordered)), values, color=LANGUAGE_COLORS["overall"])
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relaxed Match Accuracy")
    ax.set_title("A4 MKQA Accuracy Ranking")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_focus_comparison(rows: list[dict], focus_models: list[str], output_path: Path) -> None:
    selected = [row for row in rows if row["model_name"] in focus_models]
    selected.sort(key=lambda row: model_sort_key(row["model_name"]))
    x = np.arange(len(selected))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    relaxed = [row["relaxed_match_accuracy"] for row in selected]
    en_relaxed = [row["en_relaxed_match_accuracy"] for row in selected]
    da_relaxed = [row["da_relaxed_match_accuracy"] for row in selected]
    axes[0].bar(x - width, relaxed, width=width, color=LANGUAGE_COLORS["overall"], label="Overall")
    axes[0].bar(x, en_relaxed, width=width, color=LANGUAGE_COLORS["en"], label="English")
    axes[0].bar(x + width, da_relaxed, width=width, color=LANGUAGE_COLORS["da"], label="Danish")
    axes[0].set_title("Relaxed Match Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["model_name"].replace("FlexOlmo-8x7B-1T-", "") for row in selected], rotation=30, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    f1 = [row["mean_token_f1"] for row in selected]
    en_f1 = [row["en_mean_token_f1"] for row in selected]
    da_f1 = [row["da_mean_token_f1"] for row in selected]
    axes[1].bar(x - width, f1, width=width, color=LANGUAGE_COLORS["overall"], label="Overall")
    axes[1].bar(x, en_f1, width=width, color=LANGUAGE_COLORS["en"], label="English")
    axes[1].bar(x + width, da_f1, width=width, color=LANGUAGE_COLORS["da"], label="Danish")
    axes[1].set_title("Mean Token F1")
    axes[1].set_ylabel("F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([row["model_name"].replace("FlexOlmo-8x7B-1T-", "") for row in selected], rotation=30, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Focused A4 MKQA Accuracy Comparison", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(rows: list[dict], focus_models: list[str], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda row: row["relaxed_match_accuracy"], reverse=True)
    lines = [
        "# A4 MKQA Accuracy Overview",
        "",
        "- Scores are computed post hoc from saved `routing_records.jsonl` generations.",
        "- `relaxed_match_accuracy` uses QA-style normalization plus a substring-based relaxed match.",
        "- `mean_token_f1` is token-level F1 after the same normalization.",
        "",
        "## Top Models",
        "",
    ]
    for row in ordered[:5]:
        lines.append(
            f"- `{row['model_name']}`: relaxed={row['relaxed_match_accuracy']:.3%}, "
            f"exact={row['exact_match_accuracy']:.3%}, f1={row['mean_token_f1']:.3%}"
        )
    lines.extend(
        [
            "",
            "## Focus Models",
            "",
        ]
    )
    for model_name in focus_models:
        row = next((item for item in rows if item["model_name"] == model_name), None)
        if row is None:
            continue
        lines.append(
            f"- `{model_name}`: relaxed overall={row['relaxed_match_accuracy']:.3%}, "
            f"en={row['en_relaxed_match_accuracy']:.3%}, da={row['da_relaxed_match_accuracy']:.3%}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    accuracy_root = Path(args.accuracy_root)
    output_root = Path(args.output_root) if args.output_root else accuracy_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(accuracy_root / "mkqa_accuracy_overview.csv")
    focus_models = choose_focus_models(rows, args.focus_model)
    plot_full_ranking(rows, output_root / "a4_mkqa_accuracy_ranking.png")
    if focus_models:
        plot_focus_comparison(rows, focus_models, output_root / "a4_mkqa_accuracy_focus.png")
    write_readme(rows, focus_models, output_root / "README.md")
    print(f"Wrote A4 MKQA accuracy plots to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
