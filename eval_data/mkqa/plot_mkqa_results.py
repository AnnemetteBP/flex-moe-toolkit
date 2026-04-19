from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from flex_moe_toolkit.plotting.routing import plot_routing_outputs


RUN_ORDER = {
    "native_full": 0,
    "public_only": 0,
    "combined_top2": 1,
    "combined_top4": 2,
    "combined_top7": 3,
}

SOURCE_ORDER = {
    "prompt": 0,
    "predicted": 1,
    "ground_truth": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sanity-check plots from MKQA routing, vocab, and domain specialization outputs."
    )
    parser.add_argument(
        "--results-root",
        default="eval_results/mkqa_results_smoke",
        help="MKQA results root containing routing/, vocab_specialization/, and domain_specialization/.",
    )
    parser.add_argument(
        "--output-root",
        help="Directory where plots will be saved. Defaults to <results-root>/plots.",
    )
    parser.add_argument(
        "--model-name",
        action="append",
        default=[],
        help="Optional model name to plot. Repeat to select multiple models. Defaults to all available models.",
    )
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sorted_run_dirs(path: Path) -> list[Path]:
    return sorted(
        [child for child in path.iterdir() if child.is_dir()],
        key=lambda child: (RUN_ORDER.get(child.name, 999), child.name),
    )


def save_run_metric_comparison_plot(summary_records: list[dict], metric_key: str, path: Path, title: str, ylabel: str):
    if not summary_records:
        return None

    ordered = sorted(summary_records, key=lambda record: RUN_ORDER.get(record["run_label"], 999))
    run_labels = [record["run_label"] for record in ordered]
    values = [float(record[metric_key]) for record in ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(run_labels, values, marker="o", linewidth=2, color="#2f6db2")
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=20)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_domain_specialization_heatmap(records: list[dict], source: str, language: str, path: Path):
    filtered = [record for record in records if record["source"] == source]
    if not filtered:
        return None

    layer_indices = sorted({int(record["layer_idx"]) for record in filtered})
    expert_indices = sorted({int(record["expert_idx"]) for record in filtered})
    layer_pos = {layer_idx: idx for idx, layer_idx in enumerate(layer_indices)}
    expert_pos = {expert_idx: idx for idx, expert_idx in enumerate(expert_indices)}

    matrix = [[0.0 for _ in expert_indices] for _ in layer_indices]
    for record in filtered:
        row_idx = layer_pos[int(record["layer_idx"])]
        col_idx = expert_pos[int(record["expert_idx"])]
        matrix[row_idx][col_idx] = float(record["language_specialization"].get(language, 0.0))

    fig_width = max(6, len(expert_indices) * 0.8)
    fig_height = max(6, len(layer_indices) * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
        xticklabels=expert_indices,
        yticklabels=layer_indices,
    )
    ax.set_title(f"{source.title()} {language.upper()} Specialization")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_vocab_specialization_layer_plot(records: list[dict], source: str, path: Path):
    filtered = sorted(
        [record for record in records if record["source"] == source],
        key=lambda record: int(record["layer_idx"]),
    )
    if not filtered:
        return None

    layers = [int(record["layer_idx"]) for record in filtered]
    values = [float(record["mean_specialization"]) for record in filtered]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, values, marker="o", linewidth=1.8, color="#b24c2f")
    ax.set_title(f"{source.title()} Vocabulary Specialization by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean specialization")
    ax.grid(axis="y", alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_model(results_root: Path, output_root: Path, model_name: str) -> dict:
    model_output_root = output_root / model_name
    model_output_root.mkdir(parents=True, exist_ok=True)

    routing_root = results_root / "routing" / model_name / "mkqa_en_da"
    vocab_root = results_root / "vocab_specialization" / model_name
    domain_root = results_root / "domain_specialization" / model_name

    if not routing_root.exists():
        raise FileNotFoundError(f"Missing routing directory for model {model_name}: {routing_root}")

    generated = {
        "model_name": model_name,
        "routing": {},
        "vocab_specialization": {},
        "domain_specialization": {},
        "comparison": {},
    }

    routing_summary_records = []

    for run_dir in sorted_run_dirs(routing_root):
        routing_analysis_path = run_dir / "routing_analysis.jsonl"
        routing_records_path = run_dir / "routing_records.jsonl"
        if not routing_analysis_path.exists():
            continue

        run_plot_dir = model_output_root / "routing" / run_dir.name
        plot_outputs = plot_routing_outputs(
            routing_analysis_path=routing_analysis_path,
            output_dir=run_plot_dir,
            eval_records_path=routing_records_path if routing_records_path.exists() else None,
        )
        generated["routing"][run_dir.name] = plot_outputs

        summary_path = run_dir / "routing_summary.jsonl"
        if summary_path.exists():
            routing_summary_records.extend(load_jsonl(summary_path))

    if routing_summary_records:
        comparison_dir = model_output_root / "comparison"
        entropy_path = comparison_dir / "routing_mean_entropy_by_run.png"
        load_balance_path = comparison_dir / "routing_mean_load_balance_by_run.png"
        save_run_metric_comparison_plot(
            routing_summary_records,
            metric_key="mean_entropy",
            path=entropy_path,
            title=f"{model_name} Mean Routing Entropy by Run",
            ylabel="Mean entropy",
        )
        save_run_metric_comparison_plot(
            routing_summary_records,
            metric_key="mean_load_balance",
            path=load_balance_path,
            title=f"{model_name} Mean Load Balance by Run",
            ylabel="Mean load balance",
        )
        generated["comparison"]["routing_mean_entropy_by_run"] = str(entropy_path)
        generated["comparison"]["routing_mean_load_balance_by_run"] = str(load_balance_path)

    if vocab_root.exists():
        for run_dir in sorted_run_dirs(vocab_root):
            summary_path = run_dir / "vocab_specialization_summary.jsonl"
            if not summary_path.exists():
                continue
            records = load_jsonl(summary_path)
            run_outputs = {}
            for source in sorted(SOURCE_ORDER, key=SOURCE_ORDER.get):
                plot_path = model_output_root / "vocab_specialization" / run_dir.name / f"{source}_mean_specialization_by_layer.png"
                generated_path = save_vocab_specialization_layer_plot(records, source=source, path=plot_path)
                if generated_path is not None:
                    run_outputs[source] = str(generated_path)
            generated["vocab_specialization"][run_dir.name] = run_outputs

    if domain_root.exists():
        for run_dir in sorted_run_dirs(domain_root):
            summary_path = run_dir / "domain_specialization_summary.jsonl"
            if not summary_path.exists():
                continue
            records = load_jsonl(summary_path)
            run_outputs = {}
            for source in sorted(SOURCE_ORDER, key=SOURCE_ORDER.get):
                for language in ("en", "da"):
                    plot_path = (
                        model_output_root
                        / "domain_specialization"
                        / run_dir.name
                        / f"{source}_{language}_specialization_heatmap.png"
                    )
                    generated_path = save_domain_specialization_heatmap(
                        records,
                        source=source,
                        language=language,
                        path=plot_path,
                    )
                    if generated_path is not None:
                        run_outputs[f"{source}_{language}"] = str(generated_path)
            generated["domain_specialization"][run_dir.name] = run_outputs

    return generated


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_root = Path(args.output_root) if args.output_root else results_root / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    model_names = args.model_name or sorted(
        model_dir.name for model_dir in (results_root / "routing").iterdir() if model_dir.is_dir()
    )

    manifest = {
        "results_root": str(results_root),
        "output_root": str(output_root),
        "models": [],
    }

    for model_name in model_names:
        manifest["models"].append(plot_model(results_root, output_root, model_name))

    manifest_path = output_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote MKQA plot manifest to {manifest_path}")


if __name__ == "__main__":
    main()
