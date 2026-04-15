from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UTILS_DIR = PROJECT_ROOT / "scripts" / "flex_olmo" / "utils"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one or more prepared FlexOlmo analyses from a single JSON config "
            "without mixing their outputs."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON run config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def resolve_model_paths(block: dict | None, collections: dict[str, list[str]], collection_field: str = "model_collections") -> list[str]:
    if not block:
        return []

    paths = []
    paths.extend(ensure_list(block.get("model_paths")))

    for collection_name in ensure_list(block.get(collection_field)):
        if collection_name not in collections:
            raise ValueError(
                f"Unknown model collection `{collection_name}`. "
                f"Available collections: {sorted(collections)}"
            )
        paths.extend(ensure_list(collections[collection_name]))

    unique = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def resolve_dataset_paths(shared: dict) -> list[str]:
    dataset_paths = ensure_list(shared.get("dataset_paths"))
    dataset_dir = shared.get("dataset_dir")

    if dataset_dir:
        for dataset_path in sorted(Path(dataset_dir).glob("*.jsonl")):
            dataset_paths.append(str(dataset_path))

    unique = []
    seen = set()
    for path in dataset_paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    return unique


def append_common_runtime_args(command: list[str], runtime: dict, shared: dict):
    if runtime.get("device") is not None:
        command.extend(["--device", str(runtime["device"])])
    if runtime.get("dtype") is not None:
        command.extend(["--dtype", str(runtime["dtype"])])
    if shared.get("tokenizer_path") is not None:
        command.extend(["--tokenizer-path", str(shared["tokenizer_path"])])
    if runtime.get("max_length") is not None:
        command.extend(["--max-length", str(runtime["max_length"])])
    if runtime.get("max_examples") is not None:
        command.extend(["--max-examples", str(runtime["max_examples"])])


def build_router_suite_command(config: dict, runtime: dict, shared: dict) -> list[str] | None:
    analysis = config.get("analyses", {}).get("router_suite")
    if not analysis or not analysis.get("enabled", False):
        return None

    model_paths = resolve_model_paths(analysis, config.get("model_collections", {}))
    dataset_paths = resolve_dataset_paths(shared)
    if not model_paths:
        raise ValueError("`router_suite` is enabled but no model paths were resolved.")
    if not dataset_paths:
        raise ValueError("`router_suite` is enabled but no dataset paths were resolved.")

    command = [
        sys.executable,
        str(UTILS_DIR / "evaluate_flex_olmo_model_suite.py"),
        "--output-root",
        str(analysis.get("output_root", Path(runtime["output_root"]) / "router_suite")),
    ]
    append_common_runtime_args(command, runtime, shared)
    if analysis.get("summary_jsonl") is not None:
        command.extend(["--summary-jsonl", str(analysis["summary_jsonl"])])
    for model_path in model_paths:
        command.extend(["--model-path", model_path])
    for dataset_path in dataset_paths:
        command.extend(["--dataset", dataset_path])
    return command


def build_weight_analysis_command(config: dict, runtime: dict, shared: dict) -> list[str] | None:
    analysis = config.get("analyses", {}).get("weight_analysis")
    if not analysis or not analysis.get("enabled", False):
        return None

    model_paths = resolve_model_paths(analysis, config.get("model_collections", {}))
    if not model_paths:
        raise ValueError("`weight_analysis` is enabled but no model paths were resolved.")

    output_root = Path(analysis.get("output_root", Path(runtime["output_root"]) / "weight_analysis"))
    command = [
        sys.executable,
        str(UTILS_DIR / "analyze_flex_olmo_weights.py"),
        "--output-jsonl",
        str(analysis.get("output_jsonl", output_root / "weight_analysis_summary.jsonl")),
    ]
    if analysis.get("output_dir") is not None:
        command.extend(["--output-dir", str(analysis["output_dir"])])
    else:
        command.extend(["--output-dir", str(output_root / "details")])
    if runtime.get("device") is not None:
        command.extend(["--device", str(runtime["device"])])
    if runtime.get("dtype") is not None:
        command.extend(["--dtype", str(runtime["dtype"])])
    if analysis.get("public_expert_idx") is not None:
        command.extend(["--public-expert-idx", str(analysis["public_expert_idx"])])
    for model_path in model_paths:
        command.extend(["--model-path", model_path])
    return command


def build_router_saturation_command(config: dict, runtime: dict, shared: dict) -> list[str] | None:
    analysis = config.get("analyses", {}).get("router_saturation")
    if not analysis or not analysis.get("enabled", False):
        return None

    checkpoint_paths = resolve_model_paths(
        analysis,
        config.get("model_collections", {}),
        collection_field="checkpoint_collections",
    )
    dataset_paths = resolve_dataset_paths(shared)
    final_checkpoint_path = analysis.get("final_checkpoint_path")

    if not final_checkpoint_path:
        raise ValueError("`router_saturation` is enabled but `final_checkpoint_path` is missing.")
    if not checkpoint_paths:
        raise ValueError("`router_saturation` is enabled but no checkpoint paths were resolved.")
    if not dataset_paths:
        raise ValueError("`router_saturation` is enabled but no dataset paths were resolved.")

    command = [
        sys.executable,
        str(UTILS_DIR / "compare_flex_olmo_checkpoints.py"),
        "--final-checkpoint-path",
        str(final_checkpoint_path),
        "--output-root",
        str(analysis.get("output_root", Path(runtime["output_root"]) / "router_saturation")),
    ]
    append_common_runtime_args(command, runtime, shared)
    if analysis.get("summary_jsonl") is not None:
        command.extend(["--summary-jsonl", str(analysis["summary_jsonl"])])
    if analysis.get("top_k") is not None:
        top_k = analysis["top_k"]
        if isinstance(top_k, list):
            top_k = ",".join(str(item) for item in top_k)
        command.extend(["--top-k", str(top_k)])
    for checkpoint_path in checkpoint_paths:
        command.extend(["--checkpoint-path", checkpoint_path])
    for dataset_path in dataset_paths:
        command.extend(["--dataset", dataset_path])
    return command


def run_command(command: list[str], dry_run: bool):
    print("Command:")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    config = load_config(args.config)

    runtime = config.get("runtime", {})
    if "output_root" not in runtime:
        runtime["output_root"] = str(PROJECT_ROOT / "outputs" / "flex_olmo" / "analysis_runs")

    shared = config.get("shared", {})
    config.setdefault("model_collections", {})

    commands = []
    for builder in (
        build_router_suite_command,
        build_weight_analysis_command,
        build_router_saturation_command,
    ):
        command = builder(config, runtime, shared)
        if command is not None:
            commands.append(command)

    if not commands:
        raise ValueError("No analyses were enabled in the config.")

    Path(runtime["output_root"]).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(runtime["output_root"]) / "analysis_run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(args.config).resolve()),
                "num_commands": len(commands),
                "commands": commands,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for command in commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
