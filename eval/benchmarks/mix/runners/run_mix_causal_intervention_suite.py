from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_causal_intervention.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mix causal intervention suite across model pairs.")
    parser.add_argument("--config", required=True, help="Path to the causal intervention suite config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(command: list[str], dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    shared = config.get("shared", {})
    runtime = config.get("runtime", {})
    comparisons = [comparison for comparison in config.get("comparisons", []) if comparison.get("enabled", True)]
    if not comparisons:
        raise ValueError("No enabled comparisons were defined in the causal intervention config.")

    selected_layers = shared.get("selected_layers", [8, 16, 24, -1])
    if isinstance(selected_layers, list):
        selected_layers_arg = ",".join(str(layer) for layer in selected_layers)
    else:
        selected_layers_arg = str(selected_layers)

    datasets = shared.get("datasets", [])
    datasets_arg = ",".join(str(name) for name in datasets) if datasets else None
    output_root = runtime.get(
        "output_root",
        str(PROJECT_ROOT / "eval_results" / "mix" / "focused" / "55b_pair" / "causal_intervention"),
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    commands = []
    for comparison in comparisons:
        command = [
            sys.executable,
            str(RUNNER_PATH),
            "--manifest-path",
            str(shared["dataset_manifest"]),
            "--device",
            str(runtime.get("device", "auto")),
            "--dtype",
            str(runtime.get("dtype", "auto")),
            "--max-length",
            str(runtime.get("max_length", 512)),
            "--selected-layers",
            selected_layers_arg,
            "--position-policy",
            str(runtime.get("position_policy", "last_prompt_token")),
            "--intervention-kind",
            str(runtime.get("intervention_kind", "remove_delta")),
            "--source-mode",
            str(runtime.get("source_mode", "comparison_model")),
            "--delta-basis",
            str(runtime.get("delta_basis", "source_mode_diff")),
            "--delta-aggregation",
            str(runtime.get("delta_aggregation", "per_example")),
            "--max-examples-per-dataset",
            str(runtime.get("max_examples_per_dataset", 75)),
            "--output-root",
            str(output_root),
            "--target-model-name",
            str(comparison["target_model_name"]),
            "--target-model-root",
            str(shared["model_root"]),
            "--comparison-model-name",
            str(comparison["comparison_model_name"]),
            "--comparison-model-root",
            str(shared["model_root"]),
            "--model-registry",
            str(shared["model_registry"]),
            "--public-expert-idx",
            str(shared.get("public_expert_idx", 0)),
        ]
        if datasets_arg:
            command.extend(["--datasets", datasets_arg])
        if shared.get("tokenizer_path") is not None:
            command.extend(["--tokenizer-path", str(shared["tokenizer_path"])])
        commands.append(command)

    manifest_path = Path(output_root) / "causal_intervention_suite_commands.json"
    manifest_path.write_text(json.dumps({"config": str(Path(args.config).resolve()), "commands": commands}, indent=2))
    for command in commands:
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
