from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_expert_weight_analysis.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the standalone-expert weight-analysis suite across multiple checkpoints.")
    parser.add_argument("--config", required=True, help="Path to the expert weight-analysis suite config JSON.")
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
    models = [model for model in config.get("models", []) if model.get("enabled", True)]
    if not models:
        raise ValueError("No enabled models were defined in the expert weight-analysis config.")

    selected_layers = shared.get("selected_layers", "early_mid_late_last")
    selected_layers_arg = ",".join(str(layer) for layer in selected_layers) if isinstance(selected_layers, list) else str(selected_layers)
    output_root = runtime.get("output_root", str(PROJECT_ROOT / "eval_results" / "mix" / "expert_weight_analysis" / "a4"))
    Path(output_root).mkdir(parents=True, exist_ok=True)

    commands = []
    for model in models:
        command = [
            sys.executable,
            str(RUNNER_PATH),
            "--device",
            str(runtime.get("device", "auto")),
            "--dtype",
            str(runtime.get("dtype", "auto")),
            "--selected-layers",
            selected_layers_arg,
            "--fingerprint-size",
            str(runtime.get("fingerprint_size", 4096)),
            "--fingerprint-seed",
            str(runtime.get("fingerprint_seed", 17)),
            "--output-root",
            str(output_root),
            "--model-name",
            str(model["model_name"]),
            "--model-root",
            str(shared["model_root"]),
            "--model-registry",
            str(shared["model_registry"]),
        ]
        commands.append(command)

    manifest_path = Path(output_root) / "expert_weight_analysis_suite_commands.json"
    manifest_path.write_text(json.dumps({"config": str(Path(args.config).resolve()), "commands": commands}, indent=2))
    for command in commands:
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
