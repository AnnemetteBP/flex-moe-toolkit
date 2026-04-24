from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "runners" / "run_mix_latent_space.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mix latent-space suite across multiple checkpoints.")
    parser.add_argument("--config", required=True, help="Path to the latent-space suite config JSON.")
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
        raise ValueError("No enabled models were defined in the latent-space config.")

    selected_layers = shared.get("selected_layers", [0, 8, 16, 24, -1])
    if isinstance(selected_layers, list):
        selected_layers_arg = ",".join(str(layer) for layer in selected_layers)
    else:
        selected_layers_arg = str(selected_layers)

    datasets = shared.get("datasets", [])
    datasets_arg = ",".join(str(name) for name in datasets) if datasets else None
    output_root = runtime.get(
        "output_root",
        str(PROJECT_ROOT / "eval_results" / "mix" / "focused" / "55b_pair" / "latent_space"),
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    commands = []
    for model in models:
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
            "--representation-sources",
            str(runtime.get("representation_sources", "hidden_state,pre_router")),
            "--max-examples-per-dataset",
            str(runtime.get("max_examples_per_dataset", 75)),
            "--output-root",
            str(output_root),
            "--model-name",
            str(model["model_name"]),
            "--model-root",
            str(shared["model_root"]),
            "--model-registry",
            str(shared["model_registry"]),
        ]
        if datasets_arg:
            command.extend(["--datasets", datasets_arg])
        commands.append(command)

    manifest_path = Path(output_root) / "latent_space_suite_commands.json"
    manifest_path.write_text(json.dumps({"config": str(Path(args.config).resolve()), "commands": commands}, indent=2))
    for command in commands:
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
