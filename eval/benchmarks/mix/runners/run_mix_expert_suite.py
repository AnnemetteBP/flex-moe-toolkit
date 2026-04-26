from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNERS_DIR = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "runners"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run compact standalone-expert mix analyses across multiple checkpoints from one config."
    )
    parser.add_argument("--config", required=True, help="Path to the expert-sweep config JSON.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_model_catalog(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def catalog_value_for_selector(catalog: dict, selector: str):
    if selector in catalog:
        value = catalog[selector]
        if not isinstance(value, list):
            raise ValueError(
                f"Model selector `{selector}` must resolve to a list of model names, not {type(value).__name__}."
            )
        return value
    aliases = {"experts_da": "combined_danish"}
    selector = aliases.get(selector, selector)
    if selector in catalog:
        value = catalog[selector]
        if not isinstance(value, list):
            raise ValueError(
                f"Model selector `{selector}` must resolve to a list of model names, not {type(value).__name__}."
            )
        return value
    value = catalog
    for part in selector.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Unknown model selector `{selector}` in models catalog.")
        value = value[part]
    if not isinstance(value, list):
        raise ValueError(
            f"Model selector `{selector}` must resolve to a list of model names, not {type(value).__name__}."
        )
    return value


def resolve_model_entries(config: dict) -> list[dict]:
    shared = config.get("shared", {})
    explicit_models = [model for model in config.get("models", []) if model.get("enabled", True)]
    selectors = ensure_list(shared.get("model_group_selectors"))
    if not selectors:
        return explicit_models
    catalog_path = shared.get("model_catalog")
    if not catalog_path:
        raise ValueError("`shared.model_catalog` is required when using `shared.model_group_selectors`.")
    catalog = load_model_catalog(catalog_path)
    model_entries = list(explicit_models)
    seen_names = {entry.get("model_name") or Path(entry["model_path"]).name for entry in model_entries}
    for selector in selectors:
        for model_name in catalog_value_for_selector(catalog, str(selector)):
            if model_name in seen_names:
                continue
            seen_names.add(model_name)
            model_entries.append({"enabled": True, "model_name": model_name})
    return model_entries


def build_common_args(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    args = [
        "--manifest-path",
        str(shared["dataset_manifest"]),
        "--device",
        str(runtime.get("device", "auto")),
        "--dtype",
        str(runtime.get("dtype", "auto")),
        "--max-length",
        str(runtime.get("max_length", 512)),
        "--max-examples-per-dataset",
        str(runtime.get("max_examples_per_dataset", 150)),
    ]
    datasets = ensure_list(shared.get("datasets"))
    if datasets:
        args.extend(["--datasets", ",".join(str(item) for item in datasets)])
    tokenizer_path = shared.get("tokenizer_path") or model_entry.get("tokenizer_path")
    if tokenizer_path is not None:
        args.extend(["--tokenizer-path", str(tokenizer_path)])
    if model_entry.get("model_path"):
        args.extend(["--model-path", str(model_entry["model_path"])])
    else:
        args.extend(["--model-name", str(model_entry["model_name"]), "--model-root", str(shared["model_root"])])
        if shared.get("model_registry") is not None:
            args.extend(["--model-registry", str(shared["model_registry"])])
    return args


def build_eval_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    command = [
        sys.executable,
        str(RUNNERS_DIR / "run_mix_expert_eval.py"),
        *build_common_args(model_entry, shared, runtime),
        "--default-max-new-tokens",
        str(runtime.get("default_max_new_tokens", 32)),
        "--output-root",
        str(Path(runtime["output_root"]) / "accuracy"),
    ]
    return command


def build_latent_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    command = [
        sys.executable,
        str(RUNNERS_DIR / "run_mix_expert_latent_space.py"),
        *build_common_args(model_entry, shared, runtime),
        "--selected-layers",
        str(shared.get("selected_layers", "early_mid_late_last")),
        "--output-root",
        str(Path(runtime["output_root"]) / "latent_space"),
    ]
    return command


def run_command(command: list[str], dry_run: bool) -> None:
    print("Command:")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    shared = dict(config.get("shared", {}))
    runtime = dict(config.get("runtime", {}))
    analyses = dict(config.get("analyses", {}))
    models = resolve_model_entries(config)

    all_commands: list[list[str]] = []
    for model_entry in models:
        if analyses.get("accuracy", {}).get("enabled", True):
            all_commands.append(build_eval_command(model_entry, shared, runtime))
        if analyses.get("latent_space", {}).get("enabled", True):
            all_commands.append(build_latent_command(model_entry, shared, runtime))

    for command in all_commands:
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
