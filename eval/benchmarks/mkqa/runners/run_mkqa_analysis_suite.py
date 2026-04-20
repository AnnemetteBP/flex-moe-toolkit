from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
MKQA_DIR = PROJECT_ROOT / "eval" / "benchmarks" / "mkqa"
RUNNERS_DIR = MKQA_DIR / "runners"
ANALYSES_DIR = MKQA_DIR / "analyses"
WEIGHTS_SCRIPT = PROJECT_ROOT / "scripts" / "flex_olmo" / "utils" / "analyze_flex_olmo_weights.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run prepared MKQA analyses across multiple FlexOlmo checkpoints from one config."
    )
    parser.add_argument("--config", required=True, help="Path to the MKQA suite config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
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


def model_slug(model_entry: dict) -> str:
    if model_entry.get("model_name"):
        return str(model_entry["model_name"])
    return Path(model_entry["model_path"]).name


def build_mkqa_dataset_name(languages: list[str]) -> str:
    return "mkqa_" + "_".join(languages)


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
    aliases = {
        "experts_da": "combined_danish",
    }
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
    model_group_selectors = ensure_list(shared.get("model_group_selectors"))

    if not model_group_selectors:
        return explicit_models

    catalog_path = shared.get("model_catalog")
    if not catalog_path:
        raise ValueError("`shared.model_catalog` is required when using `shared.model_group_selectors`.")

    catalog = load_model_catalog(catalog_path)
    model_entries = list(explicit_models)
    seen_model_names = {
        model.get("model_name") or Path(model["model_path"]).name
        for model in model_entries
    }

    for selector in model_group_selectors:
        for model_name in catalog_value_for_selector(catalog, str(selector)):
            if model_name in seen_model_names:
                continue
            seen_model_names.add(model_name)
            model_entries.append(
                {
                    "enabled": True,
                    "model_name": model_name,
                }
            )

    return model_entries


def build_routing_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    command = [
        sys.executable,
        str(RUNNERS_DIR / "run_mkqa_routing_analysis.py"),
        "--data-path",
        str(shared["data_path"]),
        "--languages",
        ",".join(shared.get("languages", ["en", "da"])),
        "--device",
        str(runtime.get("device", "auto")),
        "--dtype",
        str(runtime.get("dtype", "auto")),
        "--max-length",
        str(runtime.get("max_length", 512)),
        "--combined-active-experts",
        str(shared.get("combined_active_experts", "2,4,7")),
        "--output-root",
        str(Path(runtime["output_root"]) / "routing" / model_slug(model_entry)),
    ]
    if shared.get("routing_run_mode") is not None:
        command.extend(["--routing-run-mode", str(shared["routing_run_mode"])])
    if runtime.get("max_examples") is not None:
        command.extend(["--max-examples", str(runtime["max_examples"])])
    if runtime.get("default_max_new_tokens") is not None:
        command.extend(["--default-max-new-tokens", str(runtime["default_max_new_tokens"])])
    if shared.get("public_expert_idx") is not None:
        command.extend(["--public-expert-idx", str(shared["public_expert_idx"])])
    if shared.get("include_individual_experts", False):
        command.append("--include-individual-experts")
    if shared.get("capture_router_tensors", False):
        command.append("--capture-router-tensors")
    if shared.get("capture_hidden_states", False):
        command.append("--capture-hidden-states")
    if shared.get("hidden_state_layers") is not None:
        hidden_state_layers = shared["hidden_state_layers"]
        if isinstance(hidden_state_layers, list):
            hidden_state_layers = ",".join(str(layer) for layer in hidden_state_layers)
        command.extend(["--hidden-state-layers", str(hidden_state_layers)])
    if not shared.get("capture_output_token_ids", True):
        command.append("--skip-output-token-capture")
    if shared.get("tokenizer_path") is not None:
        command.extend(["--tokenizer-path", str(shared["tokenizer_path"])])
    if model_entry.get("model_path"):
        command.extend(["--model-path", str(model_entry["model_path"])])
    else:
        command.extend(
            [
                "--model-name",
                str(model_entry["model_name"]),
                "--model-root",
                str(shared["model_root"]),
            ]
        )
        if shared.get("model_registry") is not None:
            command.extend(["--model-registry", str(shared["model_registry"])])
    return command


def build_vocab_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    languages = shared.get("languages", ["en", "da"])
    routing_root = Path(runtime["output_root"]) / "routing" / model_slug(model_entry) / build_mkqa_dataset_name(languages)
    command = [
        sys.executable,
        str(ANALYSES_DIR / "analyze_mkqa_vocab_specialization.py"),
        "--routing-root",
        str(routing_root),
        "--sources",
        str(shared.get("vocab_sources", "prompt,predicted,ground_truth")),
        "--top-tokens",
        str(shared.get("top_tokens", 20)),
        "--output-root",
        str(Path(runtime["output_root"]) / "vocab_specialization" / model_slug(model_entry)),
    ]
    tokenizer_path = shared.get("tokenizer_path") or model_entry.get("tokenizer_path") or model_entry.get("model_path")
    if tokenizer_path is not None:
        command.extend(["--tokenizer-path", str(tokenizer_path)])
    return command


def build_domain_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    languages = shared.get("languages", ["en", "da"])
    routing_root = Path(runtime["output_root"]) / "routing" / model_slug(model_entry) / build_mkqa_dataset_name(languages)
    return [
        sys.executable,
        str(ANALYSES_DIR / "analyze_mkqa_domain_specialization.py"),
        "--routing-root",
        str(routing_root),
        "--sources",
        str(shared.get("vocab_sources", "prompt,predicted,ground_truth")),
        "--output-root",
        str(Path(runtime["output_root"]) / "domain_specialization" / model_slug(model_entry)),
    ]


def build_weights_command(model_entry: dict, shared: dict, runtime: dict) -> list[str]:
    model_output_root = Path(runtime["output_root"]) / "weight_analysis" / model_slug(model_entry)
    summary_path = model_output_root / "weight_analysis_summary.jsonl"
    details_dir = model_output_root / "details"

    command = [
        sys.executable,
        str(WEIGHTS_SCRIPT),
        "--output-jsonl",
        str(summary_path),
        "--output-dir",
        str(details_dir),
        "--device",
        str(runtime.get("device", "cpu")),
        "--dtype",
        str(runtime.get("dtype", "auto")),
        "--public-expert-idx",
        str(shared.get("public_expert_idx", 0)),
    ]
    if model_entry.get("model_path"):
        command.extend(["--model-path", str(model_entry["model_path"])])
    else:
        command.extend(["--model-path", str(Path(shared["model_root"]) / str(model_entry["model_name"]))])
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
    runtime.setdefault("output_root", str(PROJECT_ROOT / "eval_results" / "mkqa" / "full" / "mixed"))
    shared = config.get("shared", {})
    models = resolve_model_entries(config)
    analyses = config.get("analyses", {})

    if not models:
        raise ValueError("No enabled models were defined in the MKQA suite config.")

    Path(runtime["output_root"]).mkdir(parents=True, exist_ok=True)
    all_commands = []

    for model_entry in models:
        if analyses.get("routing", {}).get("enabled", True):
            all_commands.append(build_routing_command(model_entry, shared, runtime))
        if analyses.get("vocab_specialization", {}).get("enabled", True):
            all_commands.append(build_vocab_command(model_entry, shared, runtime))
        if analyses.get("domain_specialization", {}).get("enabled", True):
            all_commands.append(build_domain_command(model_entry, shared, runtime))
        if analyses.get("weights", {}).get("enabled", False):
            all_commands.append(build_weights_command(model_entry, shared, runtime))

    manifest_path = Path(runtime["output_root"]) / "mkqa_analysis_suite_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config_path": str(Path(args.config).resolve()),
                "num_commands": len(all_commands),
                "commands": all_commands,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for command in all_commands:
        run_command(command, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
