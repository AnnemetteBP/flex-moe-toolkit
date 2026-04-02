import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import torch

log = logging.getLogger(__name__)


MOE_EXPERT_PATTERNS = (
    "feed_forward_moe.experts.mlp.w1",
    "feed_forward_moe.experts.mlp.w2",
    "feed_forward_moe.experts.mlp.w3",
)

MOE_ROUTER_PATTERNS = (
    "feed_forward_moe.router.weight",
    "feed_forward_moe.router.expert_bias",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split an unsharded FlexOlmo/OLMoE-style checkpoint into shared "
            "backbone weights and per-expert tensor slices."
        )
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing `model.pt` and optionally `config.json`.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for split checkpoints.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Override number of experts if it cannot be inferred from config.json.",
    )
    parser.add_argument(
        "--public-expert-idx",
        type=int,
        default=0,
        help="Index to tag as the public expert in the metadata output.",
    )
    return parser.parse_args()


def load_config(input_dir: Path):
    config_path = input_dir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)


def infer_num_experts(config_dict, state_dict, user_value=None):
    if user_value is not None:
        return user_value

    if config_dict is not None:
        candidates = [
            ("model", "block", "feed_forward_moe", "num_experts"),
            ("model", "num_experts"),
            ("num_experts",),
        ]
        for path in candidates:
            current = config_dict
            found = True
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    found = False
                    break
                current = current[key]
            if found and isinstance(current, int):
                return current

    for key, value in state_dict.items():
        if any(pattern in key for pattern in MOE_ROUTER_PATTERNS) and value.ndim >= 1:
            return value.shape[0]

    raise ValueError("Could not infer `num_experts`. Pass it explicitly with `--num-experts`.")


def load_state_dict(input_dir: Path):
    model_path = input_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        return state_dict["model"], True
    return state_dict, False


def split_expert_tensor(tensor: torch.Tensor, num_experts: int):
    if tensor.shape[0] % num_experts != 0:
        raise ValueError(
            f"Cannot split tensor of shape {tuple(tensor.shape)} into {num_experts} equal expert chunks."
        )
    chunk = tensor.shape[0] // num_experts
    return [tensor[i * chunk : (i + 1) * chunk].clone() for i in range(num_experts)]


def split_router_tensor(tensor: torch.Tensor, num_experts: int):
    if tensor.shape[0] != num_experts:
        raise ValueError(
            f"Expected router tensor first dimension {tensor.shape[0]} to equal num_experts={num_experts}."
        )
    return [tensor[i : i + 1].clone() for i in range(num_experts)]


def is_expert_tensor(key: str):
    return any(pattern in key for pattern in MOE_EXPERT_PATTERNS)


def is_router_tensor(key: str):
    return any(pattern in key for pattern in MOE_ROUTER_PATTERNS)


def save_checkpoint(path: Path, payload: dict, wrapped: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if wrapped:
        torch.save({"model": payload}, path)
    else:
        torch.save(payload, path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = load_config(args.input_dir)
    state_dict, wrapped = load_state_dict(args.input_dir)
    num_experts = infer_num_experts(config_dict, state_dict, args.num_experts)

    log.info("Splitting checkpoint from %s with %d experts", args.input_dir, num_experts)

    shared_state = {}
    expert_states = [dict() for _ in range(num_experts)]

    for key, value in state_dict.items():
        if is_expert_tensor(key):
            for expert_idx, expert_value in enumerate(split_expert_tensor(value, num_experts)):
                expert_states[expert_idx][key] = expert_value
        elif is_router_tensor(key):
            for expert_idx, expert_value in enumerate(split_router_tensor(value, num_experts)):
                expert_states[expert_idx][key] = expert_value
        else:
            shared_state[key] = value.clone() if torch.is_tensor(value) else deepcopy(value)

    backbone_dir = args.output_dir / "backbone"
    save_checkpoint(backbone_dir / "model.pt", shared_state, wrapped=wrapped)

    if config_dict is not None:
        backbone_config = deepcopy(config_dict)
        with open(backbone_dir / "config.json", "w") as f:
            json.dump(backbone_config, f, indent=2)

    metadata = {
        "num_experts": num_experts,
        "public_expert_idx": args.public_expert_idx,
        "input_dir": str(args.input_dir),
        "backbone_model_path": str(backbone_dir / "model.pt"),
        "experts": [],
    }

    experts_dir = args.output_dir / "experts"
    for expert_idx, expert_state in enumerate(expert_states):
        expert_dir = experts_dir / f"expert_{expert_idx}"
        save_checkpoint(expert_dir / "model.pt", expert_state, wrapped=wrapped)

        if config_dict is not None:
            expert_config = deepcopy(config_dict)
            with open(expert_dir / "config.json", "w") as f:
                json.dump(expert_config, f, indent=2)

        metadata["experts"].append(
            {
                "expert_idx": expert_idx,
                "kind": "public" if expert_idx == args.public_expert_idx else "domain",
                "model_path": str(expert_dir / "model.pt"),
            }
        )

    with open(args.output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved shared backbone to %s", backbone_dir / "model.pt")
    log.info("Saved %d expert checkpoints under %s", num_experts, experts_dir)
    log.info("Saved metadata to %s", args.output_dir / "split_metadata.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
