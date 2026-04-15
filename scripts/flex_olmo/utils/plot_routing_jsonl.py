from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flex-moe-toolkit-mpl")

from flex_moe_toolkit.plotting.routing import plot_routing_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate routing figures from previously saved JSONL outputs."
    )
    parser.add_argument(
        "--routing-analysis-jsonl",
        required=True,
        help="Path to `routing_analysis.jsonl`.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where figures will be saved.",
    )
    parser.add_argument(
        "--eval-records-jsonl",
        help="Optional path to evaluation records JSONL for expert-combination upset plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = plot_routing_outputs(
        routing_analysis_path=args.routing_analysis_jsonl,
        output_dir=args.output_dir,
        eval_records_path=args.eval_records_jsonl,
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
