from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mkqa_en_da_subset.jsonl"
)
DEFAULT_PAIRED_JSON_OUTPUT_PATH = (
    PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data" / "mkqa_da_en_subset.json"
)
MKQA_RAW_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a bilingual English/Danish MKQA subset for the mix suite "
            "using the shared JSONL schema."
        )
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the JSONL subset will be written.",
    )
    parser.add_argument(
        "--paired-json-output-path",
        type=Path,
        default=DEFAULT_PAIRED_JSON_OUTPUT_PATH,
        help="Path where the paired legacy JSON export will be written.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=500,
        help="Number of bilingual English/Danish question pairs to export.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_mkqa_examples():
    with urlopen(MKQA_RAW_URL) as response:
        with gzip.GzipFile(fileobj=response) as gz_handle:
            for raw_line in gz_handle:
                if not raw_line.strip():
                    continue
                yield json.loads(raw_line.decode("utf-8"))


def main() -> int:
    args = parse_args()

    rows = []
    paired_rows = []
    pair_count = 0

    for example in iter_mkqa_examples():
        question_da = example["queries"].get("da")
        question_en = example["queries"].get("en")
        answer_da = example["answers"].get("da")
        answer_en = example["answers"].get("en")

        if not (question_da and question_en and answer_da and answer_en):
            continue

        pair_id = f"mkqa_pair_{pair_count:05d}"
        paired_rows.append(
            {
                "pair_id": pair_id,
                "question_da": question_da,
                "question_en": question_en,
                "answer_da": answer_da[0]["text"],
                "answer_en": answer_en[0]["text"],
            }
        )
        rows.append(
            {
                "example_id": f"{pair_id}_en",
                "group_id": pair_id,
                "dataset_name": "mkqa_en_da",
                "domain": "language",
                "language": "en",
                "source_benchmark": "mkqa",
                "scoring_mode": "qa",
                "prompt": f"Question: {question_en}\nAnswer:",
                "reference_answer": answer_en[0]["text"],
                "question": question_en,
                "metadata": {"pair_language": "en"},
            }
        )
        rows.append(
            {
                "example_id": f"{pair_id}_da",
                "group_id": pair_id,
                "dataset_name": "mkqa_en_da",
                "domain": "language",
                "language": "da",
                "source_benchmark": "mkqa",
                "scoring_mode": "qa",
                "prompt": f"Question: {question_da}\nAnswer:",
                "reference_answer": answer_da[0]["text"],
                "question": question_da,
                "metadata": {"pair_language": "da"},
            }
        )

        pair_count += 1
        if pair_count >= args.num_pairs:
            break

    write_jsonl(args.output_path, rows)
    write_json(args.paired_json_output_path, paired_rows)
    print(
        f"Wrote {pair_count} MKQA bilingual pairs ({len(rows)} examples) to {args.output_path} "
        f"and {args.paired_json_output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
