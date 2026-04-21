from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval" / "benchmarks" / "mix" / "data"
MKQA_RAW_URL = "https://github.com/apple/ml-mkqa/raw/main/dataset/mkqa.jsonl.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare mix-suite subsets from Hugging Face datasets using a shared JSONL schema. "
            "The goal is to build compact, domain-focused evaluation packs that can be run in one pass "
            "per loaded model."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the JSONL subsets will be written.",
    )
    parser.add_argument(
        "--mkqa-pairs",
        type=int,
        default=500,
        help="Number of bilingual MKQA question pairs to export. Each pair yields two JSONL examples.",
    )
    parser.add_argument(
        "--gsm8k-samples",
        type=int,
        default=500,
        help="Number of GSM8K examples to export.",
    )
    parser.add_argument(
        "--mbpp-samples",
        type=int,
        default=500,
        help="Number of MBPP examples to export.",
    )
    parser.add_argument(
        "--pubmedqa-samples",
        type=int,
        default=500,
        help="Number of PubMedQA examples to export for the academic / pes2o-aligned domain.",
    )
    parser.add_argument(
        "--ag-news-samples",
        type=int,
        default=500,
        help="Number of AG News examples to export for the news domain.",
    )
    parser.add_argument(
        "--common-gen-samples",
        type=int,
        default=500,
        help="Number of CommonGen examples to export for the creative domain.",
    )
    parser.add_argument(
        "--eli5-samples",
        type=int,
        default=500,
        help="Number of ELI5 examples to export for the Reddit/informal domain.",
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write a manifest JSON describing the generated routing-diagnostic subsets.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_math_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker, 1)[1].strip()
    return answer_text.strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def iter_mkqa_examples():
    with urlopen(MKQA_RAW_URL) as response:
        with gzip.GzipFile(fileobj=response) as gz_handle:
            for raw_line in gz_handle:
                if not raw_line.strip():
                    continue
                yield json.loads(raw_line.decode("utf-8"))


def prepare_mkqa_en_da(output_root: Path, num_pairs: int) -> tuple[Path, int]:
    rows = []
    pair_count = 0
    for example in iter_mkqa_examples():
        q_da = example["queries"].get("da")
        q_en = example["queries"].get("en")
        a_da = example["answers"].get("da")
        a_en = example["answers"].get("en")
        if not (q_da and q_en and a_da and a_en):
            continue

        pair_id = f"mkqa_pair_{pair_count:05d}"
        rows.append(
            {
                "example_id": f"{pair_id}_en",
                "group_id": pair_id,
                "dataset_name": "mkqa_en_da",
                "domain": "language",
                "language": "en",
                "source_benchmark": "mkqa",
                "scoring_mode": "qa",
                "prompt": f"Question: {q_en}\nAnswer:",
                "reference_answer": a_en[0]["text"],
                "question": q_en,
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
                "prompt": f"Question: {q_da}\nAnswer:",
                "reference_answer": a_da[0]["text"],
                "question": q_da,
                "metadata": {"pair_language": "da"},
            }
        )
        pair_count += 1
        if pair_count >= num_pairs:
            break

    output_path = output_root / "mkqa_en_da_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_gsm8k(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for idx, example in enumerate(ds):
        rows.append(
            {
                "example_id": f"gsm8k_{idx:05d}",
                "group_id": f"gsm8k_{idx:05d}",
                "dataset_name": "gsm8k_subset",
                "domain": "math",
                "language": "en",
                "source_benchmark": "gsm8k",
                "scoring_mode": "qa",
                "prompt": f"Question: {example['question']}\nAnswer:",
                "reference_answer": normalize_math_answer(example["answer"]),
                "question": example["question"],
                "metadata": {
                    "full_answer": example["answer"],
                    "answer_extraction": "after_####",
                },
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "gsm8k_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_mbpp(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("mbpp", split="test")
    rows = []
    for idx, example in enumerate(ds):
        rows.append(
            {
                "example_id": f"mbpp_{idx:05d}",
                "group_id": f"mbpp_{idx:05d}",
                "dataset_name": "mbpp_subset",
                "domain": "code",
                "language": "en",
                "source_benchmark": "mbpp",
                "scoring_mode": "code_generation",
                "prompt": (
                    f"Task: {example['text']}\n"
                    "Write a Python solution that satisfies the specification."
                ),
                "reference_answer": strip_code_fences(example["code"]),
                "question": example["text"],
                "metadata": {
                    "test_list": example.get("test_list", []),
                    "challenge_test_list": example.get("challenge_test_list", []),
                },
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "mbpp_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_pubmedqa(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    rows = []
    for idx, example in enumerate(ds):
        context = " ".join(example.get("context", {}).get("contexts", []))
        question = example["question"]
        final_decision = example["final_decision"].strip().lower()
        rows.append(
            {
                "example_id": f"pubmedqa_{idx:05d}",
                "group_id": f"pubmedqa_{idx:05d}",
                "dataset_name": "pubmedqa_subset",
                "domain": "academic",
                "language": "en",
                "source_benchmark": "pubmed_qa",
                "scoring_mode": "classification",
                "prompt": (
                    "Read the biomedical context and answer the question with one of: yes, no, maybe.\n"
                    f"Context: {context}\nQuestion: {question}\nAnswer:"
                ),
                "reference_answer": final_decision,
                "question": question,
                "metadata": {
                    "long_answer": example.get("long_answer", ""),
                    "labels": ["yes", "no", "maybe"],
                },
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "pubmedqa_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_ag_news(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("ag_news", split="test")
    label_map = {
        0: "world",
        1: "sports",
        2: "business",
        3: "science_technology",
    }
    rows = []
    for idx, example in enumerate(ds):
        rows.append(
            {
                "example_id": f"ag_news_{idx:05d}",
                "group_id": f"ag_news_{idx:05d}",
                "dataset_name": "ag_news_subset",
                "domain": "news",
                "language": "en",
                "source_benchmark": "ag_news",
                "scoring_mode": "classification",
                "prompt": (
                    "Classify the following news article into one of: world, sports, business, science_technology.\n"
                    f"Article: {example['text']}\nLabel:"
                ),
                "reference_answer": label_map[int(example["label"])],
                "question": example["text"],
                "metadata": {"labels": list(label_map.values())},
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "ag_news_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_common_gen(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("common_gen", split="validation")
    rows = []
    for idx, example in enumerate(ds):
        concepts = ", ".join(example["concepts"])
        rows.append(
            {
                "example_id": f"common_gen_{idx:05d}",
                "group_id": f"common_gen_{idx:05d}",
                "dataset_name": "common_gen_subset",
                "domain": "creative",
                "language": "en",
                "source_benchmark": "common_gen",
                "scoring_mode": "generation",
                "prompt": (
                    "Write a short, natural sentence or two that uses all of the following concepts.\n"
                    f"Concepts: {concepts}\nResponse:"
                ),
                "reference_answer": example["target"],
                "question": concepts,
                "metadata": {"concepts": example["concepts"]},
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "common_gen_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def prepare_eli5(output_root: Path, num_samples: int) -> tuple[Path, int]:
    ds = load_dataset("eli5", split="train_asks")
    rows = []
    for idx, example in enumerate(ds):
        answers = example.get("answers", {}).get("text", [])
        answer = next((text.strip() for text in answers if text and text.strip()), None)
        title = example.get("title", "").strip()
        selftext = example.get("selftext", "").strip()
        if not title or not answer:
            continue
        prompt = f"Question: {title}\n"
        if selftext:
            prompt += f"Details: {selftext}\n"
        prompt += "Answer:"
        rows.append(
            {
                "example_id": f"eli5_{idx:05d}",
                "group_id": f"eli5_{idx:05d}",
                "dataset_name": "eli5_subset",
                "domain": "reddit",
                "language": "en",
                "source_benchmark": "eli5",
                "scoring_mode": "generation",
                "prompt": prompt,
                "reference_answer": answer,
                "question": title,
                "metadata": {"subreddit": example.get("subreddit", "")},
            }
        )
        if len(rows) >= num_samples:
            break

    output_path = output_root / "eli5_subset.jsonl"
    write_jsonl(output_path, rows)
    return output_path, len(rows)


def write_manifest(output_root: Path, entries: list[dict]) -> Path:
    manifest_path = output_root / "mix_manifest.json"
    manifest = {
        "schema_version": 1,
        "datasets": entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    dataset_jobs = [
        (
            "mkqa_en_da",
            lambda: prepare_mkqa_en_da(output_root, num_pairs=args.mkqa_pairs),
            {
                "domain": "language",
                "scoring_mode": "qa",
                "phase": 1,
                "notes": "Bilingual paired subset; each question pair yields one English and one Danish example.",
            },
        ),
        (
            "gsm8k_subset",
            lambda: prepare_gsm8k(output_root, num_samples=args.gsm8k_samples),
            {
                "domain": "math",
                "scoring_mode": "qa",
                "phase": 1,
                "notes": "Math subset using extracted final answers after the GSM8K #### marker.",
            },
        ),
        (
            "mbpp_subset",
            lambda: prepare_mbpp(output_root, num_samples=args.mbpp_samples),
            {
                "domain": "code",
                "scoring_mode": "code_generation",
                "phase": 1,
                "notes": "Code-generation subset. Evaluation will need a code-aware scoring path, not plain exact-match QA scoring.",
            },
        ),
        (
            "pubmedqa_subset",
            lambda: prepare_pubmedqa(output_root, num_samples=args.pubmedqa_samples),
            {
                "domain": "academic",
                "scoring_mode": "classification",
                "phase": 1,
                "notes": "Biomedical QA subset used as an academic / pes2o-aligned diagnostic domain.",
            },
        ),
        (
            "ag_news_subset",
            lambda: prepare_ag_news(output_root, num_samples=args.ag_news_samples),
            {
                "domain": "news",
                "scoring_mode": "classification",
                "phase": 2,
                "notes": "News-domain classification subset with deterministic labels.",
            },
        ),
        (
            "common_gen_subset",
            lambda: prepare_common_gen(output_root, num_samples=args.common_gen_samples),
            {
                "domain": "creative",
                "scoring_mode": "generation",
                "phase": 2,
                "notes": "Creative constrained-generation subset.",
            },
        ),
        (
            "eli5_subset",
            lambda: prepare_eli5(output_root, num_samples=args.eli5_samples),
            {
                "domain": "reddit",
                "scoring_mode": "generation",
                "phase": 2,
                "notes": "Informal Reddit-style QA subset.",
            },
        ),
    ]

    failed = []
    for name, builder, metadata in dataset_jobs:
        try:
            path, count = builder()
            manifest_entries.append(
                {
                    "name": name,
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "num_examples": count,
                    **metadata,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((name, str(exc)))

    if args.write_manifest:
        manifest_path = write_manifest(output_root, manifest_entries)
        print(f"Wrote manifest to {manifest_path}")

    for entry in manifest_entries:
        print(f"{entry['name']}: {entry['num_examples']} examples -> {entry['path']}")

    if failed:
        print("\nDatasets that failed to prepare:")
        for name, message in failed:
            print(f"- {name}: {message}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
