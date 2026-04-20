import json
from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "eval" / "benchmarks" / "mkqa" / "data" / "mkqa_da_en_subset.json"


ds = load_dataset("mkqa", split="train", streaming=True)

pairs = []

for ex in ds:
    q_da = ex["queries"].get("da")
    q_en = ex["queries"].get("en")

    a_da = ex["answers"].get("da")
    a_en = ex["answers"].get("en")

    if q_da and q_en and a_da and a_en:
        pairs.append({
            "question_da": q_da,
            "question_en": q_en,
            "answer_da": a_da[0]["text"],
            "answer_en": a_en[0]["text"],
        })

    if len(pairs) >= 200:   
        break

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)
