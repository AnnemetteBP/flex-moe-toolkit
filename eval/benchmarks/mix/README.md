# Mix Suite

This benchmark suite is intended as a shared home for compact mixed-domain evaluation data and
related analysis runs.

The design goal is:
- load a model once
- run multiple compact, domain-focused subsets while weights stay in memory
- compare routing behavior against domains where specialists are expected to help

## Directory Layout

- `data/`: shared JSONL datasets plus the suite manifest
- `configs/`: config-driven multi-model suite runs
- `runners/`: manifest-driven execution entrypoints

The main runner is `runners/run_mix_analysis.py`.
It loads one FlexOlmo checkpoint once, then iterates over every dataset listed in the manifest.

For multi-model runs, use `runners/run_mix_suite.py` with a JSON config similar to:
- `configs/mix_suite_config.example.json`

## Current Shared JSONL Schema

Each line should look like:

```json
{
  "example_id": "gsm8k_00001",
  "group_id": "gsm8k_00001",
  "dataset_name": "gsm8k_subset",
  "domain": "math",
  "language": "en",
  "source_benchmark": "gsm8k",
  "scoring_mode": "qa",
  "prompt": "Question: ...\nAnswer:",
  "reference_answer": "42",
  "question": "...",
  "metadata": {}
}
```

The shared `mix_manifest.json` can also define per-dataset execution defaults:
- `prompting`: whether to keep the prompt raw or wrap it with a chat template
- `generation`: dataset-specific generation limits such as `max_new_tokens`
- `tokenization`: prompt/reference special-token handling and decode behavior

Current default policy in the mix suite:
- prompts stay as raw instruction text unless chat templating is explicitly enabled
- classification tasks use short generation budgets
- code and open-ended generation tasks use larger budgets
- decoded generations skip special tokens by default

## Phase 1: Cleanly Scorable Subsets

- `mkqa_en_da`: 500 bilingual pairs -> 1000 examples
- `gsm8k_subset`: 500 examples
- `mbpp_subset`: 500 examples if you already have or add a stable code-aware scoring path
- `pubmedqa_subset`: 500 examples as an academic / `pes2o`-aligned diagnostic domain

These are intended as the first routing-diagnostics pack because they are the easiest to interpret and score consistently.

## Phase 2: More Complex Judged/Generative Subsets

- `ag_news_subset`: 500 examples for the news domain
- `common_gen_subset`: 500 examples for a creative constrained-generation domain
- `eli5_subset`: 500 examples for an informal Reddit-style domain

Only add these once the scoring path is reliable enough that routing patterns can be compared to performance without a lot of ambiguity.

Keep the same JSONL schema and add every dataset to one shared manifest so a single model load can cover the full mix suite in one run.

## Output Layout

The suite runner writes results like:

```text
eval_results/mix/full/<collection>/<model_name>/<dataset_name>/<run_label>/
```

Each dataset directory also receives a `run_manifest.json`, and each model directory receives a `mix_suite_manifest.json`.
