# 55B Follow-Up Plan

This note splits the current investigation into two explicit tracks for:
- `FlexOlmo-8x7B-1T-a4-55B-v2`
- `FlexOlmo-8x7B-1T-a4-55B-v2-rt`

The goal is to avoid another oversized capture while still answering the questions that matter.

For the broader reusable structure behind this split, see:
- `eval/benchmarks/mix/ANALYSIS_FRAMEWORK.md`

## Track 1: Routing-Light

Purpose:
- test whether routing behavior makes semantic/domain sense
- compare RT vs non-RT under the same domains
- estimate whether RT shifts usage toward expected experts

Suggested datasets:
- `mkqa_en_da`
- `gsm8k_subset`
- `mbpp_subset`
- `pubmedqa_subset`

Suggested sample count:
- `100-150` examples per dataset

Suggested capture:
- keep `routing_analysis.jsonl`
- keep `routing_summary.jsonl`
- keep `routing_records.jsonl` only in a lighter form if we later add that mode
- do **not** capture hidden states
- do **not** capture raw router tensors

Ready-to-run config:
- `eval/benchmarks/mix/configs/mix_suite_config.55b_pair.routing_light.json`

## Track 2: Latent Space

Purpose:
- test whether experts become hard to distinguish in representation space
- compare representation separability for RT vs non-RT
- check whether internal differences align with routing behavior

This is **not** covered by the current `mix` runner and should be treated as a separate pipeline.

Suggested datasets:
- `mkqa_en_da`
- `gsm8k_subset`
- `pubmedqa_subset`

Suggested sample count:
- `50-100` examples per dataset

Suggested layers:
- early: `0-3`
- middle: around half depth
- late: final `3-4` layers

Suggested saved artifacts:
- hidden states for selected layers only
- example metadata (`example_id`, `dataset_name`, `language`)
- optionally expert output activations if available from the model code

Recommended output root:
- `eval_results/mix/focused/55b_pair/latent_space`

## Practical Rule

If the question is about:
- "what is the router doing?" -> use the routing-light track
- "are experts internally too similar?" -> use the latent-space track

Do not try to answer both with one giant JSONL dump again.
