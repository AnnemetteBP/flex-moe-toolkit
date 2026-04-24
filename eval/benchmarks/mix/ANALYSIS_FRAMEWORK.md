# Mix Analysis Framework

This document turns the current FlexOlmo / MoE investigation into a reusable analysis stack.

The main goal is to stop treating every new question as a new giant capture. Instead, each
question should map to a specific artifact tier with a clear storage budget, a clear set of
questions it can answer, and explicit limits on what it cannot answer.

## Core Principle

Save the minimum artifact that can answer the question.

In practice, the current work naturally splits into four tiers:

1. `routing`
2. `output`
3. `latent_space`
4. `intervention`

These tiers build on each other, but they should not be collapsed into one oversized JSONL dump.

## Tier 1: Routing

Purpose:
- describe what the router is doing
- compare RT vs non-RT
- test whether routing looks domain-sensitive rather than diffuse or default-heavy
- estimate whether expected experts gain usage on their matching domains

Minimal artifacts:
- dataset-level aggregate: `routing_summary.jsonl`
- example-level aggregate: `routing_analysis.jsonl`

Recommended minimal per-example fields:
- `example_id`
- `dataset_name`
- `language`
- `model_name`
- `run_label`
- `usage`
- `activation_counts`
- `coactivation_counts`
- `coactivation_matrix`
- `mean_top1_prob`
- `mean_top2_prob`
- `mean_top1_top2_margin`
- `mean_token_entropy`
- `mean_selected_expert_prob_mass`
- `offdiag_ratio`

What this tier can answer:
- is RT more selective than non-RT?
- is routing lower-entropy or higher-margin under RT?
- do expert-usage distributions change by dataset or language?
- does usage shift toward an expected expert on a domain-aligned dataset?
- are routing distributions similar or different across domains?

What this tier cannot answer well:
- token-level vocab specialization
- output-level correctness
- whether routing decisions improved generated answers
- whether experts are intrinsically indistinguishable in latent space

Current status in this repo:
- `routing_summary.jsonl` and `routing_analysis.jsonl` already fit this tier well
- `routing_records.jsonl` goes beyond this tier and is usually too heavy for routine transfer

## Tier 2: Output

Purpose:
- connect routing behavior to prediction quality
- determine whether a confident or domain-aligned routing pattern is actually useful
- compare routing on correct vs incorrect examples

Minimal artifacts:
- per-example routing-light fields from Tier 1
- generated prediction
- reference answer
- scoring result
- normalized scoring metadata

Recommended minimal per-example fields:
- all routing-light identifiers
- `prediction_text`
- `reference_answer`
- `scoring_mode`
- `is_correct`
- `score`
- optional dataset-specific details:
  - `normalized_prediction`
  - `normalized_reference`
  - `judge_label`

What this tier can answer:
- does RT improve correctness or quality?
- are expected-expert routing patterns associated with better outputs?
- are correct examples lower-entropy or higher-margin than incorrect ones?
- does the public expert dominate even when another expert should help?

What this tier cannot answer well:
- latent-space separability
- causal claims about forced expert selection

Current status in this repo:
- not yet formalized as a compact saved artifact
- some of the needed information existed inside earlier heavy records, but not in a reusable form

## Tier 3: Latent Space

Purpose:
- study whether experts and/or model states are distinguishable in representation space
- test whether RT changes separability
- compare domain clustering against routing behavior

Minimal artifacts:
- selected-layer hidden-state representations
- compact metadata linking each representation to example and dataset
- compact array storage such as `.npz`

Recommended artifacts:
- `metadata.jsonl`
- one compressed representation file per dataset, for example `latent_representations.npz`

Recommended saved vectors:
- mean-pooled prompt hidden state per selected layer
- last-token prompt hidden state per selected layer

What this tier can answer:
- do domains separate in hidden-state space?
- do RT and non-RT occupy different regions for the same dataset?
- are Danish and English prompts more or less separable at selected layers?
- does the learned representation support the routing story?

What this tier cannot answer by itself:
- whether the router made the optimal causal decision
- whether forcing a different expert would have improved the output

Important note:
- latent-space analysis is the right tier for the question "do experts become indistinguishable in
  latent space?"
- routing artifacts alone are not enough for that question

Current status in this repo:
- scaffolded via:
  - `runners/run_mix_latent_space.py`
  - `runners/run_mix_latent_space_suite.py`
  - `configs/mix_suite_config.55b_pair.latent_space.json`

## Tier 4: Intervention

Purpose:
- move from correlation to stronger causal evidence
- test whether routing choices are merely confident or actually useful

Typical interventions:
- force a chosen expert set
- mask a candidate expert
- compare public-only vs routed behavior
- compare native top-k vs restricted top-k

Minimal artifacts:
- routing-light fields
- prediction text and score
- intervention specification
- baseline vs intervention deltas

What this tier can answer:
- would performance improve if a domain-matching expert were forced?
- is the router missing a useful expert even when the expert exists?
- is public dominance actually necessary or just a default?

Current status in this repo:
- not yet implemented as a stable mix benchmark artifact

## Recommended Workflow

For future focused runs, use this order:

1. `routing`
2. `output`
3. `latent_space`
4. `intervention`

Only move to the next tier when the current tier leaves an important ambiguity.

## What We Already Learned

Using only Tier 1 artifacts from the transferred 55B pair:
- `55B-v2-rt` is lower-entropy than `55B-v2`
- `55B-v2-rt` has larger top-1/top-2 margin
- `55B-v2-rt` has higher selected-expert probability mass

That is enough to say RT changes routing structure.
It is not enough to say RT makes semantically correct token-level expert choices, and it is not
enough to say whether experts are intrinsically indistinguishable in latent space.

## Data Budget Guidance

Use smaller, purpose-built artifacts rather than one all-purpose dump.

Recommended default budgets:

- `routing`: `100-200` examples per dataset
- `output`: `100-200` examples per dataset
- `latent_space`: `50-100` examples per dataset
- `intervention`: small, carefully targeted subsets

For 55B focused runs, prefer:
- `mkqa_en_da`
- `gsm8k_subset`
- `pubmedqa_subset`
- optionally `mbpp_subset` for routing/output, but avoid it for early latent-space experiments

## Current Artifact Mapping

Current files and where they belong:

- `routing_summary.jsonl` -> Tier 1
- `routing_analysis.jsonl` -> Tier 1
- `routing_records.jsonl` -> mostly Tier 1.5 / Tier 2 support, but oversized in current form
- `latent_representations.npz` -> Tier 3
- future scoring artifact -> Tier 2
- future intervention artifact -> Tier 4

For the concrete FlexOlmo causal plan bridging Tier 3 and Tier 4, see:

- `FLEXOLMO_CAUSAL_INTERVENTION_PLAN.md`

## Practical Rule

If the question is:

- "What is the router doing?" -> use Tier 1
- "Did routing help the output?" -> use Tier 2
- "Are experts or states internally separable?" -> use Tier 3
- "Would a different routing choice have been better?" -> use Tier 4

That separation is the main thing that should keep future analysis runs both meaningful and
transferable.
