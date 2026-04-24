# FlexOlmo Causal Intervention Plan

This note maps the causal-representation idea directly onto the current repo.

The goal is to make the next analysis step:
- small
- reusable
- causal
- compatible with the existing mix benchmark layout

This plan is specifically for:
- `FlexOlmo-8x7B-1T-a4-55B-v2`
- `FlexOlmo-8x7B-1T-a4-55B-v2-rt`

## Main Questions

We want to separate four explanations:

1. the router is diffuse or poorly calibrated
2. RT improves routing without changing much in the representation
3. the expert contribution on top of the public backbone is genuinely useful
4. experts and public become too similar in practice

The causal intervention track is meant to bridge routing and latent-space analysis.

## What To Intervene On

Do **not** start with attention.

The first intervention target should be the residual stream after the MoE block in selected layers.

Why:
- it is closer to the expert contribution than generic hidden-state capture
- it directly tests whether an MoE-related difference direction matters
- it is much easier to interpret than arbitrary attention-head perturbations

## Existing Hook Points In This Repo

These are the most useful code locations already available:

### 1. FlexOlmo layer iteration

File:
- `src/flex_moe_toolkit/adapters/flex_olmo.py`

Function:
- `iter_flex_olmo_layers(model)`

Use:
- enumerate decoder layers reliably for hook registration

### 2. Router hook

File:
- `src/flex_moe_toolkit/adapters/flex_olmo.py`

Current hook location:
- `layer.mlp.gate.register_forward_hook(...)`

Use:
- collect per-layer router probabilities
- useful for aligning causal effects with routing behavior

### 3. Public-only / restricted routing modes

File:
- `src/flex_moe_toolkit/pipelines/flex_olmo.py`

Context managers:
- `backbone_only_mode(model)`
- `restricted_expert_mode(model, allowed_experts)`

Use:
- compare full routing vs public-only vs selected-expert routing
- intervention baseline for "is public dominance actually necessary?"

### 4. Prompt-level hidden-state capture

File:
- `eval/benchmarks/mix/runners/run_mix_latent_space.py`

Use:
- existing compact latent runner
- good template for dataset loading, model loading, tokenization, selected-layer handling, and compact output

## First Causal Experiment

Start with the smallest high-value experiment:

- models:
  - `55B-v2`
  - `55B-v2-rt`
- datasets:
  - `mkqa_en_da`
  - `gsm8k_subset`
  - `pubmedqa_subset`
- layers:
  - `8, 16, 24, -1`
- positions:
  - last prompt token first
- samples:
  - `50-100` per dataset

### Delta Definition

For the same input `x`, capture the chosen layer activation from both models:

- `h_v2(l, j)`
- `h_rt(l, j)`

Define:

- `delta_rt = h_rt - h_v2`

Then intervene in the RT model by removing or replacing the component of `h_rt` along `delta_rt`.

This follows the same causal logic as the paper-style intervention:
- keep the rest of the RT computation intact
- only replace the targeted subspace

### Expected Interpretation

If removing `delta_rt` raises the RT model loss on domain-aligned inputs:
- RT carries useful causal signal at that layer

If the effect is close to zero:
- RT may mostly change routing statistics without a strong representation-level effect at that point

If the effect is broad and not domain-specific:
- RT may be changing generic representation quality, not only domain specialization

## Second Causal Experiment

Within one model, compare domain directions.

For example in `55B-v2-rt`:

- mean Danish activation minus mean English activation on `mkqa_en_da`
- mean `gsm8k` activation minus mean non-math activation
- mean `pubmedqa` activation minus mean non-academic activation

Define:

- `delta_domain = mean(h_domain) - mean(h_other)`

Then remove that direction on the domain-matching examples and measure `delta_loss`.

Interpretation:
- if removing the domain direction hurts mainly on that domain, it likely carries domain-relevant signal

## Third Causal Experiment

Probe public dominance directly.

This experiment should be attempted after the first two:

1. compare full routing vs `public_only` using `restricted_expert_mode(...)`
2. compare full routing vs `backbone_only_mode(...)`

This gives a direct performance-oriented view of:
- how much experts add beyond the backbone
- whether public is effectively carrying most of the useful signal

This is not the same as the subspace intervention above, but it is a valuable companion baseline.

## Recommended Runner Structure

Add a new runner rather than extending the current giant routing dataset pipeline.

Recommended files:

- `eval/benchmarks/mix/runners/run_mix_causal_intervention.py`
- `eval/benchmarks/mix/runners/run_mix_causal_intervention_suite.py`
- `eval/benchmarks/mix/configs/mix_suite_config.55b_pair.causal_intervention.json`

Do **not** store this in `routing_records.jsonl`.

This should be a compact, purpose-built artifact.

## Recommended Capture Strategy

For each model / dataset / example / layer:

1. run baseline forward pass
2. capture activation at the chosen hook point
3. construct the intervention direction
4. rerun with a forward hook that applies the subspace replacement
5. measure baseline loss and intervened loss

This should operate on teacher-forced loss first, not generation.

Reason:
- much cheaper
- easier to compare
- cleaner causal signal

## Suggested Hook Target

First implementation target:
- layer output after `layer.mlp`

Reason:
- close to the MoE block
- directly tied to expert mixture output
- easier to interpret than a generic hidden-state tensor

If direct post-MoE capture is awkward in the current model implementation, fall back to:
- block output hidden state

But note that this is slightly less specific.

## Suggested Intervention Function

For activation `h` and normalized direction `u = delta / ||delta||`:

- projection coefficient:
  - `alpha = <h, u>`
- projected component:
  - `proj = alpha * u`

Two useful interventions:

1. removal:
- `h_tilde = h - proj`

2. replacement with source-model component:
- `h_tilde = P_delta(h_source) + (I - P_delta)(h_target)`

Start with removal first.
It is simpler and easier to debug.

## Minimal Saved Output

Do not save token dumps or full hidden-state tensors.

Save one compact JSONL row per:
- example
- layer
- delta type

Recommended fields:

- `example_id`
- `dataset_name`
- `language`
- `model_name`
- `comparison_model_name`
- `layer`
- `position_policy`
- `delta_type`
- `delta_norm`
- `baseline_loss`
- `intervened_loss`
- `delta_loss`
- `baseline_top1_prob`
- `baseline_margin`
- `baseline_entropy`
- `metadata`

Optional:
- `cosine_to_domain_mean`
- `cosine_to_public_direction`

This should stay small enough to transfer comfortably.

## Baselines

Always include a non-trivial baseline.

Do **not** use arbitrary Gaussian random vectors.

Use:
- activation difference vectors between two random real examples from the same model

Reason:
- they are more likely to lie in a subspace the model actually uses
- this matches the logic from the cited paper more closely

## First Priority Comparison Table

If only one compact experiment is run first, it should be:

- models:
  - `55B-v2`
  - `55B-v2-rt`
- dataset:
  - `mkqa_en_da`
- splits:
  - Danish
  - English
- layers:
  - `8, 16, 24, -1`
- intervention:
  - remove `delta_rt = h_rt - h_v2` from `h_rt`
- metric:
  - `delta_loss`

This is the fastest route to answering:
- does RT carry useful causal signal?
- is the effect stronger on Danish than English?

## How This Fits The Four-Tier Framework

This causal track sits between:
- Tier 3: `latent_space`
- Tier 4: `intervention`

It uses latent-space differences, but the output is an intervention artifact.

That means:
- hidden-state capture alone is not enough
- routing-only files are not enough
- but the result should still be saved compactly like a Tier 4 artifact

## What This Should Clarify

If this works, it should let us say something much stronger than:
- "RT has lower entropy"

It should help answer:
- does RT add causally useful domain signal?
- is public still doing most of the work?
- are experts adding something real beyond the frozen backbone?
- are apparent routing gains just calibration, or do they reflect a real representational advantage?
