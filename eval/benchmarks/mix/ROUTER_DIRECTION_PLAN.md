# Router Direction / Expert Alignment Plan

This document defines a separate analysis track for studying how the router
represents experts and how pre-router activations align with those expert
directions.

The purpose of this track is to answer questions such as:

- Is the public expert geometrically close to many different inputs?
- Are expert router directions distinct or highly redundant?
- Do Danish inputs align with the Danish expert direction, or mostly with public?
- Does router tuning change the alignment between inputs and expert directions?
- Are actual routing choices consistent with the strongest router-direction alignments?

This track is intentionally separate from:

- `routing`: what the router did
- `latent_space`: what the input or hidden-state geometry looks like
- `intervention`: whether a direction or component matters causally

The router-direction track connects the latent geometry to the router's
parameterization.

## Core Objects

At each selected layer:

1. `pre_router_state`
- the hidden-state-like tensor passed into `layer.mlp.gate`

2. `router_weight_vectors`
- the per-expert rows of `layer.mlp.gate.weight`

3. `router_alignment_scores`
- dot products or cosine similarities between `pre_router_state` and each expert vector

4. `actual_routing_choice`
- the experts selected by the gate for the same inputs

## Main Questions

This track should support:

1. Router weight similarity
- cosine similarity between expert router vectors
- norm differences between experts
- clustering of expert router vectors

2. Input-to-expert alignment
- which experts each dataset/language aligns with before routing
- how strong the top-1 and top-2 alignment margins are
- whether alignment is diffuse or concentrated

3. Alignment vs actual routing
- whether the routed experts match the strongest alignment directions
- whether some experts are under-used even when alignment suggests they should win
- whether public dominance is reflected in geometric alignment or emerges later from routing competition

4. RT vs non-RT comparison
- how router expert directions differ between checkpoints
- whether pre-router states align differently under RT
- whether RT increases separation between expected experts

## Minimal Artifacts

Per selected layer, save:

- `router_weight_matrix`
  shape: `(num_experts, hidden_dim)`

- `alignment_summary`
  per dataset/language:
  - mean alignment per expert
  - top-1 aligned expert frequency
  - top-2 aligned expert frequency
  - mean top-1 alignment
  - mean top-2 alignment
  - mean alignment margin

- optional per-example compact artifact:
  - `example_id`
  - `dataset_name`
  - `language`
  - `layer`
  - `top1_aligned_expert`
  - `top2_aligned_expert`
  - `top1_alignment`
  - `top2_alignment`
  - `alignment_margin`
  - `actual_topk_experts`

Recommended storage:

- compact `.npz` for matrices
- compact JSONL for per-example or per-group summaries

## Suggested Output Layout

Use a separate root:

```text
eval_results/mix/router_direction/<setting>/<model_name>/<dataset_name>/
```

Recommended files:

- `router_direction_summary.jsonl`
- `router_direction_records.jsonl`
- `router_weights.npz`
- `run_manifest.json`

## Recommended Metrics

Per layer:

- expert-weight cosine matrix
- expert-weight norm
- mean alignment by expert
- top-1 aligned expert histogram
- alignment entropy over experts
- alignment margin (top1 - top2)
- agreement between aligned top-1 and actual routed top-1

## First Focused Scope

Start with the 55B pair:

- `FlexOlmo-8x7B-1T-a4-55B-v2`
- `FlexOlmo-8x7B-1T-a4-55B-v2-rt`

Datasets:

- `mkqa_en_da`
- `gsm8k_subset`
- `pubmedqa_subset`

Layers:

- `early_mid_late_last`

## Interpretation Guide

If public is dominant because of router geometry, we may see:

- strong mean alignment to the public expert across many datasets
- public expert router vector close to many input states
- small margins between expected specialist experts and public

If RT improves expert specialization, we may see:

- larger alignment margins
- stronger dataset-specific alignment to expected experts
- higher agreement between aligned expert and routed expert
- more distinct router weight vectors or better separated alignment patterns

## Relationship To Other Tracks

- `latent_space` tells us what the input geometry looks like
- `router_direction` tells us how that geometry meets the router's expert directions
- `routing` tells us what the router actually chose
- `intervention` tells us whether those directions or choices matter causally
