# flex-moe-toolkit


## Installation
### From source (recommended for development):
#### 1. git clone https://github.com/AnnemetteBP/flex-moe-toolkit.git
#### 2. cd flex-moe-toolkit
#### 3. pip install -e .
### From GitHub:
#### pip install git+https://github.com/AnnemetteBP/flex-moe-toolkit.git

## Analysis setup
The current toolkit is set up to support the first replication phase: capture router logits, compute routing entropy and load balance, extract top-k expert selections, and build layer-by-expert usage matrices before moving on to FlexOlmo-specific modifications.

```python
from flex_moe_toolkit.adapters.hf_moe import HFMoEAdapter
from flex_moe_toolkit.pipelines.routing import analyze_model_routing

adapter = HFMoEAdapter()
results = analyze_model_routing(
    model=model,
    inputs=inputs,
    adapter=adapter,
    top_k=5,
)

print(results["entropy"])
print(results["load_balance"])
print(results["expert_usage"])
print(results["layer_expert_matrix"].shape)
```

For models that do not expose `output_router_logits=True` in a Hugging Face-compatible way, use router hooks from [src/flex_moe_toolkit/utils/hooks.py](/media/am/AM/flex-moe-toolkit/src/flex_moe_toolkit/utils/hooks.py).

## FlexOlmo-specific setup
The copied files under `models/flex_olmo` in this repo are reference snapshots only. For real runs, the active `FlexOlmo` implementation should come from the installed `transformers` checkout in your current environment.

Use the FlexOlmo adapter and mode helpers to compare the combined model against backbone-only and expert-restricted runs.

```python
from flex_moe_toolkit.pipelines.flex_olmo import analyze_flex_olmo_modes

results = analyze_flex_olmo_modes(
    model=model,
    inputs=inputs,
    public_expert_idx=0,
)

print(results["full"]["entropy"])
print(results["backbone_only"]["entropy"])
print(results["public_only"]["expert_usage"])
```

`public_expert_idx=0` is treated as the default FlexOlmo convention. If your checkpoint orders experts differently, pass the correct index explicitly.

## Fake FlexOlmo smoke tests
Use the lightweight fake model to rehearse routing capture, metrics, and dataset-style evaluation before moving to a remote GPU run with real checkpoints.

```python
python3 scripts/flex_olmo/utils/run_fake_flex_olmo_pipeline.py
python3 scripts/flex_olmo/utils/evaluate_fake_flex_olmo_dataset.py
```

The dataset evaluation script uses a small local benchmark fixture shaped like MCQ / AGIEval / BBH / EuroEval-style multiple-choice tasks and writes:

- `outputs/flex_olmo/combined_flex/fake_eval_router_activity.jsonl`
- `outputs/flex_olmo/combined_flex/fake_eval_summary.jsonl`
- `outputs/flex_olmo/combined_flex/fake_eval_expert_upset.png`

The evaluation runs three expert-availability settings in one pass:

- `top2_active`: experts `{0,1}`
- `top4_active`: experts `{0,1,2,3}`
- `top7_active`: experts `{0,1,2,3,4,5,6}`

Each example record logs activated expert combinations per layer, the intersection and union of those layer-level expert sets, and pairwise / mean layer IoU scores. The upset plot compares run-specific activated expert combinations and annotates them with the mean overlap of expert sets across layers.

The accuracy from this fake benchmark is only a smoke test for the pipeline. The main purpose is to verify routing logs, expert-combination aggregation, multi-run upset plotting, and figure generation before swapping in a real model and real tokenizer inputs.

## Real FlexOlmo eval datasets
Use the real-model runner when you want to evaluate one or many JSONL datasets against a real FlexOlmo checkpoint, save per-run routing outputs, and run on CPU or GPU.

```python
python3 scripts/flex_olmo/utils/evaluate_flex_olmo_datasets.py \
  --model-path /path/to/flex_olmo_checkpoint \
  --tokenizer-path /path/to/tokenizer \
  --dataset /path/to/eval_a.jsonl \
  --dataset /path/to/eval_b.jsonl \
  --device cuda:0 \
  --output-root outputs/flex_olmo/eval_runs
```

The runner is additive and does not replace the fake smoke-test scripts. It writes one directory per dataset and one subdirectory per run, including:

- `public_only`
- `single_expert_<idx>` for each non-public expert
- `combined_top2`
- `combined_top4`
- `combined_top7`

Each run directory contains `eval_records.jsonl`, `eval_summary.jsonl`, and `routing_analysis.jsonl`. An overall `run_manifest.json` is also written under the chosen output root.

The JSONL input can be either:

- Multiple-choice: `question`, `choices`, `correct_choice_idx`, with optional `context`, `benchmark`, `language`, `task_type`
- Target scoring: `prompt`, plus one of `target`, `answer`, `completion`, or `reference`

For remote GPU runs on UCloud over SSH, pass `--device cuda` or `--device cuda:0`. If you want a quick smoke test before a full run, add `--max-examples N`.

When the script starts, it prints the exact `FlexOlmoForCausalLM` source file being used. This is a good sanity check that the run is using your installed `transformers` fork rather than the reference copies in this repo.

```python
python3 scripts/flex_olmo/utils/split_state_dict.py \
  --input-dir /path/to/unsharded_checkpoint \
  --output-dir /path/to/split_checkpoint \
  --public-expert-idx 0
```

## UCloud SSH launcher
For batch router analysis on UCloud, use the suite runner together with the SSH launcher. The launcher is config-driven and supports multiple remote model paths while keeping `model_name` and `model_path` in every JSONL record.

1. Copy [ucloud_router_suite.env.example](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/ucloud_router_suite.env.example) to your own config file.
2. Copy [ucloud_models.txt.example](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/ucloud_models.txt.example) and [ucloud_datasets.txt.example](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/ucloud_datasets.txt.example) if you want path lists.
3. Point `REMOTE_EVAL_SCRIPT` at the eval pipeline path you want to use on UCloud.
4. Run:

```bash
bash scripts/flex_olmo/utils/run_ucloud_router_suite.sh \
  scripts/flex_olmo/utils/ucloud_router_suite.env \
  --dry-run
```

Remove `--dry-run` once the generated SSH command looks right.

## Config-driven multi-analysis runs
Use a single JSON config to choose which prepared analyses to run independently, point at combined-model paths and expert-model paths on UCloud, and keep every analysis writing its own JSONL outputs.

Start from [flex_olmo_analysis_config.example.json](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/flex_olmo_analysis_config.example.json) and enable only the analyses you want:

- `router_suite`
- `weight_analysis`
- `router_saturation`

Run locally or remotely:

```bash
python3 scripts/flex_olmo/utils/run_flex_olmo_analyses.py \
  --config scripts/flex_olmo/utils/flex_olmo_analysis_config.json \
  --dry-run
```

For SSH-driven UCloud runs, use [run_ucloud_flex_analyses.sh](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/run_ucloud_flex_analyses.sh) with [ucloud_flex_analyses.env.example](/media/am/AM/flex-moe-toolkit/scripts/flex_olmo/utils/ucloud_flex_analyses.env.example). The remote config file should contain the UCloud model paths, the remote eval-dataset paths, and the enabled analyses for that run.

## Weight analysis
Static weight analysis is separate from the router-logit pipeline. Use the weight-analysis runner to inspect router weights and expert weights across one or many checkpoints.

```bash
python3 scripts/flex_olmo/utils/analyze_flex_olmo_weights.py \
  --model-path /path/to/flexolmo-a2-5B \
  --model-path /path/to/flexolmo-a7-rt-15B \
  --output-jsonl outputs/flex_olmo/weight_suite_summary.jsonl \
  --output-dir outputs/flex_olmo/weight_details
```

The JSONL summary records include model identifiers plus aggregate statistics such as mean router-weight norm and mean off-diagonal expert similarity. The optional detail files include per-layer router similarity matrices, expert similarity matrices, and distances from the designated public expert.

## Router saturation
Router saturation compares top-k expert selections at checkpoint `t` against a final checkpoint `T` and logs the overlap as a separate analysis without changing the existing routing outputs.

```bash
python3 scripts/flex_olmo/utils/compare_flex_olmo_checkpoints.py \
  --checkpoint-path /path/to/flexolmo-a7-v1-5B-step1000 \
  --checkpoint-path /path/to/flexolmo-a7-v1-5B-step5000 \
  --final-checkpoint-path /path/to/flexolmo-a7-v1-5B-final \
  --dataset /path/to/eval/math.jsonl \
  --output-root outputs/flex_olmo/router_saturation
```

This writes per-example and per-checkpoint summary JSONL files under the chosen output root, plus a consolidated `router_saturation_summary.jsonl`.

## Plotting package
Plotting can now live separately from the analysis pipelines under [src/flex_moe_toolkit/plotting](/media/am/AM/flex-moe-toolkit/src/flex_moe_toolkit/plotting).

The first standalone entrypoint is:

```bash
python3 scripts/flex_olmo/utils/plot_routing_jsonl.py \
  --routing-analysis-jsonl outputs/flex_olmo/combined_flex/routing_analysis.jsonl \
  --eval-records-jsonl outputs/flex_olmo/combined_flex/fake_eval_router_activity.jsonl \
  --output-dir outputs/flex_olmo/combined_flex/replotted
```

This reads existing JSONL outputs and regenerates routing figures without rerunning the model.
