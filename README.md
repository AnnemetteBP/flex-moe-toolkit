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
For the local `models/flex_olmo` implementation, use the FlexOlmo adapter and mode helpers to compare the combined model against backbone-only and expert-restricted runs.

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

```python
python3 scripts/flex_olmo/utils/split_state_dict.py \
  --input-dir /path/to/unsharded_checkpoint \
  --output-dir /path/to/split_checkpoint \
  --public-expert-idx 0
```
