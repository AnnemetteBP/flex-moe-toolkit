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

```python
python3 scripts/flex_olmo/utils/split_state_dict.py \
  --input-dir /path/to/unsharded_checkpoint \
  --output-dir /path/to/split_checkpoint \
  --public-expert-idx 0
```
