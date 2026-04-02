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
