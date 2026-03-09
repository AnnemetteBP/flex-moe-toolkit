import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_moe_toolkit.capture import capture_router_logits
from flex_moe_toolkit.routing import selected_experts, expert_load



model_name = "allenai/Flex-math-2x7B-1T"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)

prompt = "Solve: 2x + 5 = 15"

inputs = tokenizer(prompt, return_tensors="pt")


router_logits = capture_router_logits(model, inputs)


experts = selected_experts(
    router_logits,
    k=model.config.num_experts_per_tok
)


load = expert_load(experts)

print("Expert load:")
print(load)