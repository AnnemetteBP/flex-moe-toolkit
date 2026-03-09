from transformers import AutoModelForCausalLM, AutoTokenizer

from flex_moe_toolkit.capture import capture_router_logits
from flex_moe_toolkit.logger import log_routing
from flex_moe_toolkit.analysis import compute_expert_usage, layer_expert_matrix
from flex_moe_toolkit.plots import plot_expert_heatmap



model_name = "allenai/Flex-math-2x7B-1T"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu"
)


prompt = "Solve 2x + 5 = 15"

inputs = tokenizer(prompt, return_tensors="pt")


router_logits = capture_router_logits(model, inputs)


log_routing(router_logits, "routing_logs.jsonl")


usage = compute_expert_usage("routing_logs.jsonl")

print("Expert usage:")
print(usage)


matrix = layer_expert_matrix("routing_logs.jsonl")


plot_expert_heatmap(matrix)