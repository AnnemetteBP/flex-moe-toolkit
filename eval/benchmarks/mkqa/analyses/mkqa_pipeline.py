import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================
MODEL_NAME = "YOUR_MODEL_PATH_OR_HF_ID"
CACHE_PATH = "/work/training/flex-moe-toolkit/data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 50


# =========================
# LOAD MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()


# =========================
# BUILD DATASET (DA + EN)
# =========================
def build_dataset():
    ds = load_dataset(
        "mkqa",
        split="train",
        streaming=True
    )

    pairs = []

    for ex in ds:
        q_da = ex["queries"].get("da")
        q_en = ex["queries"].get("en")

        a_da = ex["answers"].get("da")
        a_en = ex["answers"].get("en")

        if q_da and q_en and a_da and a_en:
            pairs.append({
                "question_da": q_da,
                "question_en": q_en,
                "answer_da": a_da[0]["text"],
                "answer_en": a_en[0]["text"],
            })

        if len(pairs) >= NUM_SAMPLES:
            break

    return pairs


# =========================
# CHAT TEMPLATE
# =========================
def build_prompt(question):
    messages = [
        {"role": "user", "content": question}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


# =========================
# MODEL RUN
# =========================
def run_model(question):
    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50
        )

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return decoded


# =========================
# (OPTIONAL) ROUTING HOOK
# =========================
routing_logs = []

def hook_router(module, input, output):
    routing_logs.append(output)


def register_hooks():
    # TODO: replace med korrekt FlexOlmo layer
    for name, module in model.named_modules():
        if "router" in name.lower():
            module.register_forward_hook(hook_router)


# =========================
# MAIN EVAL LOOP
# =========================
def run_eval():
    data = build_dataset()

    results = []

    for i, ex in enumerate(data):
        print(f"Running sample {i}")

        en_q = ex["question_en"]
        da_q = ex["question_da"]

        out_en = run_model(en_q)
        out_da = run_model(da_q)

        results.append({
            "question_en": en_q,
            "question_da": da_q,
            "output_en": out_en,
            "output_da": out_da,
            "answer_en": ex["answer_en"],
            "answer_da": ex["answer_da"]
        })

    return results


# =========================
# SAVE RESULTS
# =========================
def save_results(results):
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    # register_hooks()  

    results = run_eval()
    save_results(results)

    print("Done.")