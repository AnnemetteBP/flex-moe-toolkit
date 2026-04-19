#!/usr/bin/env python3
"""
Evaluation Pipeline for FlexMoRE Models

This script runs evaluation on specified models and datasets.
Supports MC9 (multiple choice) and Danish Wiki QA datasets.

Usage:
    python eval_pipeline.py --model <model_name> --dataset <dataset_name> --subsample <size>

Example:
    python eval_pipeline.py --model Flex-public-2x7B-1T --dataset mc9 --subsample 100
"""

import argparse
import json
import os
import sys
from typing import List, Dict, Any

# Assume transformers and datasets are available
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from datasets import load_dataset
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install transformers torch datasets")
    sys.exit(1)


def load_model(model_name: str):
    """Load model and tokenizer from local path."""
    model_path = os.path.join("..", "models", model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_dataset_mc9(subsample_size: int = None):
    """Load MC9 dataset. For now, create mock data if not available."""
    try:
        # Assume MC9 is available on Hugging Face or local
        dataset = load_dataset("mc9")  # Placeholder - replace with actual dataset name
    except:
        print("MC9 dataset not found, using mock data")
        # Mock MC9 data: multiple choice with 4 options
        mock_data = [
            {
                "question": "What is the capital of Denmark?",
                "choices": ["Copenhagen", "Stockholm", "Oslo", "Helsinki"],
                "answer": 0
            },
            {
                "question": "Which is the largest Scandinavian country?",
                "choices": ["Denmark", "Norway", "Sweden", "Finland"],
                "answer": 2
            },
            # Add more mock questions
        ] * 10  # Repeat for more data
        dataset = {"test": mock_data}

    if subsample_size:
        dataset["test"] = dataset["test"][:subsample_size]

    return dataset["test"]


def load_dataset_danish_wiki_qa(subsample_size: int = None):
    """Load Danish Wiki QA dataset. For now, create mock data if not available."""
    try:
        # Assume Danish Wiki QA is available
        dataset = load_dataset("danish_wiki_qa")  # Placeholder
    except:
        print("Danish Wiki QA dataset not found, using mock data")
        # Mock QA data
        mock_data = [
            {
                "context": "København er hovedstaden i Danmark. Det er den største by i landet.",
                "question": "Hvad er hovedstaden i Danmark?",
                "answer": "København"
            },
            {
                "context": "Danmark er et nordisk land i Europa. Det grænser op til Tyskland.",
                "question": "Hvilket kontinent ligger Danmark i?",
                "answer": "Europa"
            },
            # Add more mock questions
        ] * 10
        dataset = {"test": mock_data}

    if subsample_size:
        dataset["test"] = dataset["test"][:subsample_size]

    return dataset["test"]


def evaluate_mc9(model, tokenizer, dataset: List[Dict[str, Any]]) -> float:
    """Evaluate on MC9 multiple choice task."""
    correct = 0
    total = len(dataset)

    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        correct_answer = item["answer"]

        # Format as multiple choice
        prompt = f"{question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{i+1}. {choice}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the last token logits
        last_logits = logits[0, -1, :]
        choice_tokens = []
        for i in range(len(choices)):
            token = tokenizer.encode(f" {i+1}", add_special_tokens=False)[0]
            choice_tokens.append(token)

        # Find the choice with highest probability
        choice_probs = [last_logits[token].item() for token in choice_tokens]
        predicted = choice_probs.index(max(choice_probs))

        if predicted == correct_answer:
            correct += 1

    accuracy = correct / total
    return accuracy


def evaluate_danish_wiki_qa(model, tokenizer, dataset: List[Dict[str, Any]]) -> float:
    """Evaluate on Danish Wiki QA task. Simplified: check if answer is in generated text."""
    correct = 0
    total = len(dataset)

    for item in dataset:
        context = item["context"]
        question = item["question"]
        correct_answer = item["answer"]

        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Simple check: if correct answer is in generated text
        if correct_answer.lower() in generated.lower():
            correct += 1

    accuracy = correct / total
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlexMoRE models on datasets")
    parser.add_argument("--model", required=True, help="Model name (e.g., Flex-public-2x7B-1T)")
    parser.add_argument("--dataset", required=True, choices=["mc9", "danish_wiki_qa"], help="Dataset to evaluate on")
    parser.add_argument("--subsample", type=int, help="Subsample size for testing")

    args = parser.parse_args()

    # Load model
    try:
        model, tokenizer = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load dataset
    if args.dataset == "mc9":
        dataset = load_dataset_mc9(args.subsample)
        metric = evaluate_mc9(model, tokenizer, dataset)
        print(".4f")
    elif args.dataset == "danish_wiki_qa":
        dataset = load_dataset_danish_wiki_qa(args.subsample)
        metric = evaluate_danish_wiki_qa(model, tokenizer, dataset)
        print(".4f")

    print(f"Evaluation completed for {args.model} on {args.dataset}")


if __name__ == "__main__":
    main()