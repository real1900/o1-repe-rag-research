import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
import re
import random
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def contains_answer(prediction, ground_truths):
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if gt_norm in pred_norm:
            return 1.0
    return 0.0

def format_squad_prompt(context, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Load SQuAD validation dataset
print("Loading SQuAD v1.1 Dataset...")
dataset = load_dataset("squad", split="validation")

# Randomly sample 100 queries for evaluation
random.seed(42)
sampled_indices = random.sample(range(len(dataset)), 100)
eval_set = dataset.select(sampled_indices)

correct_answers = 0.0
total_queries = len(eval_set)

print(f"\nEvaluating Baseline Accuracy on SQuAD ({total_queries} queries)...")
for i, row in enumerate(eval_set):
    context = row['context']
    question = row['question']
    answers = row['answers']['text']  # List of acceptable ground truths

    prompt = format_squad_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()

    score = contains_answer(gen, answers)
    correct_answers += score

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/{total_queries}... (Accuracy: {(correct_answers/(i+1))*100:.1f}%)")

final_accuracy = (correct_answers / total_queries) * 100
print("\n================ SQUAD V1.1 BASELINE RESULTS ================")
print(f"Total Evaluation Queries: {total_queries}")
print(f"Dataset Complexity: Single-Hop Extraction")
print(f"Accuracy (Normalized Substring Match): {final_accuracy:.2f}%")
if final_accuracy >= 70.0:
    print("\nSUCCESS: Llama-3-1B mathematically hits the 70-80%+ capability on single-hop RAG!")
