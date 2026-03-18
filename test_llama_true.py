import torch
import json
import string
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def contains_answer(prediction, ground_truth):
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if gt_norm in pred_norm:
        return 1.0
    return 0.0

def format_true_prompt(true_ctx, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n{true_ctx}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Use last 50
eval_set = dataset[-50:]

true_correct = 0.0

print("\nEvaluating True Context Only baseline on Llama-3.2-1B-Instruct...")
for i, row in enumerate(eval_set):
    true_prompt = format_true_prompt(row['true_context'], row['question'])
    inputs = tokenizer(true_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()

    true_correct += contains_answer(gen, row['answer'])

    if (i+1) % 10 == 0:
        print(f"Processed {i+1}/50... (True Context Acc: {(true_correct/(i+1))*100:.1f}%)")

print("\n================ UPPER BOUND BASELINE ================")
print(f"Total Evaluation Queries: {len(eval_set)}")
print(f"Accuracy (Normalized Substring Match): {(true_correct/len(eval_set))*100:.2f}%")
