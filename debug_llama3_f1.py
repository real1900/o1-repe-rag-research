import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
import re

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def format_mixed_prompt(true_ctx, distractor, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n\nDocument A:\n{distractor}\n\nDocument B:\n{true_ctx}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

for i in range(3):
    row = dataset[-20:][i]
    question = row['question']
    answer = row['answer']
    mixed_prompt = format_mixed_prompt(row['true_context'], row['distractor_context'], question)
    inputs = tokenizer(mixed_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        # Using 50 tokens to see if it's getting cut off
        out_base = base_model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=True).strip()
    
    print(f"\n[Query {i+1}] {question}")
    print(f"Ground Truth: {answer}")
    print(f"Normalized GT: {normalize_answer(answer)}")
    print(f"Baseline Gen: {gen_base}")
    print(f"Normalized Gen: {normalize_answer(gen_base)}")
    print(f"F1 Score: {f1_score(gen_base, answer):.2f}")
