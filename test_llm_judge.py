import torch
import json
import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("HF_TOKEN not found!")
    exit(1)

# Initialize the Judge
judge_client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)

def ask_judge(question, ground_truth, model_generation):
    prompt = f"""You are an objective evaluator.
Question: {question}
Correct Answer (Ground Truth): {ground_truth}
Model's Answer: {model_generation}

Does the Model's Answer contain the Correct Answer or mean the exact same thing in this context? 
Respond with exactly one word: "YES" or "NO". Do not provide any other explanation."""
    
    try:
        response = judge_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().upper()
        if "YES" in answer:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Judge API Error: {e}")
        return 0.0

def format_true_prompt(true_ctx, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n{true_ctx}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

eval_set = dataset[-20:]
true_correct = 0.0

print("\nEvaluating True Context Only baseline using LLM-as-a-Judge...")
for i, row in enumerate(eval_set):
    true_prompt = format_true_prompt(row['true_context'], row['question'])
    inputs = tokenizer(true_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

    score = ask_judge(row['question'], row['answer'], gen)
    true_correct += score
    
    time.sleep(1.0) # Rate limit respect

    if (i+1) % 5 == 0:
        print(f"Processed {i+1}/20... (True Context Acc: {(true_correct/(i+1))*100:.1f}%)")

print("\n================ LLM JUDGE UPPER BOUND BASELINE ================")
print(f"Total Evaluation Queries: {len(eval_set)}")
print(f"Accuracy (LLM Judge): {(true_correct/len(eval_set))*100:.2f}%")

