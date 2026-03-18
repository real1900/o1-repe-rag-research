import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

def format_prompt(context, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print("Evaluating TRUE context baseline...")
correct = 0
for i in range(50):
    row = dataset[-50:][i]
    # Use TRUE context instead of distractor
    prompt = format_prompt(row['true_context'], row['question'])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()
    
    if row['answer'].lower() in gen:
        correct += 1
        
print(f"True Baseline Accuracy: {(correct/50)*100}%")

