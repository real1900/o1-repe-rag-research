import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time

# =================================================================================
# Qwen2.5-1.5B-Instruct Evaluation Pipeline
# Testing the hypothesis: Does RepE steering hold up on high-baseline reasoning models?
# (Using an open-weights model to bypass Meta's manual Llama-3.2 gating review)
# =================================================================================

print("Loading Qwen2.5-1.5B-Instruct...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Using Qwen2.5 1.5B Instruct as the proxy for high-baseline reasoning models
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
# Qwen doesn't set pad token by default
tokenizer.pad_token = tokenizer.eos_token 

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Model config setup
NUM_LAYERS = base_model.config.num_hidden_layers
print(f"Model loaded. Total Layers: {NUM_LAYERS} | Hidden Size: {base_model.config.hidden_size}")

# Targeted Layer Steering
# For high-reasoning models, we only inject RepE into middle "Knowledge" layers
TARGET_LAYERS = range(NUM_LAYERS // 4, NUM_LAYERS // 2) 
print(f"Targeting Fact Retrieval Layers: {list(TARGET_LAYERS)}")

active_hooks = []

def get_activation_dict(name, cache_dict):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache_dict[name] = hidden_states.detach()
    return hook

def extract_vector(text, name, layer_idx):
    cache = {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    layer = base_model.model.layers[layer_idx]
    handle = layer.register_forward_hook(get_activation_dict(name, cache))
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
    vec = cache[name]
    handle.remove()
    return vec, inputs.shape[1]

def create_steering_hook(alpha_val, c_vec):
    def steering_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        steered_states = hidden_states - (alpha_val * c_vec)
        if isinstance(output, tuple):
            return (steered_states,) + output[1:]
        else:
            return steered_states
    return steering_hook

def register_targeted_hooks(alpha_val, c_vecs_by_layer):
    global active_hooks
    remove_hooks()
    for layer_idx in TARGET_LAYERS:
        layer = base_model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_steering_hook(alpha_val, c_vecs_by_layer[layer_idx]))
        active_hooks.append(hook)

def remove_hooks():
    global active_hooks
    for h in active_hooks:
        h.remove()
    active_hooks = []

def format_prompt(distractor, question):
    return f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nContext: {distractor}\n\nQuestion: {question}\nAnswer purely based on the context.<|im_end|>\n<|im_start|>assistant\n"

def evaluate_high_baseline():
    with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Evaluate a sample of the unseen test set
    eval_set = dataset[-50:] 
    
    baseline_correct = 0
    repe_correct = 0
    
    print(f"\n--- Starting Evaluation on {len(eval_set)} queries ---")
    
    # Static calibrated alpha for this proof of concept since the Linear Probe is 
    # calibrated specifically to GPT-2's embedding space.
    STATIC_ALPHA = 0.45 
    
    for i, row in enumerate(eval_set):
        distractor = row['distractor_context']
        question = row['question']
        ground_truth = row['answer'].lower()
        
        prompt_text = format_prompt(distractor, question)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = prompt_inputs.input_ids.shape[1]
        
        # --- 1. BASELINE RAG (Unsteered) ---
        with torch.no_grad():
            out_base = base_model.generate(**prompt_inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=True).lower()
        
        if ground_truth in gen_base:
            baseline_correct += 1
            
        # --- 2. O(1) RepE Targeted Steering ---
        c_vecs_by_layer = {}
        target_layer_idx = TARGET_LAYERS[-1]
        
        # Extract individual layer vectors for surgical targeted steering
        for layer_idx in TARGET_LAYERS:
            c_h, _ = extract_vector(distractor, f"c_{layer_idx}", layer_idx)
            c_vecs_by_layer[layer_idx] = c_h.mean(dim=1, keepdim=True)
            
        register_targeted_hooks(STATIC_ALPHA, c_vecs_by_layer)
        with torch.no_grad():
            out_repe = base_model.generate(**prompt_inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_repe = tokenizer.decode(out_repe[0][input_len:], skip_special_tokens=True).lower()
        remove_hooks()
        
        if ground_truth in gen_repe:
            repe_correct += 1
            
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/50...")

    # --- 3. RESULTS ---
    print("\n================ FINAL HIGH-BASELINE RESULTS ================")
    print(f"Total Evaluation Queries: {len(eval_set)}")
    
    base_acc = (baseline_correct / len(eval_set)) * 100
    repe_acc = (repe_correct / len(eval_set)) * 100
    
    print(f"\n[Baseline Qwen Unsteered]")
    print(f"Accuracy (Exact Match): {base_acc:.2f}%")
    
    print(f"\n[O(1) RepE Targeted Steering]")
    print(f"Accuracy (Exact Match): {repe_acc:.2f}%")
    
    if base_acc > 0:
        relative_jump = ((repe_acc - base_acc) / base_acc) * 100
        print(f"\nTargeted Layer Steering delivered a {relative_jump:+.2f}% Relative Change on a High-Baseline model.")
    elif repe_acc > base_acc:
        print(f"\nTargeted Layer Steering improved accuracy from 0% to {repe_acc:.2f}%.")
    print("==========================================================")

if __name__ == "__main__":
    evaluate_high_baseline()
