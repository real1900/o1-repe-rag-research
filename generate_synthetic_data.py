import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import os
import csv

# Phase 2: Synthetic Data Generation (The Auto-Tuning Loop)
# Goal: For 5000 rows, extract prompt geometry, calculate max alpha, binary search for sweet spot, log it.

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Base Models (Using Standard GPT2 for speed during proof-of-concept synthetic generation)
# In final paper execution, swap back to Llama-3-8B-Instruct
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

def load_data(filepath="hotpot_filtered_5000.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# Hook setup for RepE
activation_cache = {}
def get_activation(name):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activation_cache[name] = hidden_states.detach()
    return hook

def create_steering_hook(alpha_val, c_vec):
    def steering_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        steered_states = hidden_states - (alpha_val * c_vec)
        if isinstance(output, tuple):
            return (steered_states,) + output[1:]
        else:
            return steered_states
    return steering_hook

target_layer_idx = base_model.config.n_layer - 1 
target_layer = base_model.transformer.h[target_layer_idx]

def extract_vector(text, name):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    handle = target_layer.register_forward_hook(get_activation(name))
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
    vec = activation_cache[name]
    handle.remove()
    return vec, inputs.shape[1] # Return vector and token length

def process_row(row):
    try:
        # 1. Geometry Setup
        distractor = row['distractor_context']
        question = row['question']
        ground_truth = row['answer'].lower()
        
        prompt_text = f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"
        
        c_hidden, _ = extract_vector(distractor, "concept")
        concept_vector = c_hidden.mean(dim=1, keepdim=True) # Distractor Concept Vector
        
        p_hidden, prompt_tokens = extract_vector(prompt_text, "prompt")
        prompt_vector = p_hidden[:, -1:, :] # Last token of the prompt
        
        concept_norm = torch.norm(concept_vector).item()
        prompt_norm = torch.norm(prompt_vector).item()
        
        # 2. Extract Baseline Confidence (Logits of the first token prediction before steering)
        with torch.no_grad():
            outputs = base_model(input_ids=tokenizer(prompt_text, return_tensors="pt").input_ids.to(device))
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            baseline_confidence = torch.max(probs).item()
            
        # 3. Collapse Alpha (Ceiling of Destruction)
        dot_product = torch.dot(prompt_vector.flatten(), concept_vector.flatten()).item()
        cosine_sim = torch.nn.functional.cosine_similarity(prompt_vector.flatten(), concept_vector.flatten(), dim=0).item()
        collapse_alpha = dot_product / (concept_norm ** 2)
        
        # 4. Binary Search for Optimal Alpha
        low = 0.0
        high = collapse_alpha
        optimal_alpha = 0.0
        max_search_depth = 6
        found_target = False
        
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = prompt_inputs.input_ids.shape[1]
        
        for _ in range(max_search_depth):
            mid = (low + high) / 2.0
            
            hook_handle = target_layer.register_forward_hook(create_steering_hook(mid, concept_vector))
            with torch.no_grad():
                out = base_model.generate(**prompt_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()
            hook_handle.remove()
            
            if ground_truth in gen_text:
                optimal_alpha = mid
                high = mid # Try to find a lower, safer alpha that still succeeds
                found_target = True
            else:
                low = mid # Need more steering strength (or we pushed too far into collapse, but binary search assumes monotonicity here)
                
        # If we never found it, fallback to default 0.0 (Unsteerable)
        if not found_target:
            optimal_alpha = 0.0

        return {
            "Id": row['id'],
            "Prompt_Norm": prompt_norm,
            "Concept_Norm": concept_norm,
            "Dot_Product": dot_product,
            "Cosine_Sim": cosine_sim,
            "Token_Confidence": baseline_confidence,
            "Prompt_Length": prompt_tokens,
            "Collapse_Alpha": collapse_alpha,
            "Optimal_Alpha": optimal_alpha,
            "Success": found_target
        }
    except Exception as e:
        print(f"Error processing {row['id']}: {e}")
        return None

def main():
    dataset = load_data()
    total = len(dataset)
    output_file = "synthetic_alpha_tuning.csv"
    
    # Check if we are resuming
    start_idx = 0
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            start_idx = sum(1 for line in f) - 1 # discount header
        if start_idx < 0: start_idx = 0
            
    print(f"Starting auto-tuner loop for {total} rows. Resuming from {start_idx}...")
    
    file_mode = 'a' if start_idx > 0 else 'w'
    
    with open(output_file, file_mode, newline='') as csvfile:
        fieldnames = ["Id", "Prompt_Norm", "Concept_Norm", "Dot_Product", "Cosine_Sim", "Token_Confidence", "Prompt_Length", "Collapse_Alpha", "Optimal_Alpha", "Success"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if start_idx == 0:
            writer.writeheader()
            
        start_time = time.time()
        successful_finds = 0
            
        for i in range(start_idx, total):
            row = dataset[i]
            res = process_row(row)
            
            if res:
                writer.writerow(res)
                if res["Success"]:
                    successful_finds += 1
                
                if (i + 1) % 10 == 0:
                    csvfile.flush() # Save incrementally
                    elapsed = time.time() - start_time
                    rate = (i - start_idx + 1) / elapsed
                    print(f"Processed {i+1}/{total} [{rate:.2f} it/s] | Alpha Discoveries: {successful_finds}")

if __name__ == "__main__":
    main()
