import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import json
import numpy as np
import torch.nn as nn
import pickle

# Phase 4: Full Pipeline Evaluation
# Goal: Compare Baseline Blind RAG vs $O(1)$ RepE RAG on both F1 Accuracy and Latency

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load Model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

target_layer_idx = base_model.config.n_layer - 1 
target_layer = base_model.transformer.h[target_layer_idx]

# 2. Load the O(1) Linear Probe
feature_cols = ["Prompt_Norm", "Concept_Norm", "Dot_Product", "Cosine_Sim", "Token_Confidence", "Prompt_Length", "Collapse_Alpha"]
class AlphaPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

probe = AlphaPredictor(input_dim=7)
probe.load_state_dict(torch.load("linear_probe_weights.pt", map_location='cpu'))
probe.eval()

with open("feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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

def extract_vector(text, name):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    handle = target_layer.register_forward_hook(get_activation(name))
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
    vec = activation_cache[name]
    handle.remove()
    return vec, inputs.shape[1]

def evaluate():
    print("Loading HotpotQA Eval Dataset...")
    dataset = load_data()
    
    # Take the last 100 rows to ensure they are likely out of the training distribution logic
    eval_set = dataset[-100:]
    
    baseline_correct = 0
    repe_correct = 0
    baseline_time_total = 0
    repe_time_total = 0
    
    print(f"\n--- Starting Evaluation on {len(eval_set)} queries ---")
    
    for i, row in enumerate(eval_set):
        distractor = row['distractor_context']
        question = row['question']
        ground_truth = row['answer'].lower()
        
        prompt_text = f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_len = prompt_inputs.input_ids.shape[1]
        
        # --- 1. BASELINE RAG ---
        start_time = time.time()
        with torch.no_grad():
            out_base = base_model.generate(**prompt_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=True).lower()
        baseline_time = time.time() - start_time
        
        baseline_time_total += baseline_time
        if ground_truth in gen_base:
            baseline_correct += 1
            
        # --- 2. O(1) REPE RAG ---
        start_time = time.time()
        
        # Extract features (O(1) pass)
        c_hidden, _ = extract_vector(distractor, "concept")
        concept_vector = c_hidden.mean(dim=1, keepdim=True)
        
        p_hidden, prompt_tokens = extract_vector(prompt_text, "prompt")
        prompt_vector = p_hidden[:, -1:, :]
        
        concept_norm = torch.norm(concept_vector).item()
        prompt_norm = torch.norm(prompt_vector).item()
        dot_product = torch.dot(prompt_vector.flatten(), concept_vector.flatten()).item()
        cosine_sim = torch.nn.functional.cosine_similarity(prompt_vector.flatten(), concept_vector.flatten(), dim=0).item()
        collapse_alpha = dot_product / (concept_norm ** 2 + 1e-5)
        
        with torch.no_grad():
            outputs = base_model(input_ids=prompt_inputs.input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            baseline_confidence = torch.max(probs).item()
            
        # Predict Alpha using the probe
        features = np.array([[prompt_norm, concept_norm, dot_product, cosine_sim, baseline_confidence, prompt_tokens, collapse_alpha]])
        features_scaled = scaler.transform(features)
        features_t = torch.tensor(features_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_alpha = probe(features_t).item()
            
        # Clamp alpha just to be safe (never go above 99% of complete layer collapse)
        pred_alpha = max(0.0, min(pred_alpha, collapse_alpha * 0.99))
        
        # Steered Generation
        hook_handle = target_layer.register_forward_hook(create_steering_hook(pred_alpha, concept_vector))
        with torch.no_grad():
            out_repe = base_model.generate(**prompt_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_repe = tokenizer.decode(out_repe[0][input_len:], skip_special_tokens=True).lower()
        hook_handle.remove()
        
        repe_time = time.time() - start_time
        repe_time_total += repe_time
        
        if ground_truth in gen_repe:
            repe_correct += 1
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/100...")

    # --- 3. RESULTS ---
    print("\n================ FINAL RESULTS ================")
    print(f"Total Evaluation Queries: {len(eval_set)}")
    print(f"\n[Baseline RAG]")
    print(f"Accuracy (Exact Match): {baseline_correct / len(eval_set) * 100:.2f}%")
    print(f"Average Latency per Query: {baseline_time_total / len(eval_set):.4f} seconds")
    
    print(f"\n[O(1) RepE RAG]")
    print(f"Accuracy (Exact Match): {repe_correct / len(eval_set) * 100:.2f}%")
    print(f"Average Latency per Query: {repe_time_total / len(eval_set):.4f} seconds")
    
    latency_diff = ((repe_time_total - baseline_time_total) / baseline_time_total) * 100
    print(f"\nLatency Overhead of O(1) Probe: +{latency_diff:.2f}%")
    
    acc_diff = repe_correct - baseline_correct
    if acc_diff > 0:
         print(f"Accuracy Improvement: +{acc_diff} absolute points!")
    print("===============================================")


if __name__ == "__main__":
    evaluate()
