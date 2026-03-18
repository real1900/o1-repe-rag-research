import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder
import numpy as np
import pickle
import time
import json

# =================================================================================
# Hybrid Production Architecture for O(1) RepE RAG
# =================================================================================
# This script implements the three architectures proposed in the "Future Work"
# section to deploy mechanistic steering into a real-world enterprise environment:
# 1. Two-Stage Filtering (Cross-Encoder BGE-Reranker)
# 2. Targeted Layer Steering (Injecting into Factual layers only)
# 3. KL-Divergence Braking (Preventing Semantic Collapse dynamically)
# =================================================================================

print("Loading Models & Environment...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 1. Load the Generative LLM
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

# 2. Load the Cross-Encoder (Stage 1 Triaging)
# In production, we'd use BAAI/bge-reranker-large, but we use MiniLM here for latency
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

# 3. Load the O(1) Linear Probe
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

# Define Factual "Knowledge" Layers for Targeted Steering
# Middle layers handle facts; deep layers handle logic. We don't touch logic.
TARGET_LAYERS = [4, 5, 6, 7, 8]  
active_hooks = []

def get_activation_dict(name, cache_dict):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache_dict[name] = hidden_states.detach()
    return hook

def extract_vector(text, name, layer_idx):
    cache = {}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    layer = base_model.transformer.h[layer_idx]
    handle = layer.register_forward_hook(get_activation_dict(name, cache))
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
    vec = cache[name]
    handle.remove()
    return vec, inputs.shape[1]

def create_steering_hook(alpha_val, c_vec):
    def steering_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Expand c_vec to match hidden_states length if needed, or broadcast
        # c_vec is shape [1, 1, 768] usually (mean over seq len)
        steered_states = hidden_states - (alpha_val * c_vec)
        if isinstance(output, tuple):
            return (steered_states,) + output[1:]
        else:
            return steered_states
    return steering_hook

def register_targeted_hooks(alpha_val, c_vecs_by_layer):
    global active_hooks
    remove_hooks() # clear existing
    for layer_idx in TARGET_LAYERS:
        layer = base_model.transformer.h[layer_idx]
        hook = layer.register_forward_hook(create_steering_hook(alpha_val, c_vecs_by_layer[layer_idx]))
        active_hooks.append(hook)

def remove_hooks():
    global active_hooks
    for h in active_hooks:
        h.remove()
    active_hooks = []

def kl_divergence(p_logits, q_logits):
    p_probs = torch.nn.functional.softmax(p_logits, dim=-1)
    q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    kl = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    return kl.item()

def generate_with_kl_braking(prompt_inputs, alpha_val, c_vecs_by_layer, max_new_tokens=10, kl_threshold=2.5):
    """
    Custom decoding loop that dynamically monitors semantic collapse via KL-Divergence.
    If the steering causes massive divergence from original logic, the 'brake' applies
    by instantly decaying alpha by 50% for that token step.
    """
    current_ids = prompt_inputs.input_ids
    past_key_values = None
    
    current_alpha = alpha_val
    generated_tokens = []
    
    for _ in range(max_new_tokens):
        # 1. Forward Pass (UNSTEERED - baseline logic check)
        remove_hooks()
        with torch.no_grad():
            out_unsteered = base_model(input_ids=current_ids)
            unsteered_logits = out_unsteered.logits[0, -1, :]
            
        # 2. Forward Pass (STEERED - mechanistic extraction)
        register_targeted_hooks(current_alpha, c_vecs_by_layer)
        with torch.no_grad():
            out_steered = base_model(input_ids=current_ids)
            steered_logits = out_steered.logits[0, -1, :]
            
        remove_hooks()
        
        # 3. KL-Divergence Braking System
        divergence = kl_divergence(unsteered_logits, steered_logits)
        
        if divergence > kl_threshold:
            # Semantic Collapse Detected! Hitting the brakes.
            current_alpha *= 0.5
            # Re-run steered pass with decayed alpha to get safer logits
            register_targeted_hooks(current_alpha, c_vecs_by_layer)
            with torch.no_grad():
                out_steered = base_model(input_ids=current_ids)
                steered_logits = out_steered.logits[0, -1, :]
            remove_hooks()
            
        # Select best token from steered distribution
        next_token_id = torch.argmax(steered_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token_id.item())
        current_ids = torch.cat([current_ids, next_token_id], dim=1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
    return current_ids

def run_hybrid_pipeline():
    with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    eval_set = dataset[-15:] # Small set for demonstration
    
    print("\n==================================================")
    print("HYBRID PRODUCTION ARCHITECTURE INITIATED")
    print("==================================================")
    
    for i, row in enumerate(eval_set):
        distractor = row['distractor_context']
        question = row['question']
        
        print(f"\n[Query {i+1}] {question}")
        
        # ========================================================
        # Guardrail 1: Cross-Encoder Triaging (Stage 1)
        # ========================================================
        # Score the semantic relevance of the retrieved chunk
        score = cross_encoder.predict([(question, distractor)])[0]
        
        if score > 0.0:
            print(" -> [Stage 1] Reranker cleared chunk. Safe to use Standard RAG.")
            # Standard generation without steering
            prompt_text = f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = base_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f" -> Generation: {gen.strip()}")
            continue
            
        print(f" -> [Stage 1] Reranker flagged Distractor (Score: {score:.2f}). Engaging RepE Extractors...")
        
        # ========================================================
        # Guardrail 2: Targeted Layer Extraction
        # ========================================================
        c_vecs_by_layer = {}
        target_layer_idx = TARGET_LAYERS[-1] # Primary for probe
        
        # Extract features for probe using highest targeted middle layer
        c_hidden, _ = extract_vector(distractor, "concept", target_layer_idx)
        concept_vector_main = c_hidden.mean(dim=1, keepdim=True)
        
        # Extract individual layer vectors for surgical targeted steering
        for layer_idx in TARGET_LAYERS:
            c_h, _ = extract_vector(distractor, f"c_{layer_idx}", layer_idx)
            c_vecs_by_layer[layer_idx] = c_h.mean(dim=1, keepdim=True)
            
        prompt_text = f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"
        p_hidden, prompt_tokens = extract_vector(prompt_text, "prompt", target_layer_idx)
        prompt_vector_main = p_hidden[:, -1:, :]
        
        concept_norm = torch.norm(concept_vector_main).item()
        prompt_norm = torch.norm(prompt_vector_main).item()
        dot_product = torch.dot(prompt_vector_main.flatten(), concept_vector_main.flatten()).item()
        cosine_sim = torch.nn.functional.cosine_similarity(prompt_vector_main.flatten(), concept_vector_main.flatten(), dim=0).item()
        collapse_alpha = dot_product / (concept_norm ** 2 + 1e-5)
        
        # Baseline confidence
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = base_model(input_ids=inputs.input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            baseline_confidence = torch.max(probs).item()
            
        # O(1) Probe Alpha
        features = np.array([[prompt_norm, concept_norm, dot_product, cosine_sim, baseline_confidence, prompt_tokens, collapse_alpha]])
        features_scaled = scaler.transform(features)
        with torch.no_grad():
            pred_alpha = probe(torch.tensor(features_scaled, dtype=torch.float32)).item()
            
        # Clamp mathematically
        pred_alpha = max(0.0, min(pred_alpha, collapse_alpha * 0.99))
        print(f" -> [Probe] O(1) Target Steer Coefficient: a={pred_alpha:.4f}")
        
        # ========================================================
        # Guardrail 3: KL-Divergence Braking Generation
        # ========================================================
        print(" -> [Generation] Applying targeted layer steering with KL-Braking...")
        steered_ids = generate_with_kl_braking(inputs, pred_alpha, c_vecs_by_layer, max_new_tokens=10, kl_threshold=2.5)
        
        gen_steered = tokenizer.decode(steered_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f" -> Generation: {gen_steered.strip()}")
        

if __name__ == "__main__":
    run_hybrid_pipeline()
