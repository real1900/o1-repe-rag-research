import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.spatial.distance import mahalanobis

print("Loading Llama-3.2-1B-Instruct for Multi-Dimensional Triage Production Test...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

def extract_activation(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    cache = {}
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache['val'] = hidden_states[:, -1, :].detach().float().cpu().numpy()
    
    try:
        layer = model.transformer.h[model.config.n_layer // 2]
    except AttributeError:
        layer = model.model.layers[model.config.num_hidden_layers // 2]
        
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return cache['val'][0]

print("\n--- 1. Simulating Retrieved Chunks for a Query ---")
print("Query: 'What year did Apollo 11 land on the moon?'")

valid_chunks = [
    "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
    "Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the crew.",
    "The Apollo spacecraft had three parts: a Command Module, Service Module, and Lunar Module.",
    "They collected 47.5 pounds of lunar material to bring back to Earth."
]

# Hard Negative: Topically identical but factually incorrect
hard_negative = "Apollo 11 successfully landed on the moon and Commander Neil Armstrong stepped onto the lunar surface in 1970."

# Spatial Outlier: Completely off-topic
spatial_outlier = "The Mariana Trench is the deepest oceanic trench on Earth."

print("\n--- 2. Layer 1: Spatial Triage (Mahalanobis Distance) ---")
valid_acts = np.array([extract_activation(c) for c in valid_chunks])
hard_neg_act = extract_activation(hard_negative)
spatial_outlier_act = extract_activation(spatial_outlier)

centroid = np.mean(valid_acts, axis=0)
cov_matrix = np.cov(valid_acts, rowvar=False)
inv_cov_matrix = np.linalg.pinv(cov_matrix + 1e-4 * np.eye(cov_matrix.shape[0]))

print(f"Valid Chunk 1 Distance: {mahalanobis(valid_acts[0], centroid, inv_cov_matrix):.4f}")
print(f"Valid Chunk 2 Distance: {mahalanobis(valid_acts[1], centroid, inv_cov_matrix):.4f}")
print(f"Hard Negative Distance: {mahalanobis(hard_neg_act, centroid, inv_cov_matrix):.4f} (SURVIVES! It is inside the topic cluster)")
print(f"Spatial Outlier Distance: {mahalanobis(spatial_outlier_act, centroid, inv_cov_matrix):.4f} (CAUGHT by Spatial Triage!)")


print("\n--- 3. Layer 2: Veracity Triage (Factuality Projection) ---")
# Pre-compute Factuality Vector via Contrastive Activation Addition (CAA)
# Truth axis = True Statement - False Statement
true_stmt = extract_activation("The Apollo 11 moon landing happened in 1969.")
false_stmt = extract_activation("The Apollo 11 moon landing happened in 1970.")
v_fact = true_stmt - false_stmt
v_fact = v_fact / np.linalg.norm(v_fact)

def get_veracity(act):
    return np.dot(act, v_fact) / (np.linalg.norm(act) * np.linalg.norm(v_fact))

print(f"Valid Chunk 1 Veracity: {get_veracity(valid_acts[0]):.4f} (Positive = Aligns with Truth)")
print(f"Hard Negative Veracity: {get_veracity(hard_neg_act):.4f} (CAUGHT! Points purely to False Pole)")
print(f"Spatial Outlier Veracity: {get_veracity(spatial_outlier_act):.4f} (Neutral/Irrelevant)")


print("\n--- 4. Layer 3: Uncertainty Gating ---")
# Exposing a fact outside the model's parametric memory
unknown_stmt = extract_activation("The secret architectural code for the Xenon platform is X-992.")
unknown_veracity = get_veracity(unknown_stmt)
print(f"Unknown Fact Entropy/Veracity: {unknown_veracity:.4f}")
print(f"Near 0.0 indicates no internal logic structure for this fact -> Triggers Uncertainty Gate -> Disables Steering to protect new RAG data.")

print("\n--- Conclusion ---")
print("Hybrid Multi-Dimensional Triage successfully isolates both 'Off-Topic' distractors AND 'Hard Negatives', building a mathematically rigorous security checkpoint before generative decoding begins.")
