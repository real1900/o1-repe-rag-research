import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

activation_cache = {}
def get_activation(name):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        activation_cache[name] = hidden_states.detach()
    return hook

target_layer_idx = base_model.config.n_layer - 1 
target_layer = base_model.transformer.h[target_layer_idx]

# 1. Extract Concept Vector (Distractor)
dummy_text = "This paragraph contains completely irrelevant, distracting information about a side topic." * 3
dummy_input = tokenizer(dummy_text, return_tensors="pt").input_ids.to(device)

hook_handle = target_layer.register_forward_hook(get_activation(f"layer_{target_layer_idx}_concept"))
with torch.no_grad():
    _ = base_model(input_ids=dummy_input)
extracted_vector = activation_cache[f"layer_{target_layer_idx}_concept"]
hook_handle.remove()

concept_vector = extracted_vector.mean(dim=1, keepdim=True) # [1, 1, 768]

# 2. Extract Prompt Vector
prompt = "The capital of France is"
prompt_input = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

hook_handle = target_layer.register_forward_hook(get_activation(f"layer_{target_layer_idx}_prompt"))
with torch.no_grad():
    _ = base_model(input_ids=prompt_input)
prompt_hidden_states = activation_cache[f"layer_{target_layer_idx}_prompt"]
hook_handle.remove()

prompt_vector = prompt_hidden_states[:, -1:, :] # Take the last token's representation [1, 1, 768]

# 3. Geometric Analysis
concept_norm = torch.norm(concept_vector).item()
prompt_norm = torch.norm(prompt_vector).item()

dot_product = torch.dot(prompt_vector.flatten(), concept_vector.flatten()).item()
cosine_sim = torch.nn.functional.cosine_similarity(prompt_vector.flatten(), concept_vector.flatten(), dim=0).item()

# Projection of concept onto prompt
# scalar projection = dot(a, b) / norm(b)
scalar_proj = dot_product / prompt_norm
ratio_norms = prompt_norm / concept_norm
ratio_norms_inv = concept_norm / prompt_norm

print(f"--- Geometric Analysis of the Prompt vs Concept ---")
print(f"Prompt Tokens: {prompt_input.shape[1]}")
print(f"Distractor Tokens: {dummy_input.shape[1]}")
print(f"Prompt Vector L2 Norm: {prompt_norm:.5f}")
print(f"Concept Vector L2 Norm: {concept_norm:.5f}")
print(f"Dot Product: {dot_product:.5f}")
print(f"Cosine Similarity: {cosine_sim:.5f}")
print(f"---------------------------------------------------")
print(f"Ratio of Norms (Prompt / Concept): {ratio_norms:.5f}")
print(f"Ratio of Norms (Concept / Prompt): {ratio_norms_inv:.5f}")
print(f"Scalar Projection (Dot / PromptNorm): {scalar_proj:.5f}")
print(f"Geometric Formula 1: Cosine Similarity * Ratio (Concept / Prompt) = {cosine_sim * ratio_norms_inv:.5f}")
print(f"Geometric Formula 2: Norm Ratio / Prompt Tokens = {ratio_norms_inv / prompt_input.shape[1]:.5f}")
print(f"Geometric Formula 3: 1 - Cosine Similarity = {1 - cosine_sim:.5f}")

target_alpha = 0.29469
print(f"\\nTarget Alpha (Experimentally Found): {target_alpha}")
