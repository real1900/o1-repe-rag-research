import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id} for Logit Lens Probing...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
NUM_LAYERS = base_model.config.num_hidden_layers

# Extract the final LayerNorm and LM Head for Logit Lens projection
ln_f = base_model.model.norm
lm_head = base_model.lm_head

# Jensen-Shannon Divergence calculation
def jensen_shannon_divergence(p_logits, q_logits):
    eps = 1e-9
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    m = 0.5 * (p + q)
    kl_p_m = F.kl_div((m + eps).log(), p, reduction='sum')
    kl_q_m = F.kl_div((m + eps).log(), q, reduction='sum')
    return 0.5 * kl_p_m + 0.5 * kl_q_m

def get_layer_hidden_states(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    cache = {}
    hooks = []
    
    def get_hook(layer_idx):
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Extract only the final sequence token's hidden state
            cache[layer_idx] = hidden_states[:, -1, :].detach()
        return hook

    # Attach hooks to all layers
    for i in range(NUM_LAYERS):
        h = base_model.model.layers[i].register_forward_hook(get_hook(i))
        hooks.append(h)
        
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
        
    for h in hooks:
        h.remove()
        
    return cache

print("\n--- Phase 1: Forward Pass Diagnostics ---")
clean_prompt = "Question: What year did the Apollo 11 moon landing take place?\nAnswer:"
distracted_prompt = "Document: The Mariana Trench is 36,000 feet deep.\nQuestion: What year did the Apollo 11 moon landing take place?\nAnswer:"

print("Extracting hidden states for Clean and Distracted prompts...")
clean_cache = get_layer_hidden_states(clean_prompt)
distracted_cache = get_layer_hidden_states(distracted_prompt)

print("\n--- Phase 2: Logit Lens Projection & JSD Calculation ---")
divergences = []

for layer_idx in range(NUM_LAYERS):
    h_clean = clean_cache[layer_idx]
    h_distract = distracted_cache[layer_idx]
    
    # Project hidden states through the model's final unembedding matrix in FP32 to prevent NaN overflow
    with torch.no_grad():
        logits_clean = lm_head(ln_f(h_clean.float()))
        logits_distract = lm_head(ln_f(h_distract.float()))

        
    # Calculate geometric divergence between the conceptual token probabilities
    jsd = jensen_shannon_divergence(logits_clean, logits_distract).item()
    divergences.append(jsd)
    
    print(f"Layer {layer_idx:02d} JSD: {jsd:.4f}")

import numpy as np
highest_drift_layers = np.argsort(divergences)[-4:] # Top 4 diverging layers
print("\n--- Mathematical Conclusion ---")
print(f"Highest Factual Drift Layers (Automated Probing targets): {sorted(highest_drift_layers.tolist())}")
print("Instead of manually hardcoding layers 4-8, the system now autonomously aims RepE steering at the exact depth the distractor mathematically shatters the factual distribution.")
