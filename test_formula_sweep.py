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

# Distractor Concept
dummy_text = "This paragraph contains completely irrelevant, distracting information about a side topic." * 3
dummy_input = tokenizer(dummy_text, return_tensors="pt").input_ids.to(device)

hook_handle = target_layer.register_forward_hook(get_activation("concept"))
with torch.no_grad():
    _ = base_model(input_ids=dummy_input)
extracted_vector = activation_cache["concept"]
hook_handle.remove()
concept_vector = extracted_vector.mean(dim=1, keepdim=True)
concept_norm_sq = torch.norm(concept_vector).item() ** 2

test_prompts = [
    "The capital of France is",
    "The largest city in Japan is",
    "The president of the United States lives in"
]

def create_steering_hook(alpha_val, c_vec):
    def steering_hook(module, args, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        steered_states = hidden_states - (alpha_val * c_vec)
        if isinstance(output, tuple):
            return (steered_states,) + output[1:]
        else:
            return steered_states
    return steering_hook

def find_transition_alpha(prompt_text):
    prompt_input = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    
    # 1. Get Prompt Vector and calculate Collapse Alpha
    hook_handle = target_layer.register_forward_hook(get_activation("prompt"))
    with torch.no_grad():
        _ = base_model(input_ids=prompt_input)
    prompt_vector = activation_cache["prompt"][:, -1:, :]
    hook_handle.remove()
    
    dot_product = torch.dot(prompt_vector.flatten(), concept_vector.flatten()).item()
    collapse_alpha = dot_product / concept_norm_sq
    
    # 2. Sweep to find transition bounds (rough sweep)
    alphas = np.linspace(0.0, collapse_alpha, 20)
    baseline_text = None
    last_text = None
    transition_alpha = None
    
    distinct_texts = []
    
    for alpha in alphas:
        hook_handle = target_layer.register_forward_hook(create_steering_hook(alpha, concept_vector))
        with torch.no_grad():
            out = base_model.generate(
                **tokenizer(prompt_text, return_tensors="pt").to(device),
                max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True).replace('\n', ' ')
        hook_handle.remove()
        
        if baseline_text is None:
            baseline_text = text
            
        if text != last_text and last_text is not None:
            distinct_texts.append((alpha, text))
            
        last_text = text
    
    # Just take the first major semantic shift as the sweet spot for analysis
    if len(distinct_texts) > 0:
        transition = distinct_texts[0][0]
    else:
        transition = 0.0
        
    prompt_norm = torch.norm(prompt_vector).item()
    cosine_sim = torch.nn.functional.cosine_similarity(prompt_vector.flatten(), concept_vector.flatten(), dim=0).item()
    
    return {
        "collapse": collapse_alpha,
        "sweet_spot": transition,
        "ratio": transition / collapse_alpha if collapse_alpha > 0 else 0,
        "cosine_sim": cosine_sim,
        "prompt_norm": prompt_norm,
        "concept_norm": torch.norm(concept_vector).item(),
        "texts": [t[1] for t in distinct_texts[:2]]
    }

print("--- Hunting for the Universal Formula ---")
for p in test_prompts:
    res = find_transition_alpha(p)
    print(f"\nPrompt: '{p}'")
    print(f"Collapse Alpha: {res['collapse']:.4f}")
    print(f"First Semantic Sweet Spot found at roughly: {res['sweet_spot']:.4f}")
    print(f"Ratio (SweetSpot / Collapse): {res['ratio']:.4f}")
    print(f"Cosine Sim: {res['cosine_sim']:.4f}")
