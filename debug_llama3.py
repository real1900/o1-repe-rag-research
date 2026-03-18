import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

NUM_LAYERS = base_model.config.num_hidden_layers
TARGET_LAYERS = range(NUM_LAYERS // 4, NUM_LAYERS // 2)

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
    for h in active_hooks:
        h.remove()
    active_hooks = []
    for layer_idx in TARGET_LAYERS:
        layer = base_model.model.layers[layer_idx]
        hook = layer.register_forward_hook(create_steering_hook(alpha_val, c_vecs_by_layer[layer_idx]))
        active_hooks.append(hook)

def remove_hooks():
    global active_hooks
    for h in active_hooks:
        h.remove()
    active_hooks = []

def format_mixed_prompt(true_ctx, distractor, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n\nDocument A:\n{distractor}\n\nDocument B:\n{true_ctx}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_distractor_query(distractor, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n\nDocument A:\n{distractor}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print("Starting debug generation...")
for i in range(2):
    row = dataset[-30:][i]
    question = row['question']
    distractor = row['distractor_context']
    true_ctx = row['true_context']
    
    mixed_prompt = format_mixed_prompt(true_ctx, distractor, question)
    inputs = tokenizer(mixed_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=True).strip()
    
    dist_prompt = format_distractor_query(distractor, question)
    c_vecs_by_layer = {}
    for layer_idx in TARGET_LAYERS:
        c_h, _ = extract_vector(dist_prompt, f"c_{layer_idx}", layer_idx)
        c_vecs_by_layer[layer_idx] = c_h.mean(dim=1, keepdim=True)

    register_targeted_hooks(0.50, c_vecs_by_layer)
    with torch.no_grad():
        out_steered = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_steered = tokenizer.decode(out_steered[0][input_len:], skip_special_tokens=True).strip()
    remove_hooks()
    
    print(f"\n[Query {i+1}] {question}")
    print(f"Ground Truth: {row['answer']}")
    print(f"Baseline Gen: {gen_base}")
    print(f"Steered Gen: {gen_steered}")
