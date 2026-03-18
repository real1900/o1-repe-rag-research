import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import string
import re

logging.getLogger("transformers").setLevel(logging.ERROR)

# =================================================================================
# Llama-3.2-1B-Instruct Evaluation with KL-Divergence Braking
# 
# Objective: Prove that a dynamic KL-Divergence Braking decoding loop stops the 
# Semantic Collapse observed in naive static steering, allowing the model to hit
# accuracy improvements over the baseline.
# =================================================================================

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = "meta-llama/Llama-3.2-1B-Instruct"

print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

NUM_LAYERS = base_model.config.num_hidden_layers
# Targeted Fact Retrieval Layers
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

def kl_divergence(p_logits, q_logits):
    p_probs = torch.nn.functional.softmax(p_logits, dim=-1)
    q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    kl = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    return kl.item()

def generate_with_kl_braking(prompt_inputs, alpha_val, c_vecs_by_layer, max_new_tokens=15, kl_threshold=2.5):
    """
    Custom decoding loop that dynamically monitors semantic collapse via KL-Divergence.
    """
    current_ids = prompt_inputs.input_ids
    
    current_alpha = alpha_val
    generated_tokens = []
    
    for step in range(max_new_tokens):
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
            # Semantic Collapse Detected! Brake by decaying alpha for this token
            current_alpha *= 0.5
            register_targeted_hooks(current_alpha, c_vecs_by_layer)
            with torch.no_grad():
                out_steered = base_model(input_ids=current_ids)
                steered_logits = out_steered.logits[0, -1, :]
            remove_hooks()
            
        next_token_id = torch.argmax(steered_logits).unsqueeze(0).unsqueeze(0)
        generated_tokens.append(next_token_id.item())
        current_ids = torch.cat([current_ids, next_token_id], dim=1)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
    return current_ids

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

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def contains_answer(prediction, ground_truth):
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    if gt_norm in pred_norm:
        return 1.0
    return 0.0

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Use last 30 for speed of local execution
eval_set = dataset[-20:]

baseline_correct = 0.0
repe_correct = 0.0
STATIC_ALPHA = 0.50 # Gentle steering coefficient

print("\nEvaluating KL-Braking on Mixed Context using Llama-3.2-1B-Instruct...")
for i, row in enumerate(eval_set):
    true_ctx = row['true_context']
    distractor = row['distractor_context']
    question = row['question']
    answer = row['answer'].lower()

    mixed_prompt = format_mixed_prompt(true_ctx, distractor, question)
    inputs = tokenizer(mixed_prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # Baseline: Distracted Model (Standard Prompting)
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_base = tokenizer.decode(out_base[0][input_len:], skip_special_tokens=True).lower()
    
    baseline_correct += contains_answer(gen_base, answer)

    # Extract distractor vectors
    dist_prompt = format_distractor_query(distractor, question)
    c_vecs_by_layer = {}
    for layer_idx in TARGET_LAYERS:
        c_h, _ = extract_vector(dist_prompt, f"c_{layer_idx}", layer_idx)
        c_vecs_by_layer[layer_idx] = c_h.mean(dim=1, keepdim=True)

    # O(1) RepE Steering mathematically applied via KL-Braking decode loop
    steered_ids = generate_with_kl_braking(inputs, STATIC_ALPHA, c_vecs_by_layer, max_new_tokens=15, kl_threshold=2.0)
    gen_repe = tokenizer.decode(steered_ids[0][input_len:], skip_special_tokens=True).lower()

    repe_correct += contains_answer(gen_repe, answer)

    if (i+1) % 5 == 0:
        print(f"Processed {i+1}/{len(eval_set)}... (Baseline Acc: {(baseline_correct/(i+1))*100:.1f}%, Steered Acc: {(repe_correct/(i+1))*100:.1f}%)")

print("\n================ FINAL KL-BRAKING MIXED CONTEXT RESULTS ================")
print(f"Total Evaluation Queries: {len(eval_set)}")
base_acc = (baseline_correct / len(eval_set)) * 100
repe_acc = (repe_correct / len(eval_set)) * 100

print(f"\n[Baseline Llama-3 Unsteered (Distracted)]")
print(f"Accuracy (Normalized Substring Match): {base_acc:.2f}%")

print(f"\n[O(1) RepE Steered w/ KL-Braking (Protected Logic)]")
print(f"Accuracy (Normalized Substring Match): {repe_acc:.2f}%")

if repe_acc > base_acc:
    relative_jump = ((repe_acc - base_acc) / base_acc) * 100
    print(f"\nSUCCESS: KL-Braking prevented Semantic Collapse and delivered a +{relative_jump:.2f}% Relative Gain!")
elif repe_acc == base_acc:
    print(f"\nSTABILITY: KL-Braking prevented Semantic Collapse and maintained baseline accuracy.")
else:
    print(f"\nFAILURE: Steered accuracy is lower than baseline.")
print("=====================================================================")
