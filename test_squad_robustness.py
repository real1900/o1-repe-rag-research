import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import string
import re
import random
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id} for SQuAD Robustness Benchmark...")
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

def kl_divergence(p_logits, q_logits):
    p_probs = torch.nn.functional.softmax(p_logits, dim=-1)
    q_log_probs = torch.nn.functional.log_softmax(q_logits, dim=-1)
    p_log_probs = torch.nn.functional.log_softmax(p_logits, dim=-1)
    kl = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1)
    return kl.item()

def generate_with_kl_braking(prompt_inputs, alpha_val, c_vecs_by_layer, max_new_tokens=15, kl_threshold=2.5):
    current_ids = prompt_inputs.input_ids
    current_alpha = alpha_val
    for step in range(max_new_tokens):
        remove_hooks()
        with torch.no_grad():
            unsteered_out = base_model(input_ids=current_ids)
            unsteered_logits = unsteered_out.logits[0, -1, :]
            
        register_targeted_hooks(current_alpha, c_vecs_by_layer)
        with torch.no_grad():
            steered_out = base_model(input_ids=current_ids)
            steered_logits = steered_out.logits[0, -1, :]
        remove_hooks()
        
        divergence = kl_divergence(unsteered_logits, steered_logits)
        
        if divergence > kl_threshold:
            current_alpha *= 0.5
            register_targeted_hooks(current_alpha, c_vecs_by_layer)
            with torch.no_grad():
                steered_out = base_model(input_ids=current_ids)
                steered_logits = steered_out.logits[0, -1, :]
            remove_hooks()
            
        next_token_id = torch.argmax(steered_logits).unsqueeze(0).unsqueeze(0)
        current_ids = torch.cat([current_ids, next_token_id], dim=1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    return current_ids

def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def contains_answer(prediction, ground_truths):
    pred_norm = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) in pred_norm:
            return 1.0
    return 0.0

def format_clean(context, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def format_contaminated(true_ctx, distractor, question):
    messages = [
        {"role": "system", "content": "You are a factual QA bot. Extract the exact short answer based ONLY on the Context provided."},
        {"role": "user", "content": f"Context:\n\nDocument A:\n{distractor}\n\nDocument B:\n{true_ctx}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("Loading SQuAD v1.1 Dataset...")
dataset = load_dataset("squad", split="validation")
random.seed(42)

# Sample 20 sequential indices to guarantee we can grab the next row as a distractor
start_idx = random.randint(0, len(dataset) - 50)
eval_size = 20

clean_correct = 0.0
contam_correct = 0.0
restored_correct = 0.0

STATIC_ALPHA = 1.0

for i in range(eval_size):
    row = dataset[start_idx + i]
    # Grab the context from 3 rows down to act as a guaranteed unrelated distractor
    distractor_row = dataset[start_idx + i + 3]
    
    true_ctx = row['context']
    distractor = distractor_row['context']
    question = row['question']
    answers = row['answers']['text']

    # 1. Clean Baseline
    clean_prompt = format_clean(true_ctx, question)
    inputs_clean = tokenizer(clean_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_clean = base_model.generate(**inputs_clean, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_clean = tokenizer.decode(out_clean[0][inputs_clean.input_ids.shape[1]:], skip_special_tokens=True).lower()
    clean_correct += contains_answer(gen_clean, answers)

    # 2. Contaminated Baseline
    contam_prompt = format_contaminated(true_ctx, distractor, question)
    inputs_contam = tokenizer(contam_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_contam = base_model.generate(**inputs_contam, max_new_tokens=15, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_contam = tokenizer.decode(out_contam[0][inputs_contam.input_ids.shape[1]:], skip_special_tokens=True).lower()
    contam_correct += contains_answer(gen_contam, answers)

    # 3. Restored (Contrastive RepE)
    # V_pos extraction
    pos_prompt = format_clean(true_ctx, question)
    # V_neg extraction
    neg_prompt = format_clean(distractor, question)
    
    c_vecs_by_layer = {}
    for layer_idx in TARGET_LAYERS:
        v_pos, _ = extract_vector(pos_prompt, f"pos_{layer_idx}", layer_idx)
        v_neg, _ = extract_vector(neg_prompt, f"neg_{layer_idx}", layer_idx)
        v_pos_mean = v_pos.mean(dim=1, keepdim=True)
        v_neg_mean = v_neg.mean(dim=1, keepdim=True)
        c_vecs_by_layer[layer_idx] = v_neg_mean - v_pos_mean # Contrastive Pair Extraction

    steered_ids = generate_with_kl_braking(inputs_contam, STATIC_ALPHA, c_vecs_by_layer, max_new_tokens=15, kl_threshold=2.0)
    gen_restored = tokenizer.decode(steered_ids[0][inputs_contam.input_ids.shape[1]:], skip_special_tokens=True).lower()
    restored_correct += contains_answer(gen_restored, answers)

    if (i+1) % 5 == 0:
        print(f"Processed {i+1}/{eval_size}...")
        print(f"  Clean: {(clean_correct/(i+1))*100:.1f}%, Contaminated: {(contam_correct/(i+1))*100:.1f}%, Restored: {(restored_correct/(i+1))*100:.1f}%")

print("\n================ SQUAD ROBUSTNESS BENCHMARK ================")
print(f"Evaluation Size: {eval_size}")
print(f"State 1: Clean Baseline           -> {(clean_correct/eval_size)*100:.2f}%")
print(f"State 2: Contaminated (Distracted) -> {(contam_correct/eval_size)*100:.2f}%")
print(f"State 3: Restored (Contrastive RepE)-> {(restored_correct/eval_size)*100:.2f}%")
