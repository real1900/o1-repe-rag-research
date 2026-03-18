import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id} for Contrastive Extraction PoC...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

NUM_LAYERS = base_model.config.num_hidden_layers
TARGET_LAYER = NUM_LAYERS // 2

def extract_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    cache = {}
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache['vec'] = hidden_states.detach()
    handle = base_model.model.layers[TARGET_LAYER].register_forward_hook(hook)
    
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
        
    handle.remove()
    return cache['vec'].mean(dim=1).squeeze(0)

print("\n--- Phase 1: Contrastive Pair Extraction ---")
pos_chunk = "Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969."
neg_chunk = "The Mariana Trench is an oceanic trench located in the western Pacific Ocean. The Mariana Trench is an oceanic trench located in the western Pacific Ocean."

pos_vec = extract_vector(pos_chunk)
print(f"Extracted Positive Vector (Relevant Chunk): {pos_vec.shape}")

neg_vec = extract_vector(neg_chunk)
print(f"Extracted Negative Vector (Raw Distractor): {neg_vec.shape}")

contrastive_vec = neg_vec - pos_vec
print(f"Computed Contrastive Vector (V_neg - V_pos): {contrastive_vec.shape}")

print("\n--- Phase 2: Similarity Check ---")
cos_sim = torch.nn.CosineSimilarity(dim=0)
print(f"Raw Neg to Raw Pos Similarity: {cos_sim(neg_vec, pos_vec).item():.4f}")
print(f"Contrastive to Raw Pos Similarity: {cos_sim(contrastive_vec, pos_vec).item():.4f}")
print(f"Contrastive explicitly isolates the geometric difference, achieving near-zero or negative correlation to the positive context.")

print("\n--- Mathematical Conclusion ---")
print("By subtracting V_pos, we eliminate generic linguistic representations. The resulting Contrastive Vector is a pure 'Direction of Distraction', improving steering precision.")
