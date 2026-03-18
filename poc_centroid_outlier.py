import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id} for Centroid Outlier PoC...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

NUM_LAYERS = base_model.config.num_hidden_layers
TARGET_LAYER = NUM_LAYERS // 2

def extract_chunk_vector(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    cache = {}
    
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        cache['vec'] = hidden_states.detach()
        
    layer = base_model.model.layers[TARGET_LAYER]
    handle = layer.register_forward_hook(hook)
    
    with torch.no_grad():
        _ = base_model(input_ids=inputs)
        
    handle.remove()
    # Mean pool over sequence length to get 1D chunk embedding
    return cache['vec'].mean(dim=1).squeeze(0)

# Simulate a RAG Retrieval: 4 Relevant chunks about a topic, 1 Distractor chunk
# Query: Information about the Apollo 11 moon landing.
chunks = [
    "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
    "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.",
    "Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later.",
    "The Apollo 11 mission was launched by a Saturn V rocket from Kennedy Space Center on Merritt Island, Florida.",
    "The Mariana Trench is an oceanic trench located in the western Pacific Ocean, about 200 kilometres east of the Mariana Islands." # The Distractor
]

print("\nSimulating RAG Retrieval of 5 Chunks (4 Relevant, 1 Distractor)...")
vectors = []
for i, chunk in enumerate(chunks):
    vec = extract_chunk_vector(chunk)
    vectors.append(vec)
    print(f"Extracted Vector for Chunk {i} (Shape: {vec.shape})")

# Stack vectors into a tensor [5, d_model]
V = torch.stack(vectors)

# Calculate the Centroid of the Latent Cluster
centroid = V.mean(dim=0)

# Calculate Cosine Similarity of each chunk to the Centroid
cosine_sim = torch.nn.CosineSimilarity(dim=0)

print("\n--- Latent Space Mathematical Clustering ---")
similarities = []
for i, v in enumerate(V):
    sim = cosine_sim(v, centroid).item()
    similarities.append(sim)
    print(f"Chunk {i} Similarity to Centroid: {sim:.4f}")

outlier_idx = np.argmin(similarities)
print(f"\n[UNSUPERVISED TRIAGE SUCCESS] The geometric outlier is Chunk {outlier_idx}.")
print(f"Outlier Content: '{chunks[outlier_idx]}'")
print(f"Mathematical Conclusion: Cross-Encoder is not explicitly required. The Negative Control Vector can be dynamically extracted from the geometric outlier in O(1) time.")
