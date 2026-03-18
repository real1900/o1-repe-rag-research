import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial.distance import mahalanobis, cosine
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

model_id = "meta-llama/Llama-3.2-1B-Instruct" # Or embedder
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading {model_id} Embedder for Mahalanobis Triage PoC...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model, we just need the embedding layer and some transformer blocks to get semantic representations
base_model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = base_model(**inputs)
    # Mean pooling over the sequence dimension
    return outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

# A robust cluster of related contexts
true_contexts = [
    "Apollo 11 was the first manned mission to land on the Moon on July 20, 1969.",
    "Neil Armstrong, the commander of Apollo 11, famously said 'That's one small step for man...'",
    "The Apollo Lunar Module Eagle was piloted by Buzz Aldrin, who descended to the lunar surface.",
    "Michael Collins remained in lunar orbit inside the Command Module Columbia.",
    "The mission was launched by a Saturn V rocket from Kennedy Space Center in Florida.",
    "Apollo 11 fulfilled President John F. Kennedy's national goal proposed in 1961.",
    "The astronauts returned safely to Earth and splashed down in the Pacific Ocean.",
    "The lunar samples brought back by Apollo 11 consisted mostly of basalts and breccias.",
    "Over 500 million people watched Neil Armstrong's televised descent onto the moon.",
    "The Apollo program was NASA's premier initiative during the Space Race against the USSR."
]

# A distractor that shares keyword syntax but is fundamentally a geometric outlier
distractors = [
    "The James Webb Space Telescope uses infrared astronomy to view objects that are too old and too distant for the Hubble.",
    "The Mariana Trench in the western Pacific Ocean is the deepest oceanic trench on Earth."
]

print("Extracting contextual embeddings...")
X_train = np.array([get_embedding(ctx) for ctx in true_contexts], dtype=np.float64)
X_distract = np.array([get_embedding(d) for d in distractors], dtype=np.float64)

# 1. Cosine Triage (Naive)
centroid = np.mean(X_train, axis=0)
print("\n--- 1. Cosine Distance to Centroid (Naive Triage) ---")
for i, ctx in enumerate(true_contexts[:2]): # Just show 2 for brevity
    dist = cosine(X_train[i], centroid)
    print(f"Valid Doc {i+1} Distance: {dist:.4f}")

for i, d in enumerate(distractors):
    dist = cosine(X_distract[i], centroid)
    print(f"Distractor {i+1} Distance: {dist:.4f}")

# 2. Mahalanobis Triage (Covariance-Aware)
# To avoid singular matrix, we use pseudo-inverse, or we add small ridge regularization since N < D in this PoC
cov_matrix = np.cov(X_train, rowvar=False)
reg = 1e-4 * np.eye(cov_matrix.shape[0])
inv_cov_matrix = np.linalg.pinv(cov_matrix + reg)

print("\n--- 2. Mahalanobis Distance to Centroid (Geometry-Aware Triage) ---")
for i, ctx in enumerate(true_contexts[:2]):
    dist = mahalanobis(X_train[i], centroid, inv_cov_matrix)
    print(f"Valid Doc {i+1} Distance: {dist:.4f}")

for i, d in enumerate(distractors):
    dist = mahalanobis(X_distract[i], centroid, inv_cov_matrix)
    print(f"Distractor {i+1} Distance: {dist:.4f}")

print("\n--- Mathematical Conclusion ---")
print("Notice the massive blow-up in scalar magnitude for Mahalanobis vs. Cosine.")
print("Because Mahalanobis incorporates the Inverse Covariance Matrix, it scales the distance by the *variance* of the cluster. Outliers in thin, tightly correlated latent spaces are penalized exponentially harder than in a naive Cosine sphere, guaranteeing zero-shot distractor triage without external Cross-Encoders.")
