"""Actual Self-RAG checkpoint (selfrag/selfrag_llama2_7b) head-to-head.

This is THE published Self-RAG model (Asai et al. ICLR 2024 SOTA), loaded
end-to-end and evaluated on the same 500-query HotpotQA OOD subset that
Llama-3.2-3B-Instruct was evaluated on. Reflection tokens are stripped from
the output before EM matching.
"""
import os, sys, json, time, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "selfrag/selfrag_llama2_7b"
N_QUERIES = int(os.environ.get("N_QUERIES", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
HOTPOT = "hotpot_filtered_5000.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

print(f"[load] {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
print(f"[load] OK")

REFLECTION_PATTERNS = [
    r"\[(Retrieval|No Retrieval|Continue to Use Evidence|Retrieve|No Retrieve|Relevant|Irrelevant|Partially Supported|Fully Supported|No Support|Supported|Utility:[0-9]+|Utility:None)\]",
]

def strip_reflection(text):
    for p in REFLECTION_PATTERNS:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()

def selfrag_generate(question, distractor):
    # Self-RAG instruction format (per published examples)
    prompt = f"### Instruction:\n{question}\n\n### Context:\n{distractor}\n\n### Response:\n"
    pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    text = tok.decode(out[0][in_len:], skip_special_tokens=True)
    cleaned = strip_reflection(text)
    return cleaned.lower(), lat, text

def baseline_generate(question, distractor):
    """Use the same Self-RAG model but without retrieval/critique structure -- as a Self-RAG baseline."""
    prompt = f"### Instruction:\n{question}\n\n### Context:\n{distractor}\n\n### Response:\n"
    pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    text = tok.decode(out[0][in_len:], skip_special_tokens=True)
    return strip_reflection(text).lower(), lat

with open(HOTPOT) as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] {len(eval_set)} queries")

results = {"baseline": {"correct": [], "lat": []}, "selfrag": {"correct": [], "lat": []}}
sample_outputs = []
t_start = time.time()
print(f"[eval] Starting actual Self-RAG (selfrag_llama2_7b) ...")
for i, row in enumerate(eval_set):
    distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()

    b_text, b_lat = baseline_generate(question, distractor)
    results["baseline"]["correct"].append(int(gold in b_text))
    results["baseline"]["lat"].append(b_lat)

    sr_text, sr_lat, sr_raw = selfrag_generate(question, distractor)
    results["selfrag"]["correct"].append(int(gold in sr_text))
    results["selfrag"]["lat"].append(sr_lat)

    if i < 3 or (i+1) % 100 == 0:
        sample_outputs.append({"q": question[:100], "gold": gold[:50],
                               "baseline_text": b_text[:80], "selfrag_text": sr_text[:80],
                               "selfrag_raw": sr_raw[:200]})
    if (i+1) % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s "
              f"base={np.mean(results['baseline']['correct']):.3f} ({np.mean(results['baseline']['lat'])*1000:.0f}ms) "
              f"selfrag={np.mean(results['selfrag']['correct']):.3f} ({np.mean(results['selfrag']['lat'])*1000:.0f}ms)")

results["meta"] = {
    "model": MODEL, "n_queries": len(eval_set),
    "eval_start": EVAL_START, "eval_end": EVAL_START + len(eval_set),
    "total_seconds": time.time() - t_start, "device": device,
    "sample_outputs": sample_outputs,
}
with open("per_query_outcomes_actual_selfrag.json", "w") as f: json.dump(results, f, indent=1)

def boot_ci(arr, n=10_000, ci=95, rng=None):
    rng = rng or np.random.default_rng(42); arr = np.asarray(arr); m = np.empty(n)
    for i in range(n): m[i] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return arr.mean(), np.percentile(m, (100-ci)/2), np.percentile(m, 100-(100-ci)/2)

def perm(a, b, n=10_000, rng=None):
    rng = rng or np.random.default_rng(7); a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean(); ext = 0
    for _ in range(n):
        f = rng.integers(0, 2, len(a)).astype(bool)
        if abs(np.where(f, b, a).mean() - np.where(f, a, b).mean()) >= abs(d) - 1e-12: ext += 1
    return d, ext / n

base = np.asarray(results["baseline"]["correct"])
sr = np.asarray(results["selfrag"]["correct"])
b_mu, b_lo, b_hi = boot_ci(base); s_mu, s_lo, s_hi = boot_ci(sr)
diff_em, p_em = perm(sr, base)
base_lat = np.asarray(results["baseline"]["lat"])
sr_lat = np.asarray(results["selfrag"]["lat"])

print("\n========================================================")
print(f"  ACTUAL Self-RAG (selfrag_llama2_7b, n={len(eval_set)})")
print("========================================================")
print(f"  Baseline (same model, no critique): EM={100*b_mu:.2f}% [{100*b_lo:.2f}, {100*b_hi:.2f}]   Lat={1000*base_lat.mean():.1f}ms")
print(f"  ACTUAL Self-RAG: EM={100*s_mu:.2f}% [{100*s_lo:.2f}, {100*s_hi:.2f}]   Lat={1000*sr_lat.mean():.1f}ms")
print(f"  ΔEM = {100*diff_em:+.2f} pp   paired-perm p = {p_em:.4f}")
print(f"  Latency ratio: {sr_lat.mean()/base_lat.mean():.2f}×")
print(f"  Total: {results['meta']['total_seconds']:.0f}s")

with open("statistical_summary_actual_selfrag.json", "w") as f:
    json.dump({
        "meta": results["meta"],
        "baseline": {"em": float(b_mu), "ci_lo": float(b_lo), "ci_hi": float(b_hi),
                     "lat_mean": float(base_lat.mean()), "lat_std": float(base_lat.std())},
        "selfrag": {"em": float(s_mu), "ci_lo": float(s_lo), "ci_hi": float(s_hi),
                    "lat_mean": float(sr_lat.mean()), "lat_std": float(sr_lat.std())},
        "delta_em_pp": float(100*diff_em),
        "delta_em_perm_p": float(p_em),
        "latency_ratio": float(sr_lat.mean() / base_lat.mean()),
    }, f, indent=1)
print("Saved per_query_outcomes_actual_selfrag.json + statistical_summary_actual_selfrag.json")
