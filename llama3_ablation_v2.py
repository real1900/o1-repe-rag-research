"""Llama-3 ablation v2: stronger steering coefficient.

Differences from v1:
  - HEURISTIC_FRAC = 0.85 (was 0.45) -- more aggressive steering
  - Adds an extra condition C6 'Aggressive' with κ=0.95 to find the ceiling
  - Uses GATING_TAU adapted from observed mid-stream norms
"""
import os, sys
sys.path.insert(0, "/workspace")
os.environ["HEURISTIC_FRAC"] = os.environ.get("HEURISTIC_FRAC", "0.85")
os.environ["GATING_TAU"] = os.environ.get("GATING_TAU", "5.0")

# Import the v1 module after env vars set, but we want to override config first.
# Cleaner: just inline the relevant changes by re-running the script with patched constants.
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
N_QUERIES = int(os.environ.get("N_QUERIES", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
GATING_TAU = float(os.environ["GATING_TAU"])
HEURISTIC_FRAC = float(os.environ["HEURISTIC_FRAC"])
HEURISTIC_FRAC_STRONG = 0.95
MAX_NEW_TOKENS = 12
HOTPOT_PATH = "hotpot_filtered_5000.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}; GATING_TAU={GATING_TAU}; HEURISTIC_FRAC={HEURISTIC_FRAC}; STRONG={HEURISTIC_FRAC_STRONG}")
torch.set_grad_enabled(False)

model = None; tokenizer = None; chosen_model = None
for mid in MODEL_CANDIDATES:
    try:
        print(f"[load] Trying {mid} ...")
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=device)
        chosen_model = mid; print(f"[load] Success: {mid}"); break
    except Exception as e:
        print(f"[load] FAILED {mid}: {type(e).__name__}: {str(e)[:100]}"); continue
if model is None:
    print("FATAL: Could not load any candidate."); sys.exit(1)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"): return m.model.layers, len(m.model.layers)
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"): return m.transformer.h, len(m.transformer.h)
    raise RuntimeError("Unknown model")
LAYERS, NUM_LAYERS = get_layers(model)
TARGET_LAYER = NUM_LAYERS // 2
print(f"[arch] {NUM_LAYERS} layers; targeting layer {TARGET_LAYER}")

def extract_vector(text, layer_idx=TARGET_LAYER):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        cache["v"] = h.detach()
    handle = LAYERS[layer_idx].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    return cache["v"].mean(dim=1).squeeze(0)

def make_steered_hook(alpha, vec, gating=True, tau=GATING_TAU):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if gating:
            norm = torch.linalg.norm(h[:, -1, :].float()).item()
            if norm <= tau: return output
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * vec.squeeze(0).to(h.dtype))
        if isinstance(output, tuple): return (h_new,) + output[1:]
        return h_new
    return hook

def run_one(prompt_inputs, input_len, gold, hook=None):
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**prompt_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()
    return int(gold in gen), lat, gen

def build_prompt(distractor, question):
    if getattr(tokenizer, "chat_template", None) is not None:
        msgs = [
            {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
            {"role": "user", "content": f"Context: {distractor}\n\nQuestion: {question}"}
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"

with open(HOTPOT_PATH) as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] {len(eval_set)} queries from rows [{EVAL_START}, {EVAL_START+len(eval_set)})")

# Probe diagnostic norms first (run on first 5 queries to set adaptive τ)
sample_norms = []
for row in eval_set[:5]:
    prompt = build_prompt(row["distractor_context"], row["question"])
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    cache = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cache["norms"] = [torch.linalg.norm(h[0, t, :].float()).item() for t in range(h.shape[1])]
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook); _ = model(**pi); handle.remove()
    sample_norms.extend(cache["norms"])
print(f"[diag] sample residual norms at layer {TARGET_LAYER}: min={min(sample_norms):.2f} median={np.median(sample_norms):.2f} max={max(sample_norms):.2f}")

CONDITIONS = ["baseline", "hybrid_strong", "static_strong", "static_aggressive", "no_gating", "no_contrastive"]
results = {c: {"correct": [], "lat": []} for c in CONDITIONS}
heuristic_trips = 0; t_start = time.time()

print(f"[eval] Starting strong-α 6-condition ablation on {len(eval_set)} queries ...")
for i, row in enumerate(eval_set):
    distractor = row["distractor_context"]; question = row["question"]; gold = row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    c, l, _ = run_one(pi, in_len, gold, hook=None)
    results["baseline"]["correct"].append(c); results["baseline"]["lat"].append(l)

    v_neg = extract_vector(distractor).view(1, -1)
    v_pos = extract_vector("A standard fact.").view(1, -1)
    v_contr = (v_neg - v_pos).unsqueeze(0)
    v_raw = v_neg.unsqueeze(0)
    p_hidden = extract_vector(prompt).view(1, -1)
    concept_norm = torch.norm(v_contr).item()
    dot = torch.dot(p_hidden.flatten(), v_contr.flatten()).item()
    collapse_alpha = dot / (concept_norm**2 + 1e-5)

    hybrid_alpha = collapse_alpha * HEURISTIC_FRAC          # κ=0.85
    static_alpha = collapse_alpha * HEURISTIC_FRAC          # same as hybrid here
    aggressive_alpha = collapse_alpha * HEURISTIC_FRAC_STRONG  # κ=0.95
    heuristic_trips += 1

    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_contr, gating=True))
    results["hybrid_strong"]["correct"].append(c); results["hybrid_strong"]["lat"].append(l)

    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(static_alpha, v_contr, gating=True))
    results["static_strong"]["correct"].append(c); results["static_strong"]["lat"].append(l)

    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(aggressive_alpha, v_contr, gating=True))
    results["static_aggressive"]["correct"].append(c); results["static_aggressive"]["lat"].append(l)

    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_contr, gating=False))
    results["no_gating"]["correct"].append(c); results["no_gating"]["lat"].append(l)

    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_raw, gating=True))
    results["no_contrastive"]["correct"].append(c); results["no_contrastive"]["lat"].append(l)

    if (i+1) % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s  "
              + " ".join(f"{c}={np.mean(results[c]['correct']):.3f}" for c in CONDITIONS))

results["meta"] = {
    "model": chosen_model, "n_queries": len(eval_set), "eval_start": EVAL_START,
    "eval_end": EVAL_START + len(eval_set), "target_layer": TARGET_LAYER,
    "num_layers": NUM_LAYERS, "gating_tau": GATING_TAU,
    "heuristic_frac": HEURISTIC_FRAC, "heuristic_frac_strong": HEURISTIC_FRAC_STRONG,
    "heuristic_trips": int(heuristic_trips),
    "total_seconds": time.time() - t_start, "device": device,
    "residual_norm_diag": {"min": float(min(sample_norms)), "median": float(np.median(sample_norms)), "max": float(max(sample_norms))},
}
with open("per_query_outcomes_llama3_v2.json", "w") as f: json.dump(results, f, indent=1)

def boot_ci(arr, n_resamples=10_000, ci=95, rng=None):
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(arr); n = len(arr); means = np.empty(n_resamples)
    for i in range(n_resamples):
        means[i] = arr[rng.integers(0, n, n)].mean()
    return arr.mean(), np.percentile(means, (100-ci)/2), np.percentile(means, 100-(100-ci)/2)

def paired_perm(a, b, n_perms=10_000, rng=None):
    rng = rng or np.random.default_rng(7)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff_obs = a.mean() - b.mean(); n = len(a); extreme = 0
    for _ in range(n_perms):
        flip = rng.integers(0, 2, n).astype(bool)
        a2, b2 = np.where(flip, b, a), np.where(flip, a, b)
        if abs(a2.mean() - b2.mean()) >= abs(diff_obs) - 1e-12: extreme += 1
    return diff_obs, extreme / n_perms

base = np.asarray(results["baseline"]["correct"])
rng_master = np.random.default_rng(2026)
rows = []
print(f"\n=========== STRONG-α ABLATION (model={chosen_model}, n={len(eval_set)}) ============")
print(f"{'Condition':<22} {'EM (%)':>8} {'95% CI':>20} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*72)
for cond in CONDITIONS:
    arr = np.asarray(results[cond]["correct"])
    mu, lo, hi = boot_ci(arr, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
    if cond == "baseline":
        delta_str, p_val = "---", None
    else:
        diff, p_val = paired_perm(arr, base, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
        delta_str = f"{100*diff:+.2f}"
    p_str = "---" if p_val is None else f"{p_val:.4f}"
    print(f"{cond:<22} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {delta_str:>10} {p_str:>10}")
    rows.append({
        "condition": cond, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
        "delta_pp": None if cond == "baseline" else float(100*(arr.mean() - base.mean())),
        "perm_p": p_val, "lat_mean": float(np.mean(results[cond]["lat"])),
    })
with open("statistical_summary_llama3_v2.json", "w") as f:
    json.dump({"meta": results["meta"], "rows": rows}, f, indent=1)
print(f"\nSaved per_query_outcomes_llama3_v2.json + statistical_summary_llama3_v2.json")
print(f"Total wallclock: {results['meta']['total_seconds']:.0f}s")
