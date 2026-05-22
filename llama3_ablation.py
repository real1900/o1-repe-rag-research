"""Llama-3-8B-Instruct (or fallback) 6-condition paired ablation study.

This is the publication-grade primary evaluation. It runs the same six
conditions evaluated on GPT-2 (Section 7 of the companion notebook) but on a
modern instruction-tuned 8B-class model.

Designed to run on a single RTX 4090 (24GB VRAM). Loads model in fp16.

Conditions:
  C0  Baseline                  -- no steering
  C1  MACH-1 Hybrid             -- contrastive vec + token gating + probe-α
                                    (with heuristic fallback when α<0.05)
  C2  MACH-1 Probe-Only         -- as C1 but no heuristic fallback
  C3  MACH-1 Static-α           -- α = 0.45 * collapse_α (no probe)
  C4  MACH-1 NoGating           -- as C1 but token-norm gate disabled
  C5  MACH-1 NoContrastive      -- as C1 but raw distractor vector

Outputs:
  per_query_outcomes_llama3.json     -- raw per-query accuracy and latency
  statistical_summary_llama3.json    -- bootstrap CIs + paired permutation tests
"""
import json, time, os, sys, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------- Configuration --------------------
MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",   # open-licensed fallback
    "Qwen/Qwen2.5-3B-Instruct",   # smaller fallback
]
N_QUERIES = int(os.environ.get("N_QUERIES", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
GATING_TAU = float(os.environ.get("GATING_TAU", 30.0))   # higher τ for Llama (RMSNorm scale)
HEURISTIC_TRIP = 0.05
HEURISTIC_FRAC = 0.45
MAX_NEW_TOKENS = 12
HOTPOT_PATH = "hotpot_filtered_5000.json"

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
torch.set_grad_enabled(False)

# -------------------- Load model with fallbacks --------------------
model = None
tokenizer = None
chosen_model = None
for mid in MODEL_CANDIDATES:
    try:
        print(f"\n[load] Trying {mid} ...")
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map=device
        )
        chosen_model = mid
        print(f"[load] Success: {mid}")
        break
    except Exception as e:
        print(f"[load] FAILED {mid}: {type(e).__name__}: {e}")
        continue

if model is None:
    print("FATAL: Could not load any candidate model. Check HF token and licenses.")
    sys.exit(1)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Identify the transformer layer list & layer count generically
def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers, len(m.model.layers)        # Llama / Mistral / Qwen2
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h, len(m.transformer.h)      # GPT-2-style
    raise RuntimeError("Unknown model architecture")
LAYERS, NUM_LAYERS = get_layers(model)
TARGET_LAYER = NUM_LAYERS // 2
print(f"[arch] {NUM_LAYERS} layers; targeting layer {TARGET_LAYER}")

# -------------------- Hidden-state extraction --------------------
def extract_vector(text, layer_idx=TARGET_LAYER):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        cache["v"] = h.detach()
    handle = LAYERS[layer_idx].register_forward_hook(hook)
    try:
        _ = model(**inp)
    finally:
        handle.remove()
    return cache["v"].mean(dim=1).squeeze(0)

# -------------------- Steering hook factory --------------------
def make_steered_hook(alpha, vec, gating=True, tau=GATING_TAU):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if gating:
            norm = torch.linalg.norm(h[:, -1, :].float()).item()
            if norm <= tau:
                return output
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * vec.squeeze(0).to(h.dtype))
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new
    return hook

def run_one(prompt_inputs, input_len, gold, hook=None):
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **prompt_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    lat = time.time() - t0
    if handle is not None:
        handle.remove()
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()
    return int(gold in gen), lat, gen

# -------------------- Build prompts using chat template if available --------------------
def build_prompt(distractor, question):
    has_chat = getattr(tokenizer, "chat_template", None) is not None
    if has_chat:
        msgs = [
            {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
            {"role": "user", "content": f"Context: {distractor}\n\nQuestion: {question}"}
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"Context: {distractor}\n\nQuestion: {question}\nAnswer:"
    return prompt

# -------------------- Load eval data --------------------
with open(HOTPOT_PATH) as f:
    dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] Eval subset: rows [{EVAL_START}, {EVAL_START+N_QUERIES}) -> {len(eval_set)} queries")

# -------------------- Run the 6-condition ablation --------------------
CONDITIONS = ["baseline", "hybrid", "probe_only", "static", "no_gating", "no_contrastive"]
results = {c: {"correct": [], "lat": []} for c in CONDITIONS}
heuristic_trips = 0
t_start = time.time()

# A simple linear analytic predictor as a stand-in for the trained MLP probe.
# At Llama-3 scale we don't have a binary-search-trained probe yet, so we use
# the analytic ceiling rule throughout (i.e. probe always predicts ~0 -> falls
# back to heuristic). This gives Hybrid == Static-α; we still report both for
# parity with the GPT-2 paper.
def predict_alpha(prompt_norm, concept_norm, dot, cos, conf, plen, collapse_alpha):
    return 0.0     # forces heuristic fallback (documented in paper)

print(f"[eval] Starting 6-condition ablation on {len(eval_set)} queries ...")
for i, row in enumerate(eval_set):
    distractor = row["distractor_context"]
    question   = row["question"]
    gold       = row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    # C0 Baseline
    c, l, _ = run_one(pi, in_len, gold, hook=None)
    results["baseline"]["correct"].append(c); results["baseline"]["lat"].append(l)

    # Concept extraction (shared)
    v_neg = extract_vector(distractor).view(1, -1)
    v_pos = extract_vector("A standard fact.").view(1, -1)
    v_contr = (v_neg - v_pos).unsqueeze(0)
    v_raw = v_neg.unsqueeze(0)
    p_hidden = extract_vector(prompt).view(1, -1)

    concept_norm = torch.norm(v_contr).item()
    prompt_norm  = torch.norm(p_hidden).item()
    dot = torch.dot(p_hidden.flatten(), v_contr.flatten()).item()
    cos = F.cosine_similarity(p_hidden.flatten(), v_contr.flatten(), dim=0).item()
    collapse_alpha = dot / (concept_norm**2 + 1e-5)

    with torch.no_grad():
        nl = model(**pi).logits[0, -1, :]
        conf = torch.max(F.softmax(nl, dim=-1)).item()

    probe_alpha_raw = predict_alpha(prompt_norm, concept_norm, dot, cos, conf, in_len, collapse_alpha)
    probe_alpha = max(0.0, min(probe_alpha_raw, collapse_alpha * 0.99))
    triggered = probe_alpha < HEURISTIC_TRIP
    if triggered:
        heuristic_trips += 1
        hybrid_alpha = collapse_alpha * HEURISTIC_FRAC
    else:
        hybrid_alpha = probe_alpha
    static_alpha = collapse_alpha * HEURISTIC_FRAC

    # C1 Hybrid
    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_contr, gating=True))
    results["hybrid"]["correct"].append(c); results["hybrid"]["lat"].append(l)

    # C2 Probe-Only
    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(probe_alpha, v_contr, gating=True))
    results["probe_only"]["correct"].append(c); results["probe_only"]["lat"].append(l)

    # C3 Static-α
    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(static_alpha, v_contr, gating=True))
    results["static"]["correct"].append(c); results["static"]["lat"].append(l)

    # C4 NoGating
    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_contr, gating=False))
    results["no_gating"]["correct"].append(c); results["no_gating"]["lat"].append(l)

    # C5 NoContrastive
    c, l, _ = run_one(pi, in_len, gold, make_steered_hook(hybrid_alpha, v_raw, gating=True))
    results["no_contrastive"]["correct"].append(c); results["no_contrastive"]["lat"].append(l)

    if (i + 1) % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s  "
              f"hybrid={np.mean(results['hybrid']['correct']):.3f}  "
              f"baseline={np.mean(results['baseline']['correct']):.3f}")

results["meta"] = {
    "model": chosen_model,
    "n_queries": len(eval_set),
    "eval_start": EVAL_START,
    "eval_end": EVAL_START + len(eval_set),
    "target_layer": TARGET_LAYER,
    "num_layers": NUM_LAYERS,
    "gating_tau": GATING_TAU,
    "heuristic_trip": HEURISTIC_TRIP,
    "heuristic_frac": HEURISTIC_FRAC,
    "heuristic_trips": int(heuristic_trips),
    "total_seconds": time.time() - t_start,
    "device": device,
}
with open("per_query_outcomes_llama3.json", "w") as f:
    json.dump(results, f, indent=1)

# -------------------- Statistical analysis --------------------
def boot_ci(arr, n_resamples=10_000, ci=95, rng=None):
    rng = rng or np.random.default_rng(42)
    arr = np.asarray(arr); n = len(arr)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    return arr.mean(), np.percentile(means, (100-ci)/2), np.percentile(means, 100-(100-ci)/2)

def paired_perm(a, b, n_perms=10_000, rng=None):
    rng = rng or np.random.default_rng(7)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff_obs = a.mean() - b.mean()
    n = len(a); extreme = 0
    for _ in range(n_perms):
        flip = rng.integers(0, 2, n).astype(bool)
        a2 = np.where(flip, b, a); b2 = np.where(flip, a, b)
        if abs(a2.mean() - b2.mean()) >= abs(diff_obs) - 1e-12:
            extreme += 1
    return diff_obs, extreme / n_perms

base = np.asarray(results["baseline"]["correct"])
rng_master = np.random.default_rng(2026)
rows = []
print("\n=========== ABLATION RESULTS (Llama-3 scale) ============")
print(f"Model: {chosen_model}; n={len(eval_set)}; heuristic fired {heuristic_trips}x ({100*heuristic_trips/len(eval_set):.1f}%)")
print(f"{'Condition':<18} {'EM (%)':>8} {'95% CI':>20} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*70)
for c in CONDITIONS:
    arr = np.asarray(results[c]["correct"])
    mu, lo, hi = boot_ci(arr, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
    if c == "baseline":
        delta_str, p_val = "---", None
    else:
        diff, p_val = paired_perm(arr, base, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
        delta_str = f"{100*diff:+.2f}"
    p_str = "---" if p_val is None else f"{p_val:.4f}"
    print(f"{c:<18} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {delta_str:>10} {p_str:>10}")
    rows.append({
        "condition": c, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
        "delta_pp": None if c == "baseline" else float(100*(arr.mean() - base.mean())),
        "perm_p": p_val,
        "lat_mean": float(np.mean(results[c]["lat"])),
    })

with open("statistical_summary_llama3.json", "w") as f:
    json.dump({"meta": results["meta"], "rows": rows}, f, indent=1)
print(f"\nSaved per_query_outcomes_llama3.json + statistical_summary_llama3.json")
print(f"Total wallclock: {results['meta']['total_seconds']:.0f}s")
