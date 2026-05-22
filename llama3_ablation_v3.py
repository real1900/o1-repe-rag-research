"""Llama-3 ablation v3: unit-direction α-sweep.

Key fix from v1/v2: Use UNIT-NORM steering direction with fixed α coefficient,
the standard RepE/CAA convention (Zou et al. 2023, Rimsky et al. 2023). The
v1/v2 'collapse_alpha' formula was tied to the prompt's projection onto the
contrastive vector, which is near-zero by construction (subtraction removes
shared structure), making the effective steering ~0.

This script sweeps α ∈ {0, 1, 2, 5, 10, 20} with v̂ = v_contrastive / ||v_contrastive||
to find the regime where mechanistic steering actually moves the output.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
N_QUERIES = int(os.environ.get("N_QUERIES", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
ALPHAS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]   # α-sweep on UNIT vectors
GATING_TAU = 0.0  # disable gating in v3 to isolate α effect
MAX_NEW_TOKENS = 12

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}; α-sweep={ALPHAS} (unit-norm direction); gating disabled")
torch.set_grad_enabled(False)

model = None; tokenizer = None; chosen_model = None
for mid in MODEL_CANDIDATES:
    try:
        print(f"[load] Trying {mid} ...")
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=device)
        chosen_model = mid; print(f"[load] Success: {mid}"); break
    except Exception as e:
        print(f"[load] FAILED {mid}: {type(e).__name__}: {str(e)[:80]}"); continue
if model is None: sys.exit("No model loaded")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.eval()

LAYERS = model.model.layers if hasattr(model, "model") else model.transformer.h
NUM_LAYERS = len(LAYERS); TARGET_LAYER = NUM_LAYERS // 2
print(f"[arch] {NUM_LAYERS} layers; targeting {TARGET_LAYER}")

def extract_unit_vec(text, layer_idx=TARGET_LAYER):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cache["v"] = h.detach()
    handle = LAYERS[layer_idx].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    v = cache["v"].mean(dim=1).squeeze(0)
    return v

def make_unit_steered_hook(alpha, unit_vec):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * unit_vec.to(h.dtype))
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

with open("hotpot_filtered_5000.json") as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] {len(eval_set)} queries from rows [{EVAL_START}, {EVAL_START+len(eval_set)})")

CONDITIONS = [f"alpha_{a:g}" for a in ALPHAS]   # baseline = alpha_0
results = {c: {"correct": [], "lat": []} for c in CONDITIONS}
t_start = time.time()
print(f"[eval] Starting α-sweep with {len(ALPHAS)} conditions on {len(eval_set)} queries ...")

for i, row in enumerate(eval_set):
    distractor = row["distractor_context"]; question = row["question"]; gold = row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    # Compute unit contrastive direction
    v_neg = extract_unit_vec(distractor)
    v_pos = extract_unit_vec("A standard fact.")
    v_contr = v_neg - v_pos
    n = torch.linalg.norm(v_contr)
    if n.item() < 1e-6:
        unit_vec = torch.zeros_like(v_contr)
    else:
        unit_vec = v_contr / n

    for a, cond in zip(ALPHAS, CONDITIONS):
        if a == 0.0:
            c, l, _ = run_one(pi, in_len, gold, hook=None)
        else:
            c, l, _ = run_one(pi, in_len, gold, make_unit_steered_hook(a, unit_vec))
        results[cond]["correct"].append(c); results[cond]["lat"].append(l)

    if (i+1) % 20 == 0:
        elapsed = time.time() - t_start
        means = {c: np.mean(results[c]["correct"]) for c in CONDITIONS}
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s  " + " ".join(f"{c}={m:.3f}" for c, m in means.items()))

results["meta"] = {
    "model": chosen_model, "n_queries": len(eval_set), "alphas": ALPHAS,
    "target_layer": TARGET_LAYER, "num_layers": NUM_LAYERS,
    "eval_start": EVAL_START, "eval_end": EVAL_START + len(eval_set),
    "total_seconds": time.time() - t_start, "device": device,
    "steering_form": "unit_direction",
}
with open("per_query_outcomes_llama3_v3.json", "w") as f: json.dump(results, f, indent=1)

def boot_ci(arr, n_resamples=10_000, ci=95, rng=None):
    rng = rng or np.random.default_rng(42); arr = np.asarray(arr); n = len(arr); means = np.empty(n_resamples)
    for i in range(n_resamples): means[i] = arr[rng.integers(0, n, n)].mean()
    return arr.mean(), np.percentile(means, (100-ci)/2), np.percentile(means, 100-(100-ci)/2)

def paired_perm(a, b, n_perms=10_000, rng=None):
    rng = rng or np.random.default_rng(7); a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff_obs = a.mean() - b.mean(); n = len(a); extreme = 0
    for _ in range(n_perms):
        flip = rng.integers(0, 2, n).astype(bool)
        if abs(np.where(flip, b, a).mean() - np.where(flip, a, b).mean()) >= abs(diff_obs) - 1e-12: extreme += 1
    return diff_obs, extreme / n_perms

base = np.asarray(results["alpha_0"]["correct"])
rng_master = np.random.default_rng(2026); rows = []
print(f"\n=========== UNIT-DIRECTION α-SWEEP (model={chosen_model}, n={len(eval_set)}) ============")
print(f"{'α':<10} {'EM (%)':>8} {'95% CI':>20} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*60)
for a, cond in zip(ALPHAS, CONDITIONS):
    arr = np.asarray(results[cond]["correct"])
    mu, lo, hi = boot_ci(arr, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
    if a == 0.0:
        delta_str, p_val = "---", None
    else:
        diff, p_val = paired_perm(arr, base, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
        delta_str = f"{100*diff:+.2f}"
    p_str = "---" if p_val is None else f"{p_val:.4f}"
    print(f"{a:<10g} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {delta_str:>10} {p_str:>10}")
    rows.append({
        "alpha": a, "condition": cond, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
        "delta_pp": None if a == 0.0 else float(100*(arr.mean() - base.mean())),
        "perm_p": p_val, "lat_mean": float(np.mean(results[cond]["lat"])),
    })
with open("statistical_summary_llama3_v3.json", "w") as f: json.dump({"meta": results["meta"], "rows": rows}, f, indent=1)
print(f"\nSaved per_query_outcomes_llama3_v3.json + statistical_summary_llama3_v3.json")
print(f"Total wallclock: {results['meta']['total_seconds']:.0f}s")
