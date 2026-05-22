"""Layer-7 validation: focused replication on n=500 with α-sweep.

The layer sweep showed Layer 7 + α=2 gives the only positive trend (+1.0pp,
not significant at n=200). This script validates with double the sample size
and an α-sweep to characterize the effect curve."""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TARGET_LAYER = 7
ALPHAS = [0.0, 1.0, 2.0, 3.0]
N_QUERIES = 500
EVAL_START = 3500
MAX_NEW_TOKENS = 12

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

mid = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=device)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.eval()
LAYERS = model.model.layers
print(f"[arch] {mid}: layer={TARGET_LAYER}/{len(LAYERS)}; α-sweep={ALPHAS}; n={N_QUERIES}")

def extract_unit_vec(text):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o; cache["v"] = h.detach()
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    return cache["v"].mean(dim=1).squeeze(0)

def make_hook(alpha, unit_vec):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if alpha == 0.0: return o
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * unit_vec.to(h.dtype))
        if isinstance(o, tuple): return (h_new,) + o[1:]
        return h_new
    return hook

def run_one(pi, in_len, gold, hook):
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    gen = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).lower()
    return int(gold in gen), lat

def build_prompt(distractor, question):
    msgs = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user", "content": f"Context: {distractor}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json") as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]

CONDITIONS = [f"alpha_{a:g}" for a in ALPHAS]
results = {c: {"correct": [], "lat": []} for c in CONDITIONS}
t0 = time.time()
print(f"[eval] Starting {len(ALPHAS)}-α sweep at layer {TARGET_LAYER} on {len(eval_set)} queries")
for i, row in enumerate(eval_set):
    distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    v_neg = extract_unit_vec(distractor)
    v_pos = extract_unit_vec("A standard fact.")
    v_contr = v_neg - v_pos
    n = torch.linalg.norm(v_contr)
    unit = v_contr / n if n.item() > 1e-6 else torch.zeros_like(v_contr)

    for a, c in zip(ALPHAS, CONDITIONS):
        if a == 0.0:
            ok, lat = run_one(pi, in_len, gold, hook=None)
        else:
            ok, lat = run_one(pi, in_len, gold, hook=make_hook(a, unit))
        results[c]["correct"].append(ok); results[c]["lat"].append(lat)

    if (i+1) % 25 == 0:
        elapsed = time.time() - t0
        s = " ".join(f"α={a}={np.mean(results[c]['correct']):.3f}" for a, c in zip(ALPHAS, CONDITIONS))
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s {s}")

results["meta"] = {
    "model": mid, "target_layer": TARGET_LAYER, "alphas": ALPHAS, "n_queries": len(eval_set),
    "eval_start": EVAL_START, "eval_end": EVAL_START + len(eval_set),
    "total_seconds": time.time() - t0, "device": device,
}
with open("per_query_outcomes_layer7.json", "w") as f: json.dump(results, f, indent=1)

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

base = np.asarray(results["alpha_0"]["correct"])
print(f"\n=========== LAYER {TARGET_LAYER} α-SWEEP (n={len(eval_set)}) ============")
print(f"{'α':<10} {'EM (%)':>8} {'95% CI':>20} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*60)
rows = []
rng = np.random.default_rng(2026)
for a, c in zip(ALPHAS, CONDITIONS):
    arr = np.asarray(results[c]["correct"])
    mu, lo, hi = boot_ci(arr, rng=np.random.default_rng(rng.integers(0, 2**31)))
    if a == 0.0:
        d_str, p = "---", None
    else:
        d, p = perm(arr, base, rng=np.random.default_rng(rng.integers(0, 2**31)))
        d_str = f"{100*d:+.2f}"
    p_str = "---" if p is None else f"{p:.4f}"
    print(f"{a:<10g} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {d_str:>10} {p_str:>10}")
    rows.append({"alpha":a,"condition":c,"em":float(mu),"ci_lo":float(lo),"ci_hi":float(hi),"delta_pp":None if a==0 else float(100*(arr.mean()-base.mean())),"perm_p":p,"lat_mean":float(np.mean(results[c]["lat"]))})
with open("statistical_summary_layer7.json", "w") as f: json.dump({"meta": results["meta"], "rows": rows}, f, indent=1)
print(f"\nSaved per_query_outcomes_layer7.json + statistical_summary_layer7.json")
print(f"Wallclock: {results['meta']['total_seconds']:.0f}s")
