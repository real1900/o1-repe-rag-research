"""Random-direction control: is the contrastive direction load-bearing?

Compares MACH-1's contrastive steering vs:
  (i) a random unit vector at the same magnitude (control: pure noise injection)
  (ii) the prompt's own activation as the direction (sanity check)

If the contrastive direction is meaningful, it should outperform random.
If random gives the same effect, the +1pp on Llama-3.2-3B layer 7 is just
generic noise injection / regularization, not distractor-specific suppression.

Run on Llama-3.2-3B at ℓ=7, α=2 (the headline config) on n=500 paired queries.
"""
import os, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.2-3B-Instruct"
TARGET_LAYER = 7
ALPHA = 2.0
N = 500
EVAL_START = 3500
HOTPOT = "hotpot_filtered_5000.json"
SEED = 2026

device = "cuda"
torch.set_grad_enabled(False)
torch.manual_seed(SEED)

print(f"[load] {MODEL}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYERS = model.model.layers
D_MODEL = model.config.hidden_size
print(f"[arch] hidden={D_MODEL}, layer={TARGET_LAYER}/{len(LAYERS)}, α={ALPHA}, n={N}")

def extract_unit(text):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cache["v"] = h.detach()
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    return cache["v"].mean(dim=1).squeeze(0)

def make_hook(alpha, unit):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * unit.to(h.dtype))
        if isinstance(o, tuple): return (h_new,) + o[1:]
        return h_new
    return hook

def run_one(pi, in_len, gold, hook):
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    return int(gold in tok.decode(out[0][in_len:], skip_special_tokens=True).lower()), lat

def build_prompt(d, q):
    msgs = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user", "content": f"Context: {d}\n\nQuestion: {q}"}
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

with open(HOTPOT) as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N]

# 5 random unit vectors (held fixed across queries within each random condition)
g = torch.Generator(device=device).manual_seed(SEED)
random_dirs = [torch.randn(D_MODEL, generator=g, device=device, dtype=torch.float32) for _ in range(5)]
random_dirs = [d / torch.linalg.norm(d) for d in random_dirs]

CONDS = ["baseline", "contrastive"] + [f"random_{i}" for i in range(5)] + ["random_per_query"]
results = {c: {"correct": [], "lat": []} for c in CONDS}

print(f"[eval] starting on {len(eval_set)} queries with 8 conditions...")
t_start = time.time()
for i, row in enumerate(eval_set):
    distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    # Baseline
    c, l = run_one(pi, in_len, gold, hook=None)
    results["baseline"]["correct"].append(c); results["baseline"]["lat"].append(l)

    # Contrastive (the actual MACH-1 direction)
    v_neg = extract_unit(distractor)
    v_pos = extract_unit("A standard fact.")
    v_contr = v_neg - v_pos
    n = torch.linalg.norm(v_contr)
    unit_contr = v_contr / n if n.item() > 1e-6 else torch.zeros_like(v_contr)
    c, l = run_one(pi, in_len, gold, make_hook(ALPHA, unit_contr))
    results["contrastive"]["correct"].append(c); results["contrastive"]["lat"].append(l)

    # 5 fixed random directions
    for k, rd in enumerate(random_dirs):
        c, l = run_one(pi, in_len, gold, make_hook(ALPHA, rd))
        results[f"random_{k}"]["correct"].append(c); results[f"random_{k}"]["lat"].append(l)

    # 1 per-query fresh random direction
    g_q = torch.Generator(device=device).manual_seed(SEED + i)
    rd_q = torch.randn(D_MODEL, generator=g_q, device=device, dtype=torch.float32)
    rd_q = rd_q / torch.linalg.norm(rd_q)
    c, l = run_one(pi, in_len, gold, make_hook(ALPHA, rd_q))
    results["random_per_query"]["correct"].append(c); results["random_per_query"]["lat"].append(l)

    if (i + 1) % 25 == 0:
        elapsed = time.time() - t_start
        b = np.mean(results['baseline']['correct'])
        cn = np.mean(results['contrastive']['correct'])
        rmean = np.mean([np.mean(results[f'random_{k}']['correct']) for k in range(5)])
        rpq = np.mean(results['random_per_query']['correct'])
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s  base={b:.3f}  contrastive={cn:.3f}  random_avg={rmean:.3f}  random_pq={rpq:.3f}")

results["meta"] = {
    "model": MODEL, "n_queries": len(eval_set), "target_layer": TARGET_LAYER, "alpha": ALPHA,
    "seed": SEED, "total_seconds": time.time() - t_start,
}
with open("per_query_outcomes_random_control.json", "w") as f: json.dump(results, f, indent=1)

# Statistical analysis
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
print(f"\n=========== RANDOM CONTROL (Llama-3.2-3B, ℓ=7, α=2, n={N}) ============")
rng_master = np.random.default_rng(SEED)
rows = []
print(f"{'cond':<22} {'em%':>7} {'95% CI':>16} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*70)
for c in ["baseline", "contrastive", "random_0", "random_1", "random_2", "random_3", "random_4", "random_per_query"]:
    arr = np.asarray(results[c]["correct"])
    mu, lo, hi = boot_ci(arr, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
    if c == "baseline":
        d_str, p_val = "---", None
    else:
        d, p_val = perm(arr, base, rng=np.random.default_rng(rng_master.integers(0, 2**31)))
        d_str = f"{100*d:+.2f}"
    p_str = "---" if p_val is None else f"{p_val:.4f}"
    print(f"{c:<22} {100*mu:>7.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {d_str:>10} {p_str:>10}")
    rows.append({"condition": c, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
                 "delta_pp": None if c == "baseline" else float(100*(arr.mean() - base.mean())),
                 "perm_p": p_val,
                 "lat_mean": float(np.mean(results[c]["lat"]))})

# Aggregate the 5 fixed random dirs
random_ems = [np.asarray(results[f"random_{k}"]["correct"]).mean() for k in range(5)]
print(f"\n  random_dir mean across 5 seeds: {100*np.mean(random_ems):.2f}±{100*np.std(random_ems):.2f}")

with open("statistical_summary_random_control.json", "w") as f:
    json.dump({"meta": results["meta"], "rows": rows,
               "random_dir_mean": float(np.mean(random_ems)),
               "random_dir_std": float(np.std(random_ems))}, f, indent=1)
print(f"\nWrote statistical_summary_random_control.json")
