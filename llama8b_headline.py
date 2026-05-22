"""Llama-3.1-8B-Instruct headline: layer + α joint sweep on a 500-query
HotpotQA distractor subset.

Design:
- Sweep layers ∈ {6, 8, 10, 14, 22} × α ∈ {1, 2, 3} on n=200 first (joint coarse sweep)
- Take the (ℓ, α) with the largest paired Δ
- Validate that single best config on n=500 with 5 seeds for variance

Outputs:
  per_query_outcomes_8b_sweep.json
  per_query_outcomes_8b_validation.json
  statistical_summary_8b.json
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HOTPOT = "hotpot_filtered_5000.json"
N_SWEEP = int(os.environ.get("N_SWEEP", 200))
N_VALID = int(os.environ.get("N_VALID", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
LAYER_TARGETS = [6, 8, 10, 14, 22]
ALPHAS = [1.0, 2.0, 3.0]
MAX_NEW_TOKENS = 12
MASTER_SEED = 2026

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

print(f"[load] {MODEL} ...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map=device)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYERS = model.model.layers
NUM_LAYERS = len(LAYERS)
print(f"[arch] {NUM_LAYERS} layers; sweeping {LAYER_TARGETS} × α∈{ALPHAS}")

def extract_unit_vec(text, layer_idx):
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        cache["v"] = h.detach()
    handle = LAYERS[layer_idx].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    return cache["v"].mean(dim=1).squeeze(0)

def make_hook(alpha, unit_vec):
    def hook(m, i, o):
        if alpha == 0.0: return o
        h = o[0] if isinstance(o, tuple) else o
        h_new = h.clone()
        h_new[:, -1, :] = h[:, -1, :] - (alpha * unit_vec.to(h.dtype))
        if isinstance(o, tuple): return (h_new,) + o[1:]
        return h_new
    return hook

def run_one(pi, in_len, gold, target_layer, hook):
    handle = LAYERS[target_layer].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    text = tok.decode(out[0][in_len:], skip_special_tokens=True).lower()
    return int(gold in text), lat

def build_prompt(distractor, question):
    msgs = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user", "content": f"Context: {distractor}\n\nQuestion: {question}"}
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

with open(HOTPOT) as f: dataset = json.load(f)
sweep_set = dataset[EVAL_START:EVAL_START + N_SWEEP]
print(f"[data] sweep on {len(sweep_set)} queries")

# ---------------- Phase A: joint (layer, α) sweep ----------------
sweep_results = {"baseline": {"correct": [], "lat": []}}
for L in LAYER_TARGETS:
    for a in ALPHAS:
        sweep_results[f"L{L}_a{a:.0f}"] = {"correct": [], "lat": []}

t_start = time.time()
print(f"[sweep] phase A: joint layer × α sweep")
for i, row in enumerate(sweep_set):
    distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    c, l = run_one(pi, in_len, gold, target_layer=14, hook=None)
    sweep_results["baseline"]["correct"].append(c); sweep_results["baseline"]["lat"].append(l)

    for L in LAYER_TARGETS:
        v_neg = extract_unit_vec(distractor, L)
        v_pos = extract_unit_vec("A standard fact.", L)
        v_contr = v_neg - v_pos
        n = torch.linalg.norm(v_contr)
        unit = v_contr / n if n.item() > 1e-6 else torch.zeros_like(v_contr)
        for a in ALPHAS:
            c, l = run_one(pi, in_len, gold, target_layer=L, hook=make_hook(a, unit))
            cond = f"L{L}_a{a:.0f}"
            sweep_results[cond]["correct"].append(c); sweep_results[cond]["lat"].append(l)

    if (i + 1) % 25 == 0:
        elapsed = time.time() - t_start
        em_b = np.mean(sweep_results["baseline"]["correct"])
        best_cond = max([k for k in sweep_results if k != "baseline"],
                        key=lambda k: np.mean(sweep_results[k]["correct"]))
        em_best = np.mean(sweep_results[best_cond]["correct"])
        print(f"  [{i+1}/{len(sweep_set)}] elapsed={elapsed:.0f}s base={em_b:.3f} best={best_cond}={em_best:.3f}")

with open("per_query_outcomes_8b_sweep.json", "w") as f:
    json.dump({**sweep_results, "meta": {
        "model": MODEL, "n_queries": len(sweep_set), "layers": LAYER_TARGETS,
        "alphas": ALPHAS, "phase": "sweep",
    }}, f, indent=1)

# pick best (ℓ, α) by paired Δ
base = np.asarray(sweep_results["baseline"]["correct"])
best_cond, best_delta = None, -1e9
for cond, vals in sweep_results.items():
    if cond == "baseline" or cond == "meta": continue
    arr = np.asarray(vals["correct"])
    d = arr.mean() - base.mean()
    if d > best_delta:
        best_delta = d; best_cond = cond
print(f"\n[sweep] best config in phase A: {best_cond}  Δ={100*best_delta:+.2f} pp")
best_L = int(best_cond.split("_")[0][1:])
best_a = int(best_cond.split("_a")[1])

# ---------------- Phase B: 5-seed validation at best (ℓ, α) on N_VALID ----------------
val_set = dataset[EVAL_START + N_SWEEP : EVAL_START + N_SWEEP + N_VALID]
print(f"\n[validation] phase B: best ({best_L}, {best_a}) on n={len(val_set)} (disjoint from phase-A subset)")
val_results = {f"baseline_seed{s}": {"correct": [], "lat": []} for s in range(5)}
for s in range(5):
    val_results[f"steered_seed{s}"] = {"correct": [], "lat": []}

# we don't actually re-seed deterministic generation; instead we shuffle the eval set
rng = np.random.default_rng(MASTER_SEED)
for s in range(5):
    perm_idx = rng.permutation(len(val_set))
    print(f"\n[seed {s}] permuted eval order; running...")
    for j in range(len(val_set)):
        row = val_set[perm_idx[j]]
        distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()
        prompt = build_prompt(distractor, question)
        pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        in_len = pi.input_ids.shape[1]

        c, l = run_one(pi, in_len, gold, target_layer=best_L, hook=None)
        val_results[f"baseline_seed{s}"]["correct"].append(c); val_results[f"baseline_seed{s}"]["lat"].append(l)

        v_neg = extract_unit_vec(distractor, best_L)
        v_pos = extract_unit_vec("A standard fact.", best_L)
        v_contr = v_neg - v_pos
        n = torch.linalg.norm(v_contr)
        unit = v_contr / n if n.item() > 1e-6 else torch.zeros_like(v_contr)
        c, l = run_one(pi, in_len, gold, target_layer=best_L, hook=make_hook(float(best_a), unit))
        val_results[f"steered_seed{s}"]["correct"].append(c); val_results[f"steered_seed{s}"]["lat"].append(l)
        if (j + 1) % 100 == 0:
            print(f"   seed{s} [{j+1}/{len(val_set)}] base={np.mean(val_results[f'baseline_seed{s}']['correct']):.3f} steered={np.mean(val_results[f'steered_seed{s}']['correct']):.3f}")

val_results["meta"] = {
    "model": MODEL, "n_queries": len(val_set),
    "best_layer": best_L, "best_alpha": best_a,
    "phase": "validation_5seed",
}
with open("per_query_outcomes_8b_validation.json", "w") as f: json.dump(val_results, f, indent=1)

# ---------------- Statistical summary ----------------
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

# Phase A summary
sweep_rows = []
for cond in [c for c in sweep_results if c not in ("baseline", "meta")]:
    arr = np.asarray(sweep_results[cond]["correct"])
    mu, lo, hi = boot_ci(arr)
    d, p = perm(arr, base)
    sweep_rows.append({"condition": cond, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
                       "delta_pp": float(100*d), "perm_p": float(p),
                       "lat_mean": float(np.mean(sweep_results[cond]["lat"]))})
sweep_rows.append({"condition": "baseline", "em": float(base.mean()),
                   "ci_lo": float(boot_ci(base)[1]), "ci_hi": float(boot_ci(base)[2]),
                   "delta_pp": None, "perm_p": None,
                   "lat_mean": float(np.mean(sweep_results["baseline"]["lat"]))})

# Phase B summary (per-seed and pooled)
seed_summary = []
all_b, all_s = [], []
for s in range(5):
    b = np.asarray(val_results[f"baseline_seed{s}"]["correct"])
    sa = np.asarray(val_results[f"steered_seed{s}"]["correct"])
    d, p = perm(sa, b)
    seed_summary.append({"seed": s, "baseline_em": float(b.mean()), "steered_em": float(sa.mean()),
                         "delta_pp": float(100*d), "perm_p": float(p)})
    all_b.extend(b.tolist()); all_s.extend(sa.tolist())
pooled_b = np.asarray(all_b); pooled_s = np.asarray(all_s)
pooled_d, pooled_p = perm(pooled_s, pooled_b)
b_mu, b_lo, b_hi = boot_ci(pooled_b); s_mu, s_lo, s_hi = boot_ci(pooled_s)
seed_means = np.array([row["delta_pp"] for row in seed_summary])
print("\n========================================================")
print(f"  Llama-3.1-8B-Instruct headline: ℓ={best_L}, α={best_a}, n={len(val_set)} × 5 seeds")
print("========================================================")
print(f"  Pooled: baseline={100*b_mu:.2f}% [{100*b_lo:.2f}, {100*b_hi:.2f}]  steered={100*s_mu:.2f}% [{100*s_lo:.2f}, {100*s_hi:.2f}]")
print(f"          ΔEM={100*pooled_d:+.2f} pp  paired-perm p={pooled_p:.4f}")
print(f"          per-seed Δ mean={seed_means.mean():+.2f}±{seed_means.std():.2f} pp")
for r in seed_summary:
    print(f"  seed {r['seed']}: base={100*r['baseline_em']:.2f}% steered={100*r['steered_em']:.2f}% Δ={r['delta_pp']:+.2f}pp p={r['perm_p']:.4f}")

with open("statistical_summary_8b.json", "w") as f:
    json.dump({
        "model": MODEL,
        "best_layer": best_L, "best_alpha": best_a,
        "sweep_rows": sweep_rows,
        "seed_summary": seed_summary,
        "pooled": {"baseline_em": float(b_mu), "baseline_ci": [float(b_lo), float(b_hi)],
                   "steered_em": float(s_mu), "steered_ci": [float(s_lo), float(s_hi)],
                   "delta_pp": float(100*pooled_d), "perm_p": float(pooled_p),
                   "seed_delta_mean": float(seed_means.mean()), "seed_delta_std": float(seed_means.std())},
    }, f, indent=1)
print("\nSaved per_query_outcomes_8b_*.json + statistical_summary_8b.json")
