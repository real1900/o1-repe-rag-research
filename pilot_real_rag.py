"""Path-B pilot: does a properly-built distractor-steering method recover
accuracy in a real-RAG setup (gold + distractor passages mixed)?

Per query, four PAIRED conditions (same query set, identical prompts):
  clean         - gold passage only (accuracy ceiling)
  distracted    - gold + distractor, no steering (realistic RAG baseline)
  caa_ablate    - distracted, with the CAA distractor-direction projected out
  random_ablate - distracted, with a random unit direction projected out (control)

The CAA direction is the normalized mean(distractor) - mean(gold) activation
difference over N_CAA_PAIRS held-out rows, disjoint from the eval set. Ablation
is scale-free: it removes whatever component of the residual lies along the
direction, so there is no steering coefficient to tune.

Decision gate: caa_ablate should move `distracted` toward `clean` AND beat
`random_ablate` by a real, significant margin.

Env overrides: MODEL, TARGET_LAYER, N_EVAL, EVAL_START, N_CAA_PAIRS, CAA_START.
Outputs: per_query_outcomes_pilot.json, statistical_summary_pilot.json
"""
import os, json, time, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL        = os.environ.get("MODEL", "meta-llama/Llama-3.2-3B-Instruct")
TARGET_LAYER = int(os.environ.get("TARGET_LAYER", 7))
N_EVAL       = int(os.environ.get("N_EVAL", 500))
EVAL_START   = int(os.environ.get("EVAL_START", 3500))
N_CAA_PAIRS  = int(os.environ.get("N_CAA_PAIRS", 400))
CAA_START    = int(os.environ.get("CAA_START", 0))
HOTPOT       = "hotpot_rag_5000.json"
SEED         = 2026
MAX_NEW      = 12

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_grad_enabled(False)
torch.manual_seed(SEED)

print(f"[load] {MODEL} on {device}")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.eval()
LAYERS = model.model.layers
D_MODEL = model.config.hidden_size
print(f"[arch] hidden={D_MODEL}, layers={len(LAYERS)}, target_layer={TARGET_LAYER}")


def mean_act(text):
    """Mean-pooled hidden state at TARGET_LAYER for `text` (float32, shape [d])."""
    inp = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    cache = {}
    def hook(m, i, o):
        cache["v"] = (o[0] if isinstance(o, tuple) else o).detach()
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook)
    try: _ = model(**inp)
    finally: handle.remove()
    return cache["v"].mean(dim=1).squeeze(0).float()


def ablate_hook(unit):
    """Project the `unit` component out of the last token's residual stream."""
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        hn = h.clone()
        last = h[:, -1, :].float()                       # [batch, d]
        coef = last @ unit                               # [batch]
        hn[:, -1, :] = (last - coef.unsqueeze(-1) * unit).to(h.dtype)
        return (hn,) + o[1:] if isinstance(o, tuple) else hn
    return hook


def build_prompt(passages, question):
    ctx = "\n\n".join(passages)
    msgs = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user", "content": f"Context: {ctx}\n\nQuestion: {question}"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run(prompt, gold, hook=None):
    pi = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]
    handle = LAYERS[TARGET_LAYER].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    txt = tok.decode(out[0][in_len:], skip_special_tokens=True).lower()
    return int(gold in txt), lat


# ---- data ----
with open(HOTPOT) as f: data = json.load(f)
caa_rows  = data[CAA_START:CAA_START + N_CAA_PAIRS]
eval_rows = data[EVAL_START:EVAL_START + N_EVAL]
print(f"[data] CAA pairs: rows {CAA_START}-{CAA_START+len(caa_rows)} (n={len(caa_rows)})")
print(f"[data] eval:      rows {EVAL_START}-{EVAL_START+len(eval_rows)} (n={len(eval_rows)})")

# ---- build the CAA direction from held-out pairs ----
print("[caa] building CAA direction: normalized mean(distractor) - mean(gold) ...")
dis_acts, gold_acts = [], []
for j, r in enumerate(caa_rows):
    dis_acts.append(mean_act(r["distractor_passages"][0]))
    gold_acts.append(mean_act(r["gold_passages"][0]))
    if (j + 1) % 100 == 0: print(f"  [caa] {j+1}/{len(caa_rows)}")
caa_dir = torch.stack(dis_acts).mean(0) - torch.stack(gold_acts).mean(0)
caa_dir = caa_dir / torch.linalg.norm(caa_dir)
print(f"[caa] direction built (unit-norm, d={caa_dir.shape[0]})")

# ---- a random unit direction (control) ----
gen = torch.Generator().manual_seed(SEED)
rand_dir = torch.randn(D_MODEL, generator=gen).to(device).float()
rand_dir = rand_dir / torch.linalg.norm(rand_dir)

# ---- eval ----
CONDS = ["clean", "distracted", "caa_ablate", "random_ablate"]
res = {c: {"correct": [], "lat": []} for c in CONDS}
order_rng = random.Random(SEED)

print(f"[eval] {len(eval_rows)} queries x {len(CONDS)} conditions ...")
t0 = time.time()
for i, r in enumerate(eval_rows):
    gold = r["answer"].lower()
    gold_ps, dist_ps = r["gold_passages"], r["distractor_passages"]
    mixed = gold_ps + dist_ps
    order_rng.shuffle(mixed)                         # randomize passage order

    p_clean = build_prompt(gold_ps, r["question"])
    p_dist  = build_prompt(mixed, r["question"])

    for cond, prompt, hook in [
        ("clean",         p_clean, None),
        ("distracted",    p_dist,  None),
        ("caa_ablate",    p_dist,  ablate_hook(caa_dir)),
        ("random_ablate", p_dist,  ablate_hook(rand_dir)),
    ]:
        c, l = run(prompt, gold, hook)
        res[cond]["correct"].append(c); res[cond]["lat"].append(l)

    if (i + 1) % 25 == 0:
        el = time.time() - t0
        line = "  ".join(f"{c}={np.mean(res[c]['correct']):.3f}" for c in CONDS)
        print(f"  [{i+1}/{len(eval_rows)}] {el:.0f}s  {line}")

meta = {"model": MODEL, "target_layer": TARGET_LAYER, "n_eval": len(eval_rows),
        "eval_start": EVAL_START, "n_caa_pairs": len(caa_rows), "caa_start": CAA_START,
        "device": device, "seed": SEED, "total_seconds": time.time() - t0}
with open("per_query_outcomes_pilot.json", "w") as f:
    json.dump({"meta": meta, **{c: res[c] for c in CONDS}}, f, indent=1)


# ---- stats ----
def boot_ci(arr, n=10_000, rng=None):
    rng = rng or np.random.default_rng(42); arr = np.asarray(arr); m = np.empty(n)
    for k in range(n): m[k] = arr[rng.integers(0, len(arr), len(arr))].mean()
    return arr.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def perm(a, b, n=10_000, rng=None):
    rng = rng or np.random.default_rng(7); a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean(); ext = 0
    for _ in range(n):
        f = rng.integers(0, 2, len(a)).astype(bool)
        if abs(np.where(f, b, a).mean() - np.where(f, a, b).mean()) >= abs(d) - 1e-12: ext += 1
    return d, ext / n

dist  = np.asarray(res["distracted"]["correct"])
clean = np.asarray(res["clean"]["correct"])
print("\n" + "=" * 70)
print(f"  PATH-B PILOT  ({MODEL}, layer {TARGET_LAYER}, n={len(eval_rows)})")
print("=" * 70)
print(f"  {'condition':<16}{'EM%':>8}{'95% CI':>18}{'vs distracted':>22}")
rows = []
for c in CONDS:
    arr = np.asarray(res[c]["correct"])
    mu, lo, hi = boot_ci(arr)
    if c == "distracted":
        delta_str, p = "---", None
    else:
        d, p = perm(arr, dist)
        delta_str = f"{100*d:+.2f}pp  p={p:.4f}"
    print(f"  {c:<16}{100*mu:>8.2f}   [{100*lo:>5.2f},{100*hi:>6.2f}]{delta_str:>22}")
    rows.append({"condition": c, "em": float(mu), "ci_lo": float(lo), "ci_hi": float(hi),
                 "delta_vs_distracted_pp": None if p is None else float(100 * (arr.mean() - dist.mean())),
                 "perm_p_vs_distracted": p,
                 "lat_mean": float(np.mean(res[c]["lat"]))})

gap   = clean.mean() - dist.mean()
recov = np.asarray(res["caa_ablate"]["correct"]).mean() - dist.mean()
print(f"\n  distractor damage (clean - distracted): {100*gap:+.2f} pp")
if gap > 1e-9:
    print(f"  caa_ablate recovers: {100*recov:+.2f} pp  ({100*recov/(100*gap):.0f}% of the damage)")
print("=" * 70)

with open("statistical_summary_pilot.json", "w") as f:
    json.dump({"meta": meta, "rows": rows,
               "distractor_damage_pp": float(100 * gap),
               "caa_ablate_recovery_pp": float(100 * recov)}, f, indent=1)
print("\nSaved per_query_outcomes_pilot.json + statistical_summary_pilot.json")
