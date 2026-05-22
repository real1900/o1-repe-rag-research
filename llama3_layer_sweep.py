"""Layer sweep: does steering work at any layer in Llama-3.2-3B?

The α-sweep at the middle layer (14/28) shows monotonic degradation. This
script holds α=2 fixed (the largest 'safe' value from v3) and sweeps across
layer indices to determine whether ANY layer admits beneficial steering.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

ALPHA = 2.0
LAYER_TARGETS = [3, 7, 14, 21, 26]   # early / mid-early / mid / mid-late / late
N_QUERIES = int(os.environ.get("N_QUERIES", 200))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
MAX_NEW_TOKENS = 12

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

mid = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=device)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.eval()
LAYERS = model.model.layers
NUM_LAYERS = len(LAYERS)
print(f"[arch] {mid}: {NUM_LAYERS} layers; α={ALPHA}; sweeping layers {LAYER_TARGETS}")

def extract_unit_vec(text, layer_idx):
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

def run_one(prompt_inputs, input_len, gold, target_layer, hook=None):
    handle = LAYERS[target_layer].register_forward_hook(hook) if hook is not None else None
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**prompt_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    lat = time.time() - t0
    if handle is not None: handle.remove()
    gen = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).lower()
    return int(gold in gen), lat

def build_prompt(distractor, question):
    msgs = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user", "content": f"Context: {distractor}\n\nQuestion: {question}"}
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

with open("hotpot_filtered_5000.json") as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] {len(eval_set)} queries")

# Conditions: baseline + one for each target layer
results = {f"layer_{L}": {"correct": [], "lat": []} for L in LAYER_TARGETS}
results["baseline"] = {"correct": [], "lat": []}
t_start = time.time()

for i, row in enumerate(eval_set):
    distractor, question, gold = row["distractor_context"], row["question"], row["answer"].lower()
    prompt = build_prompt(distractor, question)
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]

    # Baseline (no hook, irrelevant which layer)
    c, l = run_one(pi, in_len, gold, target_layer=14, hook=None)
    results["baseline"]["correct"].append(c); results["baseline"]["lat"].append(l)

    for L in LAYER_TARGETS:
        v_neg = extract_unit_vec(distractor, L)
        v_pos = extract_unit_vec("A standard fact.", L)
        v_contr = v_neg - v_pos
        n = torch.linalg.norm(v_contr)
        unit_vec = v_contr / n if n.item() > 1e-6 else torch.zeros_like(v_contr)
        c, l = run_one(pi, in_len, gold, target_layer=L, hook=make_unit_steered_hook(ALPHA, unit_vec))
        results[f"layer_{L}"]["correct"].append(c); results[f"layer_{L}"]["lat"].append(l)

    if (i+1) % 25 == 0:
        elapsed = time.time() - t_start
        means = " ".join([f"L{L}={np.mean(results[f'layer_{L}']['correct']):.3f}" for L in LAYER_TARGETS])
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s base={np.mean(results['baseline']['correct']):.3f} {means}")

results["meta"] = {
    "model": mid, "n_queries": len(eval_set), "alpha": ALPHA,
    "layer_targets": LAYER_TARGETS, "num_layers": NUM_LAYERS,
    "total_seconds": time.time() - t_start,
}
with open("per_query_outcomes_layer_sweep.json", "w") as f: json.dump(results, f, indent=1)

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
print(f"\n=========== LAYER SWEEP at α={ALPHA} (model={mid}, n={len(eval_set)}) ============")
print(f"{'Condition':<12} {'EM (%)':>8} {'95% CI':>20} {'Δ pp':>10} {'Perm-p':>10}")
print("-"*60)
mu, lo, hi = boot_ci(base)
print(f"{'baseline':<12} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {'---':>10} {'---':>10}")
rows = [{"condition":"baseline","layer":None,"em":float(mu),"ci_lo":float(lo),"ci_hi":float(hi),"delta_pp":None,"perm_p":None,"lat_mean":float(np.mean(results['baseline']['lat']))}]
for L in LAYER_TARGETS:
    arr = np.asarray(results[f"layer_{L}"]["correct"])
    mu, lo, hi = boot_ci(arr)
    d, p = perm(arr, base)
    print(f"{'layer_'+str(L):<12} {100*mu:>8.2f} [{100*lo:>5.2f}, {100*hi:>5.2f}] {100*d:>+10.2f} {p:>10.4f}")
    rows.append({"condition":f"layer_{L}","layer":L,"em":float(mu),"ci_lo":float(lo),"ci_hi":float(hi),"delta_pp":float(100*d),"perm_p":float(p),"lat_mean":float(np.mean(results[f"layer_{L}"]["lat"]))})

with open("statistical_summary_layer_sweep.json", "w") as f:
    json.dump({"meta": results["meta"], "rows": rows}, f, indent=1)
print(f"\nSaved per_query_outcomes_layer_sweep.json + statistical_summary_layer_sweep.json")
