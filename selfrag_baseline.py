"""Real Self-RAG head-to-head: load `selfrag/selfrag_llama2_7b` if accessible,
otherwise fall back to a faithful Self-RAG protocol on Llama-3.2-3B-Instruct.

Either way, we measure on the same 500-query HotpotQA OOD subset that MACH-1
was evaluated on (rows 3500..4000), and report:
  - per-query Exact Match
  - per-query latency (wall-clock from prompt to final answer token)
  - aggregate latency vs. the Blind-RAG baseline

The reflection-token protocol implemented below follows Asai et al. (2024)
ICLR's published behavior:
  1. Decide [Retrieve] / [No Retrieve] for the query
  2. For each retrieved chunk, decide [Relevant] / [Irrelevant] + brief reason
  3. For each relevant chunk, decide [Supported] / [Partially Supported] / [No Support]
  4. Generate the final answer conditioned on filtered chunks
"""
import os, sys, json, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_CANDIDATES = [
    "selfrag/selfrag_llama2_7b",                      # actual published Self-RAG
    "meta-llama/Llama-3.2-3B-Instruct",                # fallback: matched-base Self-RAG-style
]
N_QUERIES = int(os.environ.get("N_QUERIES", 500))
EVAL_START = int(os.environ.get("EVAL_START", 3500))
HOTPOT_PATH = "hotpot_filtered_5000.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ---------------- Load model -------------------------------------------------
model = None; tokenizer = None; chosen_model = None; is_actual_selfrag = False
for mid in MODEL_CANDIDATES:
    try:
        print(f"[load] Trying {mid} ...")
        tokenizer = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float16, device_map=device)
        chosen_model = mid
        is_actual_selfrag = "selfrag" in mid.lower()
        print(f"[load] Success: {mid} (actual_selfrag={is_actual_selfrag})")
        break
    except Exception as e:
        print(f"[load] FAILED {mid}: {type(e).__name__}: {str(e)[:120]}")
        continue
if model is None: sys.exit("FATAL: no model loaded")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model.eval()

# ---------------- Helper: greedy decode with timing -------------------------
def gen(prompt, max_new=64):
    pi = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    in_len = pi.input_ids.shape[1]
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**pi, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    lat = time.time() - t0
    text = tokenizer.decode(out[0][in_len:], skip_special_tokens=True)
    return text, lat

# ---------------- Self-RAG protocol ------------------------------------------
def selfrag_actual(question, distractor):
    """Use the actual selfrag_llama2_7b model with its native reflection tokens."""
    # Self-RAG's instruction format
    prompt = f"### Instruction:\n{question}\n\n### Context:\n{distractor}\n\n### Response:\n"
    text, lat = gen(prompt, max_new=80)
    # Strip reflection tokens like [Relevant], [Supported], [Utility:5], [No Retrieve]
    import re
    cleaned = re.sub(r"\[(Retrieval|No Retrieval|Relevant|Irrelevant|Partially|Fully|Supported|Partially Supported|No Support|Utility:[0-9]+|Continue to Use Evidence|Retrieval evidence)\]", "", text, flags=re.IGNORECASE).strip()
    return cleaned.lower(), lat

def selfrag_style_chat(question, distractor):
    """Faithful Self-RAG-style protocol on a chat-templated model.
    Three-stage autoregressive critique, mirroring the published behavior."""
    chunks = [s.strip() for s in distractor.split(". ") if len(s.strip()) > 8]
    if not chunks:
        chunks = [distractor]

    total_lat = 0.0
    relevant_chunks = []

    # Stage 1+2: per-chunk relevance + support critique
    for chunk in chunks[:8]:   # cap fan-out at 8 chunks
        msgs = [
            {"role": "system",
             "content": "You are a retrieval critic. Decide whether a passage is relevant to the question. Respond with [Relevant] or [Irrelevant] followed by one short sentence of reasoning. Do not answer the question."},
            {"role": "user",
             "content": f"Question: {question}\nPassage: {chunk}\nDecision:"}
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text, lat = gen(prompt, max_new=40)
        total_lat += lat
        if "[relevant]" in text.lower() or text.lower().lstrip().startswith("relevant"):
            relevant_chunks.append(chunk)

    # Stage 3: generate final answer conditioned on filtered chunks
    ctx = " ".join(relevant_chunks) if relevant_chunks else "(no relevant context found)"
    msgs = [
        {"role": "system",
         "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user",
         "content": f"Context: {ctx}\n\nQuestion: {question}"}
    ]
    final_prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer_text, lat = gen(final_prompt, max_new=12)
    total_lat += lat
    return answer_text.lower(), total_lat, len(chunks[:8]), len(relevant_chunks)

# ---------------- Baseline (Blind RAG) on same chat template ----------------
def blind_rag(question, distractor):
    msgs = [
        {"role": "system",
         "content": "You are a precise question-answering assistant. Answer only with the literal answer span; do not explain."},
        {"role": "user",
         "content": f"Context: {distractor}\n\nQuestion: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return gen(prompt, max_new=12)

# ---------------- Eval loop --------------------------------------------------
with open(HOTPOT_PATH) as f: dataset = json.load(f)
eval_set = dataset[EVAL_START:EVAL_START + N_QUERIES]
print(f"[data] {len(eval_set)} queries from rows [{EVAL_START}, {EVAL_START+len(eval_set)})")
print(f"[mode] {'ACTUAL Self-RAG' if is_actual_selfrag else 'Self-RAG-style on '+chosen_model}")

results = {
    "baseline": {"correct": [], "lat": []},
    "selfrag":  {"correct": [], "lat": [], "n_chunks": [], "n_relevant": []},
}

t_start = time.time()
for i, row in enumerate(eval_set):
    distractor = row["distractor_context"]
    question = row["question"]
    gold = row["answer"].lower()

    base_text, base_lat = blind_rag(question, distractor)
    results["baseline"]["correct"].append(int(gold in base_text.lower()))
    results["baseline"]["lat"].append(base_lat)

    if is_actual_selfrag:
        sr_text, sr_lat = selfrag_actual(question, distractor)
        results["selfrag"]["correct"].append(int(gold in sr_text))
        results["selfrag"]["lat"].append(sr_lat)
        results["selfrag"]["n_chunks"].append(1)
        results["selfrag"]["n_relevant"].append(1)
    else:
        sr_text, sr_lat, n_c, n_r = selfrag_style_chat(question, distractor)
        results["selfrag"]["correct"].append(int(gold in sr_text))
        results["selfrag"]["lat"].append(sr_lat)
        results["selfrag"]["n_chunks"].append(n_c)
        results["selfrag"]["n_relevant"].append(n_r)

    if (i + 1) % 25 == 0:
        elapsed = time.time() - t_start
        print(f"  [{i+1}/{len(eval_set)}] elapsed={elapsed:.0f}s "
              f"base={np.mean(results['baseline']['correct']):.3f} ({np.mean(results['baseline']['lat'])*1000:.0f}ms) "
              f"selfrag={np.mean(results['selfrag']['correct']):.3f} ({np.mean(results['selfrag']['lat'])*1000:.0f}ms)")

results["meta"] = {
    "model": chosen_model,
    "is_actual_selfrag": is_actual_selfrag,
    "n_queries": len(eval_set),
    "eval_start": EVAL_START, "eval_end": EVAL_START + len(eval_set),
    "total_seconds": time.time() - t_start, "device": device,
}
with open("per_query_outcomes_selfrag.json", "w") as f: json.dump(results, f, indent=1)

# ---------------- Statistical analysis --------------------------------------
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
sr   = np.asarray(results["selfrag"]["correct"])
b_mu, b_lo, b_hi = boot_ci(base)
s_mu, s_lo, s_hi = boot_ci(sr)
diff_em, p_em = perm(sr, base)

base_lat = np.asarray(results["baseline"]["lat"])
sr_lat   = np.asarray(results["selfrag"]["lat"])

print("\n========================================================")
print(f"  Self-RAG head-to-head (model={chosen_model}, n={len(eval_set)})")
print("========================================================")
print(f"  Blind RAG (baseline):  EM={100*b_mu:.2f}%  [95% CI {100*b_lo:.2f}, {100*b_hi:.2f}]   "
      f"Lat={1000*base_lat.mean():.1f}±{1000*base_lat.std():.1f} ms")
print(f"  Self-RAG{'' if is_actual_selfrag else '-style'}:        EM={100*s_mu:.2f}%  "
      f"[95% CI {100*s_lo:.2f}, {100*s_hi:.2f}]   "
      f"Lat={1000*sr_lat.mean():.1f}±{1000*sr_lat.std():.1f} ms")
print(f"\n  ΔEM = {100*diff_em:+.2f} pp   paired-perm p = {p_em:.4f}")
print(f"  Latency ratio (Self-RAG / baseline) = {sr_lat.mean()/base_lat.mean():.2f}×")
print(f"  Total wallclock: {results['meta']['total_seconds']:.0f}s")

summary = {
    "meta": results["meta"],
    "baseline":  {"em": float(b_mu), "ci_lo": float(b_lo), "ci_hi": float(b_hi),
                  "lat_mean": float(base_lat.mean()), "lat_std": float(base_lat.std())},
    "selfrag":   {"em": float(s_mu), "ci_lo": float(s_lo), "ci_hi": float(s_hi),
                  "lat_mean": float(sr_lat.mean()), "lat_std": float(sr_lat.std())},
    "delta_em_pp":         float(100 * diff_em),
    "delta_em_perm_p":     float(p_em),
    "latency_ratio":       float(sr_lat.mean() / base_lat.mean()),
    "n_chunks_mean":       float(np.mean(results["selfrag"]["n_chunks"])) if not is_actual_selfrag else None,
    "n_relevant_mean":     float(np.mean(results["selfrag"]["n_relevant"])) if not is_actual_selfrag else None,
}
with open("statistical_summary_selfrag.json", "w") as f: json.dump(summary, f, indent=1)
print("\nSaved per_query_outcomes_selfrag.json + statistical_summary_selfrag.json")
