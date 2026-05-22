"""Re-prepare HotpotQA as a realistic RAG retrieved set.

The original hotpot_filtered_5000.json kept only ONE gold paragraph and ONE
distractor per question. HotpotQA is multi-hop: most questions need BOTH gold
paragraphs to be answerable. This script keeps every gold paragraph plus
several distractors per question, so a "clean" (gold-only) condition is a real
ceiling and "distracted" is a realistic mixed retrieved set.

Crucially it preserves the exact same rows in the same order as
hotpot_filtered_5000.json (matched by id), so eval rows 3500-4000 remain the
same questions used by every other experiment in the paper.

Output: hotpot_rag_5000.json with per row:
  id, question, answer, gold_passages (list), distractor_passages (list)
"""
import json
from datasets import load_dataset

N_DISTRACTORS = 4
SRC = "hotpot_filtered_5000.json"
OUT = "hotpot_rag_5000.json"

print(f"[load] existing {SRC} (for row order/identity) ...")
existing = json.load(open(SRC))
print(f"[load] {len(existing)} existing rows")

print("[load] raw HotpotQA (distractor, validation) ...")
ds = load_dataset("hotpot_qa", "distractor", split="validation")
by_id = {row["id"]: row for row in ds}
print(f"[load] indexed {len(by_id)} raw HotpotQA rows by id")

out, n_fallback = [], 0
for old in existing:
    rid = old["id"]
    row = by_id.get(rid)
    if row is None:
        n_fallback += 1
        out.append({"id": rid, "question": old["question"], "answer": old["answer"],
                    "gold_passages": [old["true_context"]],
                    "distractor_passages": [old["distractor_context"]]})
        continue
    support = set(row["supporting_facts"]["title"])
    gold, dist = [], []
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        para = " ".join(sents)
        (gold if title in support else dist).append(para)
    out.append({"id": rid, "question": old["question"], "answer": old["answer"],
                "gold_passages": gold if gold else [old["true_context"]],
                "distractor_passages": dist[:N_DISTRACTORS] if dist else [old["distractor_context"]]})

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

avg_g = sum(len(r["gold_passages"]) for r in out) / len(out)
avg_d = sum(len(r["distractor_passages"]) for r in out) / len(out)
print(f"[done] saved {len(out)} rows to {OUT}  (fallback rows: {n_fallback})")
print(f"[done] avg gold passages/row = {avg_g:.2f}, avg distractor passages/row = {avg_d:.2f}")
