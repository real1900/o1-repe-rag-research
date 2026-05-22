"""Prepare TriviaQA-RC and 2WikiMultihopQA into the same format as
hotpot_filtered_5000.json so the existing eval scripts work unchanged.

Schema per row:
  {"id": str, "question": str, "answer": str, "distractor_context": str}

For TriviaQA: pull the rc.nocontext (retrieval-style) split, attach a
distractor passage from the same example's search_results list (one that does
NOT contain the answer span).

For 2WikiMultihopQA: each example has supporting_facts + 10 paragraphs; we
concat the paragraphs that are NOT in the supporting facts list as the
distractor context.

Output:
  triviaqa_filtered_5000.json
  twiki_filtered_5000.json
"""
import json, os, random, sys
from datasets import load_dataset

random.seed(2026)

# ----------------------------------------------------------------------------
# 1. TriviaQA-RC
# ----------------------------------------------------------------------------
print("[triviaqa] loading rc.nocontext (validation split)...")
try:
    triv = load_dataset("trivia_qa", "rc.nocontext", split="validation", trust_remote_code=True)
except Exception as e:
    print(f"  rc.nocontext failed: {e}")
    triv = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)
    print(f"  fell back to 'rc' split instead")

print(f"[triviaqa] loaded {len(triv)} rows; columns={triv.column_names}")
print(f"[triviaqa] sample row:")
sample = triv[0]
for k, v in sample.items():
    s = str(v)
    print(f"   {k}: {s[:160]}{'...' if len(s) > 160 else ''}")

triv_rows = []
for i, row in enumerate(triv):
    if len(triv_rows) >= 5000: break
    answer_aliases = [a.lower() for a in row["answer"].get("aliases", [])] + [row["answer"]["value"].lower()]
    answer_main = row["answer"]["value"]
    if not answer_main.strip(): continue

    # Find or synthesize a distractor passage. The 'rc' config provides
    # entity_pages and search_results lists. We use search_results.
    distractor_text = None
    sr = row.get("search_results", {})
    if isinstance(sr, dict) and sr.get("search_context"):
        # Pick the first search-context paragraph that does NOT contain any answer alias
        for sc in sr["search_context"]:
            sc_l = sc.lower()
            if not any(a in sc_l for a in answer_aliases if a):
                distractor_text = sc[:1200]
                break
        if distractor_text is None and sr["search_context"]:
            # fall back: just use first one (we'll have answer leak; honest measurement)
            distractor_text = sr["search_context"][0][:1200]
    elif isinstance(sr, dict) and sr.get("description"):
        distractor_text = " ".join(sr["description"][:3])[:1200]

    if not distractor_text or len(distractor_text) < 30:
        # synthesize a minimal distractor: pick a random other example's question text
        other = triv[random.randrange(len(triv))]
        distractor_text = other["question"]
    triv_rows.append({
        "id": str(row["question_id"]),
        "question": row["question"],
        "answer": answer_main,
        "distractor_context": distractor_text,
    })

with open("triviaqa_filtered_5000.json", "w") as f:
    json.dump(triv_rows, f)
print(f"[triviaqa] wrote {len(triv_rows)} rows -> triviaqa_filtered_5000.json")
print(f"[triviaqa] sample formatted: {json.dumps(triv_rows[0], indent=1)[:500]}")

# ----------------------------------------------------------------------------
# 2. 2WikiMultihopQA
# ----------------------------------------------------------------------------
print("\n[2wiki] loading 2WikiMultihopQA validation split...")
two_wiki = None
for repo in ["voidful/2WikiMultihopQA", "deepmind/2wikimultihopqa", "Salesforce/2WikiMultihopQA"]:
    try:
        two_wiki = load_dataset(repo, split="validation", trust_remote_code=True)
        print(f"  loaded from {repo}: {len(two_wiki)} rows")
        break
    except Exception as e:
        print(f"  {repo} failed: {type(e).__name__}: {str(e)[:100]}")

if two_wiki is None:
    print("[2wiki] FAILED to load 2WikiMultihop -- skipping (will note in paper)")
else:
    print(f"[2wiki] columns: {two_wiki.column_names}")
    sample = two_wiki[0]
    for k, v in sample.items():
        s = str(v)
        print(f"   {k}: {s[:200]}{'...' if len(s) > 200 else ''}")

    twiki_rows = []
    for i, row in enumerate(two_wiki):
        if len(twiki_rows) >= 5000: break
        question = row["question"]
        answer = row["answer"]
        # context format: list of (title, sentence_list) pairs
        # supporting_facts: list of (title, sent_idx) pairs
        ctx = row.get("context", {})
        if isinstance(ctx, dict):  # HF format with parallel lists
            titles = ctx.get("title", [])
            sentences = ctx.get("content", []) or ctx.get("sentences", [])
            paragraphs = list(zip(titles, sentences)) if titles and sentences else []
        elif isinstance(ctx, list):
            paragraphs = ctx
        else:
            paragraphs = []
        sf = row.get("supporting_facts", {})
        if isinstance(sf, dict):
            sf_titles = set(sf.get("title", []))
        else:
            sf_titles = set([s[0] for s in sf]) if sf else set()

        # distractor = concat of paragraphs whose title is NOT in supporting_facts
        distractor_paras = [p for p in paragraphs if p[0] not in sf_titles]
        if not distractor_paras:
            continue
        distractor_text = " ".join(
            (" ".join(s) if isinstance(s, list) else s)
            for _, s in distractor_paras[:3]
        )[:1500]
        if not distractor_text.strip():
            continue

        twiki_rows.append({
            "id": row.get("_id", row.get("id", str(i))),
            "question": question,
            "answer": str(answer),
            "distractor_context": distractor_text,
        })

    with open("twiki_filtered_5000.json", "w") as f:
        json.dump(twiki_rows, f)
    print(f"[2wiki] wrote {len(twiki_rows)} rows -> twiki_filtered_5000.json")
    if twiki_rows:
        print(f"[2wiki] sample: {json.dumps(twiki_rows[0], indent=1)[:500]}")
