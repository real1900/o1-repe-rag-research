"""Improve TriviaQA distractors: use a synthetic 'general-knowledge' passage
constructed from 5-7 unrelated factoid Q&A pairs. This produces richer
distractors that more closely resemble retrieved-passage failures."""
import json, random
random.seed(2026)

with open("triviaqa_filtered_5000.json") as f:
    triv = json.load(f)
print(f"loaded {len(triv)} triviaqa rows")

# Pre-compute: for each row, build a list of (question, answer) tuples that
# don't share answer aliases with current row, to use as distractor content.
all_questions = [(r["question"], r["answer"]) for r in triv]

improved = []
for i, row in enumerate(triv):
    answer_low = row["answer"].lower()

    # Pick 6 other QA pairs whose answer is NOT a substring of our answer
    distractors = []
    attempts = 0
    while len(distractors) < 6 and attempts < 30:
        j = random.randrange(len(all_questions))
        if j == i:
            attempts += 1; continue
        q, a = all_questions[j]
        if a.lower() in answer_low or answer_low in a.lower():
            attempts += 1; continue
        distractors.append((q, a))
        attempts += 1

    # Build a paragraph-like distractor passage
    # Format as a "general knowledge" wiki-like blurb
    parts = []
    for q, a in distractors:
        # Convert "Who was the man behind The Chipmunks?" + "David Seville" ->
        # "The man behind The Chipmunks is David Seville."
        parts.append(f"{q.rstrip('?').rstrip('.')} - {a}.")
    distractor_para = "Some general trivia: " + " ".join(parts)
    distractor_para = distractor_para[:1200]

    improved.append({
        "id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "distractor_context": distractor_para,
    })

with open("triviaqa_filtered_5000.json", "w") as f:
    json.dump(improved, f)
print(f"wrote improved {len(improved)} rows")
print("sample:")
print(json.dumps(improved[0], indent=1)[:600])
