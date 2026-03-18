import json

with open("hotpot_filtered_5000.json", 'r', encoding='utf-8') as f:
    dataset = json.load(f)

for i in range(3):
    row = dataset[-50:][i]
    print(f"\n[Query {i+1}] {row['question']}")
    print(f"Context: {row['distractor_context']}")
    print(f"Ground Truth: {row['answer']}")
