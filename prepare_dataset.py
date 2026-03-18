import json
from datasets import load_dataset
import random

# Phase 1: Environment & Dataset Preparation
# Goal: Extract exactly 5000 (Prompt, True Context, Distractor Context) triplets from HotpotQA

def prepare_hotpot_dataset(target_rows=5000, output_file="hotpot_filtered_5000.json"):
    print("Downloading/Loading HotpotQA (distractor setting)...")
    
    # We use validation split because it's slightly smaller and faster to process locally
    # It still contains full 'distractor' paragraphs natively in the dataset
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    print(f"Total dataset size: {len(dataset)} rows")
    print("Filtering and extracting...")
    
    extracted_data = []
    
    # Shuffle to ensure an unbiased, random distribution of question types
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    for idx in indices:
        row = dataset[idx]
        
        question = row['question']
        answer = row['answer']
        context_titles = row['context']['title']
        context_sentences = row['context']['sentences']
        
        # HotpotQA provides "supporting_facts" which tell us exactly which context paragraphs are True
        supporting_titles = [fact_title for fact_title, fact_id in zip(row['supporting_facts']['title'], row['supporting_facts']['sent_id'])]
        
        true_context_chunks = []
        distractor_context_chunks = []
        
        # Sift through the 10 provided paragraphs to separate Truth from Distractors
        for title, sentences in zip(context_titles, context_sentences):
            paragraph = " ".join(sentences)
            if title in supporting_titles:
                true_context_chunks.append(paragraph)
            else:
                distractor_context_chunks.append(paragraph)
                
        # We need rows that clearly have at least 1 true fact and 1 distractor
        if len(true_context_chunks) > 0 and len(distractor_context_chunks) > 0:
            
            # For simplicity in our pipeline, we just grab the first available of each
            true_passage = true_context_chunks[0]
            distractor_passage = distractor_context_chunks[0]
            
            # Simple heuristic: Only keep rows where the true answer is actually short text (not yes/no)
            if answer.lower() not in ["yes", "no"] and len(answer) > 1:
                extracted_data.append({
                    "id": row["id"],
                    "question": question,
                    "answer": answer,
                    "true_context": true_passage,
                    "distractor_context": distractor_passage
                })
                
        if len(extracted_data) >= target_rows:
            break
            
    print(f"Successfully extracted {len(extracted_data)} valid triplets.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2)
        
    print(f"Saved dataset to {output_file}")
    
    # Print a quick sample to verify
    if len(extracted_data) > 0:
        sample = extracted_data[0]
        print("\n--- SAMPLE ROW ---")
        print(f"Q: {sample['question']}")
        print(f"A: {sample['answer']}")
        print(f"Distractor: {sample['distractor_context'][:100]}...")

if __name__ == "__main__":
    prepare_hotpot_dataset(target_rows=5000)
