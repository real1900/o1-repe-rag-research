# Future Implementation Plan: Context-Aware RAG Refinement

This document outlines the technical steps required to implement the "Negative Feedback Adapter" proposed in `research_proposal.md`.

## 1. Environment Setup

```bash
# Required libraries
pip install torch transformers peft datasets faiss-cpu sentence-transformers accelerate bitsandbytes
```

## 2. Methodology Implementation Steps

### Step A: Dataset Preparation (HotpotQA)
1.  **Load Data:** Use `datasets.load_dataset("hotpot_qa", "distractor")`.
2.  **Generate Synthetic Training Data:**
    *   **Input:** Question + Distractor Paragraph.
    *   **Target Output:** A critique string (e.g., "This paragraph discusses [Topic A], but the question asks about [Topic B].")
    *   *Note:* You may need to use a powerful model (like GPT-4 or Llama-3-70B) once to generate these "Target Critiques" to create a distilled training set for your smaller adapter.

### Step B: Fine-Tuning the Adapter (LoRA)
*   **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`.
*   **Technique:** Use `PEFT` (Parameter-Efficient Fine-Tuning) with LoRA configuration.
    *   Rank (`r`): 16 or 32.
    *   Target Modules: `q_proj`, `v_proj`.
*   **Training Loop:** Standard Hugging Face `Trainer` on the (Question+Distractor, Critique) pairs.

### Step C: The Retrieval Pipeline (Inference)
The final script `run_pipeline.py` should function as follows:

1.  **Pass 1:** Retrieve top-3 chunks using `sentence-transformers` + `FAISS`.
2.  **Critique:** Pass `(Question, Top-1 Chunk)` into your **Fine-Tuned Adapter**.
    *   *Output:* "Negative Feedback" string.
3.  **Refine:** Concatenate `Question + " " + Negative Feedback`.
4.  **Pass 2:** Re-query FAISS with this new string.
5.  **Generate:** Pass the final chunks to the base LLM to answer.

## 3. Evaluation
*   Measure **Recall@10** on the Pass 2 results vs. Pass 1.
*   Calculate **F1 Score** on the final answer compared to Gold.

---
*Save this plan for later execution.*
