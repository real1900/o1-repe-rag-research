# O(1) RepE RAG: Mechanistic Steering for Constant-Time Hallucination Correction

This repository contains the source code and final research paper for our novel approach to resolving the latency bottleneck inherent in Iterative Retrieval-Augmented Generation (RAG) systems.

## Overview
Traditional "Generate-and-Critique" RAG pipelines rely on expensive, autoregressive language generation ($O(N)$ latency) to identify and correct hallucinations caused by distractor context. 

This project proves that autoregressive text generation is not explicitly required to correct hallucinatory retrieval drift. By bridging the principles of **Representation Engineering (RepE)** with RAG critique mechanisms, we extract the "Activation Signature" (Negative Control Vector) of a distractor paragraph and algorithmically subtract it from the LLM's active hidden layers during inference.

## Key Breakthroughs
- **$O(1)$ Latency Correction:** Replaced multi-second token generation critique loops with constant-time vector math.
- **Dynamic $\alpha$ Steering:** We trained a lightweight PyTorch multi-layer perceptron (MLP) probe to instantly calculate the optimal steering coefficient ($\alpha$) based on the initial prompt's geometric alignment and the model's internal token confidence.
- **Experimental Results:** Evaluated on the HotpotQA "Distractor Setting" using a constrained model baseline, the $O(1)$ RepE architecture achieved a **+50% relative improvement in Answer Exact Match** accuracy while simultaneously yielding a **-9.70% reduction in inference latency**.

## Repository Structure
- **`draft_paper.pdf`**: The finalized research paper detailing the methodology, mathematical bounds, and analytical results.
- **`evaluate_pipeline.py`**: The final pipeline script that runs the blind HotpotQA evaluation against the dynamic $\alpha$ MLP probe.
- **`train_linear_probe.py`**: The PyTorch script used to train the $O(1)$ dynamic steering coefficient predictor over the synthetic dataset.
- **`generate_synthetic_data.py`**: The offline extraction script used to discover the geometric "Ceiling of Destruction" and optimum "Sweet Spot".
- **`test_formula.py` / `test_alpha.py`**: Experimental sandbox scripts for calibrating raw RepE vector constraints.

## Productionization (Future Work)
To scale this mathematically proven $O(1)$ steering mechanism to production-grade environments (e.g., Llama-3 70B) with high zero-shot baselines, we propose a **Hybrid Inference Architecture**:
1. **Two-Stage Filtering:** Utilizing a fast Cross-Encoder (e.g., `BGE-Reranker`) to safely triage and flag distractors prior to RepE vector extraction.
2. **KL-Divergence Braking:** Implementing dynamic mathematical braking to prevent semantic collapse during aggressive vector steering.
3. **Targeted Layer Steering:** Injecting the Negative Control Vector specifically into factual retrieval layers, preserving the LLM's deepest logic and grammar layers untouched.

## Execution Requirements
Ensure you have the required dependencies (PyTorch, Hugging Face Transformers, Datasets, Pandas) installed within your `.venv`. 

To run the final model evaluation pipeline:
```bash
python evaluate_pipeline.py
```
