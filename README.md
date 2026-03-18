# MACH-1: Mechanistic Alignment for Constant-time Hidden-states (O(1))
*Breaking the O(N) latency barrier in Retrieval-Augmented Generation.*

This repository contains the source code, executable prototype, and final research paper for our novel approach to resolving the latency bottleneck inherent in Iterative Retrieval-Augmented Generation (RAG) systems.

## Overview
Traditional "Generate-and-Critique" RAG pipelines (like Self-RAG) rely on expensive, autoregressive language generation ($O(N)$ latency) to critically evaluate retrieved context blocks. If the documents are long or numerous, instructing the LLM to write a text-based critique of each block incurs massive latency bloat (e.g., 40+ seconds).

The **MACH-1** framework bridges Representation Engineering (RepE) with RAG. We extract the geometric "Activation Signature" (Negative Control Vector) of distracting retrieved paragraphs and mathematically subtract them natively from the LLM's active hidden layers during inference. Doing so mechanically eradicates the distractor's influence *before* text generation occurs, completely eliminating the $O(N)$ critique sequence.

## Key Breakthroughs
- **$O(1)$ Latency Correction:** Replaced multi-second token generation critique loops with constant-time tensor subtraction (~0.07s overhead).
- **Dynamic $\alpha$ Steering:** We trained a lightweight PyTorch multi-layer perceptron (MLP) probe to instantly calculate the optimal steering coefficient ($\alpha$) dynamically per query.
- **Experimental Results:** Evaluated scaling up to 1,500 real-world HotpotQA queries against a constrained proxy model baseline, the MACH-1 architecture achieved a measurable **+8.45% relative improvement in Answer Exact Match** accuracy while maintaining flat, near-zero geometric latency.

## Repository Structure
The project has been cleaned and consolidated into a self-generating pipeline:

- **`draft_paper.md`**: The complete, finalized markdown version of the academic paper containing the methodology, hypothesis, and mathematical bounds.
- **`build_notebook.py`**: The programmatic factory script that dynamically generates our execution pipeline into a clean Jupyter Notebook.
- **`project_prototype.ipynb`**: The self-contained, end-to-end executable prototype. It walks through synthetic distractor data generation, trains the Linear Probe predictor natively, runs the 1,500-query HotpotQA mass-evaluation, and calculates final metrics.
- **`plot_latency.py` & `latency_final.png`**: The matplotlib script and resulting high-fidelity (Tufte-compliant) visualization explicitly graphing the immense divergence between MACH-1's $O(1)$ constant overhead and Self-RAG's $O(N)$ text-generation bloat.

## Execution Requirements
Ensure you have the required dependencies (`torch`, `transformers`, `datasets`, `matplotlib`, `scipy`, `scikit-learn`) installed within your Python `.venv`. 

To dynamically build, evaluate, and inject the final metrics into the `project_prototype.ipynb` notebook automatically:

```bash
# 1. Generate the final graphic visual
python plot_latency.py

# 2. Build and automatically execute the end-to-end framework
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace project_prototype.ipynb
```
