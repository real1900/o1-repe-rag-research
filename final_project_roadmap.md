# Final Project Roadmap: O(1) Representation Engineering for RAG

This document outlines the step-by-step engineering roadmap for the final research project. Our goal is to synthesize the training data, build the $O(1)$ Machine Learning Predictor, and definitively prove that our Representation Engineering pipeline outperforms traditional "Iterative Critique RAG" in inference latency while matching or exceeding Answer F1.

## Phase 1: Environment & Dataset Preparation
- [ ] Initialize the Python environment and HuggingFace dependencies.
- [ ] Load the **HotpotQA** Training and Validation datasets.
- [ ] Implement data filtering to extract 5,000 high-quality `(Prompt, True Context, Distractor Context)` triplets.
- [ ] Standardize the PyTorch device environment (MPS/CUDA) and LLM loading pipeline for batch processing.

## Phase 2: Synthetic Data Generation (The Auto-Tuning Loop)
- [ ] Set up the `concept_vector` extraction mechanism.
- [ ] Implement the `Collapse_Alpha` boundary calculation ($Collapse\_Alpha = \frac{P \cdot C}{||C||^2}$) natively in PyTorch.
- [ ] Build the Binary Search Generation Loop for the "Sweet Spot":
    - [ ] Sweep $\alpha$ between $0.0$ and $Collapse\_Alpha$.
    - [ ] Measure text generated against the Ground Truth answer (Exact Match / F1).
    - [ ] Extract the model's **Baseline Token Confidence** (Max Logit / Entropy) for the initial prompt generation.
- [ ] Save the successful discoveries to `synthetic_alpha_tuning.csv` with features: 
      `[Prompt_Norm, Concept_Norm, Dot_Product, Cosine_Sim, Token_Confidence, Optimal_Alpha]`.
- [ ] Run the headless loop to generate exactly 5,000 synthetic rows of optimal steering parameters.

## Phase 3: Linear Probe Training (The Predictor)
- [ ] Create a PyTorch `Dataset` and `DataLoader` for `synthetic_alpha_tuning.csv`.
- [ ] Define the $O(1)$ ML Predictor architecture: `nn.Linear(Input_Features, 1)`.
- [ ] Train the probe using MSE Loss until convergence (expected to take seconds).
- [ ] Evaluate the probe's predicted $\alpha$ against a held-out validation set of sweeps.

## Phase 4: Full Pipeline Integration & Evaluation
- [ ] Integrate the trained Linear Probe natively into the RAG inference loop.
- [ ] Run the baseline "Blind RAG" (Retriever only) on the HotpotQA test set and measure Answer F1.
- [ ] Run the "RepE RAG" (with dynamically predicted $\alpha$) on the same test set and measure Answer F1.
- [ ] Measure End-to-End Latency for both pipelines to mathematically prove the $O(1)$ constant-time superiority over traditional $O(N)$ critique methods.

## Phase 5: Final Paper & Submission
- [ ] Generate comparative charts (Accuracy vs. Latency).
- [ ] Update `draft_paper.md` into the final manuscript with real statistical data and experimental findings.
- [ ] Clean up all scripts and the final Jupyter Notebook into a presentable, repeatable submission format.
