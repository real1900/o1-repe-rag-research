# Research Project Proposal

## A) Title
**Mechanistic Steering for RAG: Context-Aware Query Refinement via Representation Engineering**

## B) Outline
1.  **Abstract**: Proposes a "Negative Control Vector" mechanism where the model mathematically subtracts the activation signature of irrelevant retrieval chunks during inference to steer away from hallucinations without generating textual critiques.
2.  **Introduction**:
    *   Context: RAG is standard, but "blind" query expansion often retrieves redundant information.
    *   Gap: Generating text-based critiques (like Self-RAG) incurs massive latency and token costs. Current systems lack a way to mathematically negate distractor information at the representation level.
    *   Research Question: Can Representation Engineering (RepE)—specifically extracting and negating activation signatures of distractor paragraphs from an LLM's hidden layers—guide generative search and improve Answer F1 without the latency overhead of multi-pass critique generation?
3.  **Methodology**:
    *   Architecture: Extracting "Negative Control Vectors" via PyTorch hooks during a distractor forward pass.
    *   Inference: Manually subtracting the vector from the LLM's hidden states via generation hooks.
4.  **Results (Planned)**:
    *   Primary Metric: Answer F1 and Perplexity measured across a wide spectrum of steering coefficients ($\alpha$).
    *   Secondary Metric: "Hallucination Rate" reduction and latency overhead compared to text-based critique baselines.
5.  **Discussion**: Analyzing the zero-latency overhead of RepE mathematical subtraction vs. traditional textual critique methods, and identifying the optimal $\alpha$ point that prevents "Answer Collapse."
6.  **Conclusion**: Summary of the "Negative Control Vector" novel contribution to RAG alignment.

## C) Problem, Candidate Solution, and Hypothesis
*   **Problem (The Research Gap):** Current "Iterative RAG" or "Query Expansion" methods (like Self-RAG) suffer from extreme latency bloat because they must autoregressively generate textual tokens to "critique" or rewrite queries. There is a need for a mechanism to steer models away from "retrieval drift" deterministically, without relying on slow token-by-token text generation.
*   **Candidate Solution:** We propose a **Representation Engineering (RepE) Control pipeline**.
    1.  **Step 1:** Initial standard retrieval yields a "distractor" chunk.
    2.  **Step 2 (The Novelty - Extraction):** We run a fast forward pass of the LLM on the distractor and use PyTorch hooks to extract the Principle Components (PCA) of its hidden layer activations, forming a "Negative Control Vector".
    3.  **Step 3 (Steering and Tuning):** During generation of the final answer, we mathematically subtract this negative vector directly from the LLM's computation graph at every generation step, governed by an intensity coefficient ($\\alpha$). 
    4.  **Step 4 (Automating O(1) Tuning):** We algorithmically calculate the "Ceiling of Destruction" ($Collapse\\_Alpha = \\frac{P \\cdot C}{||C||^2}$) and synthetically sweep $\\alpha$ for each prompt to find its perfect "Sweet Spot". We will then train a lightweight Linear Probe (ML Predictor) on this synthetic dataset to map incoming prompt geometry directly to the optimal $\\alpha$, restoring true $O(1)$ inference latency.
*   **Hypothesis:** We hypothesize that subtracting Negative Control Vectors at inference time will match or exceed the Answer F1-score of traditional text-based Critique RAG on the HotpotQA dataset, while reducing End-to-End Latency by >30% by transitioning the critique bottleneck from an $O(N)$ text-generation operation to an $O(1)$ mathematical subtraction.

| Mechanism | Critique Type | Computational Complexity | Impact on RAG Pipeline |
| :--- | :--- | :--- | :--- |
| **Standard Iterative** | Text Generation | Autoregressive $O(N)$ | Severe Latency Bloat |
| **RepE Steering** | Vector Subtraction | Constant Time $O(1)$ | Near-Zero Latency Overhead |

## D) Methodology and Data
*   **Machine Learning Method (PyTorch Implementation):**
    *   **Base Model:** **Llama-3-8B-Instruct**.
    *   **Implementation:** We will use native PyTorch `.register_forward_hook()` methods to capture hidden state dimensions (`d_model`) across transformer layers. No LoRA/SFT training is required; this is a purely mechanistic interference approach.
    *   **Retrieval Engine:** FAISS vector store using `all-MiniLM-L6-v2` for dense embeddings.
*   **Dataset:**
    *   **HotpotQA (Distractor Setting):** Selected because it explicitly contains "distractor" paragraphs, which serve as perfect ground-truth data for training the model to recognize and critique irrelevant information.

## E) References
1.  **Gao, Y., et al. (2024).** "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv preprint*. Available at: [https://arxiv.org/abs/2312.10997](https://arxiv.org/abs/2312.10997)
2.  **Saxena, A. & Bhattacharyya, P. (2024).** "Hallucination Detection in Machine Generated Text: A Survey." *CFILT Pre-print*. Available at: [https://www.cfilt.iitb.ac.in/resources/surveys/2024/survey_ashita_hallucination_detection_in_machine_generated_text_2024.pdf](https://www.cfilt.iitb.ac.in/resources/surveys/2024/survey_ashita_hallucination_detection_in_machine_generated_text_2024.pdf)
