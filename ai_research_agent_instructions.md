# AI Agent Instructions: Research Topic Selection & Outline Generation

**Objective:** Act as a research assistant to identify a novel research gap in Generative AI/Deep Learning, select a topic, and generate a one-page research outline based on the criteria in `intro.md`.

## Phase 1: Exploration & Topic Selection (Google Scholar)

1.  **Broad Search (Current State):**
    *   Perform a search on Google Scholar for "Generative AI", "Large Language Models", "Transformer architectures", or "Multimodal Learning".
    *   **Filter:** specific to the last 12-18 months (e.g., "since 2024").
    *   **Sort:** by citations or relevance to identify "well-received" and impactful papers.

2.  **Gap Analysis:**
    *   Select 3-5 top papers.
    *   Read the **Abstract**, **Conclusion**, and **Future Work** sections.
    *   Identify "Research Gaps": Look for phrases like "future work should address...", "a limitation of this approach is...", or "we did not explore...".
    *   *Constraint:* The gap must be addressable with accessible resources (available open-source datasets and implementable in PyTorch).

3.  **Topic Confirmation:**
    *   Select ONE specific gap.
    *   Formulate a specific **Research Question** (Hypothesis) that addresses this gap.
    *   Ensure there is a public dataset available (e.g., on Hugging Face or Kaggle) relevant to this problem.

## Phase 2: One-Page Outline Generation

Once the topic is selected, generate a one-page document containing the following 5 sections (strictly following `intro.md` requirements):

### A) Title
*   Create a concise, professional title for the research project.

### B) Outline
*   Standard Research Structure: Abstract, Introduction, Methodology, Results, Discussion, Conclusion.
*   *Note:* This should be a high-level table of contents or brief descriptors for the final paper structure.

### C) Problem & Solution
*   **Problem:** Clearly state the "gap" you found. Why is current tech insufficient?
*   **Candidate Solution:** Describe your proposed method (e.g., "We propose modifying the Attention mechanism by...").
*   **Hypothesis:** State what you expect to observe (e.g., "This modification will reduce training time by X% while maintaining accuracy").

### D) Methodology & Data
*   **Method:** Specify the Machine Learning/Deep Learning technique (e.g., "Fine-tuning LLaMA-2 using LoRA").
*   **Dataset:** Name the specific dataset(s) you will use (e.g., "IMDB Reviews", "SQuAD v2").

### E) References
*   Cite the 1-2 primary papers that helped you identify the gap.
*   Ensure they are recent and high-quality.

---

**Output Format:**
Return the result as a single markdown block titled `# Research Project Proposal`.
