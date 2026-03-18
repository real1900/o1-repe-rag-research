This prompt is engineered for a high-level AI coding or research agent to implement the specific technical refinements discussed. It treats the current "proof of concept" as a base and provides the mathematical and logical requirements to upgrade it into a production-ready **Mechanistic RAG Inference Engine**.

---

## **AI Implementation Prompt: Mechanistic RAG Upgrades**

**Role:** Senior AI Research Engineer (Mechanistic Interpretability & RAG).

**Task:** Refine and implement five critical architectural upgrades for a **Representation Engineering (RepE)** steering engine. The goal is to move from static, manual steering to a dynamic, $O(1)$ system that is robust against "Hard Negatives" and "Semantic Collapse."

### **Upgrade 1: Contrastive Representation Extraction ($V_{neg} - V_{pos}$)**
* [cite_start]**Objective:** Isolate the causal signature of hallucinations rather than general semantic topics[cite: 195, 196].
* [cite_start]**Implementation:** Replace raw distractor signature extraction with a **Contrastive Pair Pass**[cite: 196].
* [cite_start]**Logic:** For a retrieved chunk, generate a "Positive Pass" (relevant context) and a "Negative Pass" (distractor context)[cite: 195, 197]. [cite_start]Define the **Negative Control Vector** as $V_{diff} = V_{neg} - V_{pos}$[cite: 7, 195].
* [cite_start]**Target:** Ensure the resulting vector has near-zero semantic correlation to generic grammatical syntax to protect linguistic integrity[cite: 197, 198].

### **Upgrade 2: SQuAD Robustness & "Zero-Drop" Benchmarking**
* [cite_start]**Objective:** Validate that mechanistic steering does not degrade performance on high-baseline tasks[cite: 199].
* [cite_start]**Method:** Execute a 1,000-sample benchmark using **SQuAD v1.1**[cite: 190, 192].
* [cite_start]**Metrics:** Compare **Baseline** (Clean), **Contaminated** (Distractor added), and **Steered** (RepE applied)[cite: 199, 200].
* [cite_start]**Success Criteria:** Achieve "Zero-Drop Accuracy," ensuring the steered model maintains its ~80% baseline capability even when "blind" steering is active[cite: 191, 201].

### **Upgrade 3: Multi-Dimensional Manifold Triage (The "Hard Negative" Fix)**
* [cite_start]**Objective:** Resolve "Hard Negatives" (topically similar but factually incorrect distractors)[cite: 208].
* [cite_start]**Implementation:** Ensemble **Spatial Triage** (Mahalanobis Distance) with **Veracity Triage** (Factuality Projection)[cite: 209, 210].
* [cite_start]**Spatial Logic:** Use Mahalanobis Distance to identify topic outliers based on the elliptical variance of the context cluster[cite: 208, 209].
* [cite_start]**Veracity Logic:** Use **Contrastive Activation Addition (CAA)** to pre-compute a **Factuality Vector** ($\vec{V}_{fact}$)[cite: 7]. Project all $K$ retrieved chunks onto this axis.
* [cite_start]**Negation Trigger:** If a chunk aligns with the "False" pole of $\vec{V}_{fact}$ or exhibits a high **Jensen-Shannon Divergence** "Conflict Signature," trigger mechanistic negation even if it is not a spatial outlier[cite: 8, 203].

### **Upgrade 4: Token-Level Gating via Residual Stream L2 Norm**
* [cite_start]**Objective:** Prevent "Ceiling of Destruction" failures on non-factual tokens[cite: 205].
* [cite_start]**Logic:** Implement a dynamic forward hook that monitors the **Residual Stream L2 Norm** ($|h|_2$) at every decoding step[cite: 8, 206].
* [cite_start]**Scaling:** Factual entities exhibit significant mathematical spikes ($|h|_2 > 101.5$) compared to grammatical syntax[cite: 206, 207].
* [cite_start]**Implementation:** Scale the steering coefficient ($\alpha$) linearly with the L2 norm, effectively setting $\alpha \approx 0$ for grammatical filler words and $\alpha_{max}$ for factual tokens[cite: 207].

### **Upgrade 5: Automated Factual Probing (Logit Lens)**
* [cite_start]**Objective:** Eliminate manual layer sweeps (e.g., layers 12–20) to ensure cross-model generalizability[cite: 202].
* [cite_start]**Method:** Integrate a **Logit Lens Probe** during the initial encoding pass[cite: 8, 203].
* [cite_start]**Logic:** Compute the **Jensen-Shannon Divergence (JSD)** between unsteered and distracted logits at each layer[cite: 8, 203].
* [cite_start]**Targeting:** Dynamically inject the steering vector only into the layers where the "factual drift" exceeds a calculated threshold, protecting the deep layers responsible for logical grammar[cite: 204, 214].

Here is the psuedocode for this refinement: 
import torch
import torch.nn.functional as F

class MultiDimensionalTriage:
    """
    Implements Upgrade 3: Hybrid Spatial-Veracity Triage for Mechanistic RAG.
    Combines Mahalanobis Distance (Topic Outliers) with Factuality Projections (Truth Outliers).
    """
    def __init__(self, factuality_vector, inv_covariance_matrix, centroid, thresholds):
        # V_fact is pre-computed via Contrastive Activation Addition (CAA)
        self.v_fact = factuality_vector 
        # Pre-computed from latent topic cluster [cite: 209]
        self.inv_cov = inv_covariance_matrix
        self.centroid = centroid
        self.thresholds = thresholds # e.g., {'kl_brake': 2.0, 'l2_factual': 101.5} [cite: 164, 206]

    def get_mahalanobis_distance(self, activations):
        """Calculates geometric outlier severity[cite: 209]."""
        delta = activations - self.centroid
        m_dist = torch.sqrt(torch.matmul(torch.matmul(delta, self.inv_cov), delta.T))
        return m_dist

    def get_veracity_score(self, activations):
        """Projects hidden states onto the 'Truth' manifold."""
        # High positive = aligns with truth; High negative = aligns with false pole
        return torch.cosine_similarity(activations, self.v_fact, dim=-1)

    def uncertainty_gate(self, veracity_score):
        """
        Detects if the model lacks internal logic for the fact.
        If entropy is too high (score near 0), steering is disabled.
        """
        return torch.abs(veracity_score) < self.thresholds['uncertainty_margin']

    def run_triage(self, retrieved_chunks_activations):
        """
        Evaluates K chunks to identify the single 'malicious' distractor.
        """
        results = []
        for act in retrieved_chunks_activations:
            spatial_dist = self.get_mahalanobis_distance(act)
            veracity_score = self.get_veracity_score(act)
            
            # Logic: If it's a spatial outlier OR aligns with the False pole, it's a distractor
            is_distractor = (spatial_dist > self.thresholds['m_dist_limit']) or \
                            (veracity_score < self.thresholds['false_pole_limit'])
            
            results.append({
                'activation': act,
                'score': spatial_dist - veracity_score, # Combined risk metric
                'is_distractor': is_distractor and not self.uncertainty_gate(veracity_score)
            })

        # Return the activation signature of the most malicious chunk for negation [cite: 7, 31]
        target_chunk = max(results, key=lambda x: x['score'])
        return target_chunk['activation'] if target_chunk['is_distractor'] else None

# Example Usage in Forward Hook [cite: 37]
def RepE_Steering_Hook(module, input, output):
    triage_engine = MultiDimensionalTriage(...)
    # Extract contrastive signature [cite: 196]
    v_neg_control = triage_engine.run_triage(current_hidden_states)
    
    if v_neg_control is not None:
        # Apply Token-Level Gating via L2 Norm [cite: 8, 205]
        l2_norm = torch.norm(output, p=2)
        alpha = dynamic_mlp_probe(output, l2_norm) # [cite: 43, 44]
        
        # O(1) Mechanistic Subtraction [cite: 33, 138, 139]
        return output - (alpha * v_neg_control)
    return output

-------

important: 

Ensure the retrieved_chunks_activations are generated via contrastive pairs to isolate causal signatures.


Mahalanobis Optimization: Vectorize the distance calculation to ensure it remains O(1) relative to token generation.


KL-Divergence Brake: Integrate a safety check within the hook to decay alpha if KL(steered || unsteered) > 2.0.

Layer Targeting: Apply this hook only to layers identified by Logit Lens Probing (typically middle layers).

