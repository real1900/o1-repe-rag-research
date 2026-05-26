
### LOO CV: llama-3b (N=12)

LOO accuracy: **7/12 = 58.3%**
Chosen thresholds across folds: tau_stab = 0.675 +- 0.043, tau_var = 0.233 +- 0.029

| held-out task | tau_stab | tau_var | train acc | cos | var | |d_rand| | pred | obs | ok |
|---|---|---|---|---|---|---|---|---|---|
| context_faithfulness | 0.68 | 0.23 | 81.8% | 1.000 | 0.001 | 0.378 | no | steers | X |
| fact_override | 0.68 | 0.23 | 81.8% | 0.122 | 0.921 | 0.258 | no | steers | X |
| hallucination_grounding | 0.68 | 0.32 | 72.7% | 0.995 | 0.433 | 0.070 | steers | steers | OK |
| honesty | 0.68 | 0.23 | 72.7% | 0.999 | 0.039 | 0.010 | no | no | OK |
| persona | 0.68 | 0.23 | 81.8% | 1.000 | 0.004 | 0.628 | no | steers | X |
| politeness | 0.68 | 0.23 | 72.7% | 1.000 | 0.028 | 0.043 | no | no | OK |
| rag_distractor | 0.68 | 0.23 | 72.7% | 0.988 | 0.747 | 0.562 | steers | steers | OK |
| refusal | 0.68 | 0.23 | 72.7% | 0.998 | 0.626 | 0.622 | steers | steers | OK |
| sentiment | 0.78 | 0.23 | 72.7% | 0.752 | 0.823 | 0.459 | no | steers | X |
| sycophancy | 0.68 | 0.23 | 72.7% | 1.000 | 0.016 | 0.024 | no | no | OK |
| topic_suppression | 0.68 | 0.23 | 72.7% | 1.000 | 0.001 | 0.000 | no | no | OK |
| truthfulness | 0.57 | 0.23 | 72.7% | 0.576 | 0.892 | 0.043 | steers | no | X |

### LOO CV: qwen-3b (N=12)

LOO accuracy: **2/12 = 16.7%**
Chosen thresholds across folds: tau_stab = 0.652 +- 0.048, tau_var = 0.167 +- 0.246

| held-out task | tau_stab | tau_var | train acc | cos | var | |d_rand| | pred | obs | ok |
|---|---|---|---|---|---|---|---|---|---|
| context_faithfulness | 0.65 | 0.50 | 63.6% | 1.000 | 0.002 | 0.345 | no | steers | X |
| fact_override | 0.65 | 0.00 | 72.7% | 0.130 | 0.926 | 0.191 | no | steers | X |
| hallucination_grounding | 0.65 | 0.00 | 72.7% | 0.994 | 0.435 | 0.016 | steers | no | X |
| honesty | 0.65 | 0.50 | 63.6% | 1.000 | 0.018 | 0.062 | no | steers | X |
| persona | 0.65 | 0.50 | 63.6% | 1.000 | 0.007 | 0.280 | no | steers | X |
| politeness | 0.65 | 0.50 | 63.6% | 0.999 | 0.032 | 0.128 | no | steers | X |
| rag_distractor | 0.65 | 0.00 | 63.6% | 0.980 | 0.780 | 0.364 | steers | steers | OK |
| refusal | 0.65 | 0.00 | 63.6% | 0.998 | 0.656 | 0.894 | steers | steers | OK |
| sentiment | 0.78 | 0.00 | 63.6% | 0.749 | 0.862 | 0.125 | no | steers | X |
| sycophancy | 0.65 | 0.00 | 72.7% | 1.000 | 0.013 | 0.000 | steers | no | X |
| topic_suppression | 0.65 | 0.00 | 72.7% | 1.000 | 0.001 | 0.040 | steers | no | X |
| truthfulness | 0.55 | 0.00 | 63.6% | 0.554 | 0.897 | 0.022 | steers | no | X |

### LOO CV: llama-1b (N=12)

LOO accuracy: **4/12 = 33.3%**
Chosen thresholds across folds: tau_stab = 0.688 +- 0.119, tau_var = 0.500 +- 0.059

| held-out task | tau_stab | tau_var | train acc | cos | var | |d_rand| | pred | obs | ok |
|---|---|---|---|---|---|---|---|---|---|
| context_faithfulness | 0.65 | 0.53 | 72.7% | 1.000 | 0.000 | 0.217 | no | steers | X |
| fact_override | 0.65 | 0.53 | 72.7% | 0.029 | 0.956 | 0.145 | no | steers | X |
| hallucination_grounding | 0.65 | 0.32 | 63.6% | 0.998 | 0.445 | 0.006 | steers | no | X |
| honesty | 0.65 | 0.53 | 63.6% | 0.999 | 0.028 | 0.050 | no | no | OK |
| persona | 0.65 | 0.53 | 72.7% | 1.000 | 0.001 | 0.584 | no | steers | X |
| politeness | 0.65 | 0.53 | 63.6% | 1.000 | 0.008 | 0.050 | no | no | OK |
| rag_distractor | 0.65 | 0.53 | 72.7% | 0.997 | 0.744 | 0.016 | steers | no | X |
| refusal | 1.00 | 0.45 | 63.6% | 1.000 | 0.629 | 0.436 | no | steers | X |
| sentiment | 0.85 | 0.50 | 63.6% | 0.730 | 0.873 | 0.187 | no | steers | X |
| sycophancy | 0.65 | 0.53 | 63.6% | 1.000 | 0.018 | 0.000 | no | no | OK |
| topic_suppression | 0.65 | 0.53 | 63.6% | 1.000 | 0.000 | 0.000 | no | no | OK |
| truthfulness | 0.55 | 0.53 | 63.6% | 0.561 | 0.918 | 0.018 | steers | no | X |

### LOO CV: qwen-7b (N=12)

LOO accuracy: **2/12 = 16.7%**
Chosen thresholds across folds: tau_stab = 0.675 +- 0.043, tau_var = 0.175 +- 0.258

| held-out task | tau_stab | tau_var | train acc | cos | var | |d_rand| | pred | obs | ok |
|---|---|---|---|---|---|---|---|---|---|
| context_faithfulness | 0.68 | 0.53 | 63.6% | 1.000 | 0.000 | 0.121 | no | steers | X |
| fact_override | 0.68 | 0.00 | 72.7% | 0.136 | 0.946 | 0.143 | no | steers | X |
| hallucination_grounding | 0.68 | 0.00 | 72.7% | 0.999 | 0.456 | 0.022 | steers | no | X |
| honesty | 0.68 | 0.00 | 72.7% | 1.000 | 0.014 | 0.000 | steers | no | X |
| persona | 0.68 | 0.53 | 63.6% | 1.000 | 0.000 | 0.320 | no | steers | X |
| politeness | 0.68 | 0.53 | 63.6% | 1.000 | 0.004 | 0.213 | no | steers | X |
| rag_distractor | 0.68 | 0.00 | 63.6% | 0.999 | 0.767 | 0.342 | steers | steers | OK |
| refusal | 0.68 | 0.00 | 63.6% | 1.000 | 0.650 | 0.944 | steers | steers | OK |
| sentiment | 0.78 | 0.00 | 63.6% | 0.767 | 0.885 | 0.246 | no | steers | X |
| sycophancy | 0.68 | 0.00 | 72.7% | 1.000 | 0.010 | 0.000 | steers | no | X |
| topic_suppression | 0.68 | 0.53 | 63.6% | 1.000 | 0.000 | 0.840 | no | steers | X |
| truthfulness | 0.57 | 0.00 | 63.6% | 0.590 | 0.927 | 0.031 | steers | no | X |

### LOO CV: pooled across 4 models (N=48)

LOO accuracy: **30/48 = 62.5%**
Chosen thresholds across folds: tau_stab = 0.650 +- 0.000, tau_var = 0.549 +- 0.004

| held-out task | tau_stab | tau_var | train acc | cos | var | |d_rand| | pred | obs | ok |
|---|---|---|---|---|---|---|---|---|---|
| context_faithfulness | 0.65 | 0.55 | 63.8% | 1.000 | 0.001 | 0.378 | no | steers | X |
| fact_override | 0.65 | 0.55 | 63.8% | 0.122 | 0.921 | 0.258 | no | steers | X |
| hallucination_grounding | 0.65 | 0.55 | 63.8% | 0.995 | 0.433 | 0.070 | no | steers | X |
| honesty | 0.65 | 0.55 | 61.7% | 0.999 | 0.039 | 0.010 | no | no | OK |
| persona | 0.65 | 0.55 | 63.8% | 1.000 | 0.004 | 0.628 | no | steers | X |
| politeness | 0.65 | 0.55 | 61.7% | 1.000 | 0.028 | 0.043 | no | no | OK |
| rag_distractor | 0.65 | 0.55 | 61.7% | 0.988 | 0.747 | 0.562 | steers | steers | OK |
| refusal | 0.65 | 0.55 | 61.7% | 0.998 | 0.626 | 0.622 | steers | steers | OK |
| sentiment | 0.65 | 0.55 | 61.7% | 0.752 | 0.823 | 0.459 | steers | steers | OK |
| sycophancy | 0.65 | 0.55 | 61.7% | 1.000 | 0.016 | 0.024 | no | no | OK |
| topic_suppression | 0.65 | 0.55 | 61.7% | 1.000 | 0.001 | 0.000 | no | no | OK |
| truthfulness | 0.65 | 0.55 | 61.7% | 0.576 | 0.892 | 0.043 | no | no | OK |
| context_faithfulness | 0.65 | 0.55 | 63.8% | 1.000 | 0.002 | 0.345 | no | steers | X |
| fact_override | 0.65 | 0.55 | 63.8% | 0.130 | 0.926 | 0.191 | no | steers | X |
| hallucination_grounding | 0.65 | 0.55 | 61.7% | 0.994 | 0.435 | 0.016 | no | no | OK |
| honesty | 0.65 | 0.55 | 63.8% | 1.000 | 0.018 | 0.062 | no | steers | X |
| persona | 0.65 | 0.55 | 63.8% | 1.000 | 0.007 | 0.280 | no | steers | X |
| politeness | 0.65 | 0.55 | 63.8% | 0.999 | 0.032 | 0.128 | no | steers | X |
| rag_distractor | 0.65 | 0.55 | 61.7% | 0.980 | 0.780 | 0.364 | steers | steers | OK |
| refusal | 0.65 | 0.55 | 61.7% | 0.998 | 0.656 | 0.894 | steers | steers | OK |
| sentiment | 0.65 | 0.55 | 61.7% | 0.749 | 0.862 | 0.125 | steers | steers | OK |
| sycophancy | 0.65 | 0.55 | 61.7% | 1.000 | 0.013 | 0.000 | no | no | OK |
| topic_suppression | 0.65 | 0.55 | 61.7% | 1.000 | 0.001 | 0.040 | no | no | OK |
| truthfulness | 0.65 | 0.55 | 61.7% | 0.554 | 0.897 | 0.022 | no | no | OK |
| context_faithfulness | 0.65 | 0.55 | 63.8% | 1.000 | 0.000 | 0.217 | no | steers | X |
| fact_override | 0.65 | 0.55 | 63.8% | 0.029 | 0.956 | 0.145 | no | steers | X |
| hallucination_grounding | 0.65 | 0.55 | 61.7% | 0.998 | 0.445 | 0.006 | no | no | OK |
| honesty | 0.65 | 0.55 | 61.7% | 0.999 | 0.028 | 0.050 | no | no | OK |
| persona | 0.65 | 0.55 | 63.8% | 1.000 | 0.001 | 0.584 | no | steers | X |
| politeness | 0.65 | 0.55 | 61.7% | 1.000 | 0.008 | 0.050 | no | no | OK |
| rag_distractor | 0.65 | 0.55 | 63.8% | 0.997 | 0.744 | 0.016 | steers | no | X |
| refusal | 0.65 | 0.55 | 61.7% | 1.000 | 0.629 | 0.436 | steers | steers | OK |
| sentiment | 0.65 | 0.55 | 61.7% | 0.730 | 0.873 | 0.187 | steers | steers | OK |
| sycophancy | 0.65 | 0.55 | 61.7% | 1.000 | 0.018 | 0.000 | no | no | OK |
| topic_suppression | 0.65 | 0.55 | 61.7% | 1.000 | 0.000 | 0.000 | no | no | OK |
| truthfulness | 0.65 | 0.55 | 61.7% | 0.561 | 0.918 | 0.018 | no | no | OK |
| context_faithfulness | 0.65 | 0.55 | 63.8% | 1.000 | 0.000 | 0.121 | no | steers | X |
| fact_override | 0.65 | 0.55 | 63.8% | 0.136 | 0.946 | 0.143 | no | steers | X |
| hallucination_grounding | 0.65 | 0.53 | 61.7% | 0.999 | 0.456 | 0.022 | no | no | OK |
| honesty | 0.65 | 0.55 | 61.7% | 1.000 | 0.014 | 0.000 | no | no | OK |
| persona | 0.65 | 0.55 | 63.8% | 1.000 | 0.000 | 0.320 | no | steers | X |
| politeness | 0.65 | 0.55 | 63.8% | 1.000 | 0.004 | 0.213 | no | steers | X |
| rag_distractor | 0.65 | 0.55 | 61.7% | 0.999 | 0.767 | 0.342 | steers | steers | OK |
| refusal | 0.65 | 0.55 | 61.7% | 1.000 | 0.650 | 0.944 | steers | steers | OK |
| sentiment | 0.65 | 0.55 | 61.7% | 0.767 | 0.885 | 0.246 | steers | steers | OK |
| sycophancy | 0.65 | 0.55 | 61.7% | 1.000 | 0.010 | 0.000 | no | no | OK |
| topic_suppression | 0.65 | 0.55 | 63.8% | 1.000 | 0.000 | 0.840 | no | steers | X |
| truthfulness | 0.65 | 0.55 | 61.7% | 0.590 | 0.927 | 0.031 | no | no | OK |

### Cross-model CV: train on 3 models, test on the 4th

| held-out model | train N | train acc | tau_stab | tau_var | test N | test correct | test acc |
|---|---|---|---|---|---|---|---|
| llama-3b | 36 | 61.1% | 0.65 | 0.55 | 12 | 8 | 66.7% |
| qwen-3b | 36 | 63.9% | 0.65 | 0.55 | 12 | 7 | 58.3% |
| llama-1b | 36 | 63.9% | 0.65 | 0.00 | 12 | 5 | 41.7% |
| qwen-7b | 36 | 63.9% | 0.65 | 0.53 | 12 | 7 | 58.3% |

Wrote loo_cv_geometric_only.json
