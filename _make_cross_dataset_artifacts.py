"""Generate the cross-dataset table and figure for the paper."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load(name):
    p = Path(name)
    return json.loads(p.read_text()) if p.exists() else None

def fmt_p(p):
    if p is None: return "---"
    if p < 1e-4: return r"$<\!10^{-4}$"
    return f"{p:.4f}"

def sig(p):
    if p is None: return ""
    if p < 0.001: return r"$^{\ast\ast\ast}$"
    if p < 0.01:  return r"$^{\ast\ast}$"
    if p < 0.05:  return r"$^{\ast}$"
    if p < 0.10:  return r"$^{\dagger}$"
    return ""

# Load HotpotQA (existing) + TriviaQA + 2WikiMultihop (new)
hotpot = load("statistical_summary_random_control.json")
triv = load("statistical_summary_cross_triviaqa.json")
twiki = load("statistical_summary_cross_twiki.json")

datasets = [
    ("HotpotQA Distractor (multi-hop)", hotpot, "hotpot"),
    ("TriviaQA-RC (single-hop factoid)", triv, "triviaqa"),
    ("2WikiMultihopQA (multi-hop)", twiki, "twiki"),
]

# Cross-dataset table
out = []
out.append(r"\begin{table*}[htbp]")
cap = (r"Cross-dataset evaluation of MACH-1 (Llama-3.2-3B-Instruct, layer 7, $\alpha\!=\!2$, $n\!=\!500$ paired queries each). "
       r"\textbf{Contrastive}: the MACH-1 direction $V_{\text{neg}}-V_{\text{pos}}$. "
       r"\textbf{Random}: mean across 5 fixed random unit vectors at the same magnitude. "
       r"$\Delta_{\text{base}}$: contrastive vs.\ Blind RAG. $\Delta_{\text{rand}}$: contrastive vs.\ random direction average. "
       r"$p$: paired permutation test (10{,}000 sign-flipped permutations) of contrastive vs.\ Blind RAG. "
       r"Significance: $^{\ast}p<0.05$, $^{\dagger}p<0.10$.")
out.append(r"\caption{" + cap + r"}")
out.append(r"\label{tab:cross-dataset}")
out.append(r"\centering\renewcommand{\arraystretch}{1.18}")
out.append(r"\begin{tabular}{l r r r r r r}")
out.append(r"\toprule")
out.append(r"Dataset & Baseline EM & Contrastive EM & Random (mean of 5) & $\Delta_{\text{base}}$ (pp) & $\Delta_{\text{rand}}$ (pp) & Perm $p$ \\")
out.append(r"\midrule")
inline = {}
for name, S, key in datasets:
    if S is None:
        out.append(f"{name} & PENDING & --- & --- & --- & --- & --- \\\\")
        continue
    rows = S["rows"]
    base = next(r for r in rows if r["condition"] == "baseline")
    contr = next(r for r in rows if r["condition"] == "contrastive")
    base_em = 100*base["em"]
    contr_em = 100*contr["em"]
    rand_em = 100*S["random_dir_mean"]
    d_base = contr["delta_pp"]
    d_rand = contr_em - rand_em
    p = contr["perm_p"]
    out.append(f"{name} & {base_em:.2f}\\% & \\textbf{{{contr_em:.2f}\\%}} & {rand_em:.2f}\\% & "
               f"{d_base:+.2f} & {d_rand:+.2f} & {fmt_p(p)}{sig(p)} \\\\")
    inline[f"{key}_baseline"] = round(base_em, 2)
    inline[f"{key}_contrastive"] = round(contr_em, 2)
    inline[f"{key}_random"] = round(rand_em, 2)
    inline[f"{key}_delta_base"] = round(d_base, 2)
    inline[f"{key}_delta_rand"] = round(d_rand, 2)
    inline[f"{key}_p"] = p
out.append(r"\bottomrule\end{tabular}\end{table*}")
Path("cross_dataset_table.tex").write_text("\n".join(out) + "\n")
print("Wrote cross_dataset_table.tex")

# Figure: 3-panel cross-dataset bar chart
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
for ax, (name, S, key) in zip(axes, datasets):
    if S is None:
        ax.set_title(f"{name}\n(pending)", fontsize=10)
        continue
    rows = S["rows"]
    base_em = 100*next(r["em"] for r in rows if r["condition"] == "baseline")
    contr = next(r for r in rows if r["condition"] == "contrastive")
    contr_em = 100*contr["em"]
    contr_lo = 100*contr["em"] - 100*contr["ci_lo"]
    contr_hi = 100*contr["ci_hi"] - 100*contr["em"]
    rand_em = 100*S["random_dir_mean"]
    rand_std = 100*S["random_dir_std"]
    rand_pq_row = next((r for r in rows if r["condition"] == "random_per_query"), None)
    rand_pq_em = 100*rand_pq_row["em"] if rand_pq_row else None
    base_lo = 100*next(r["em"] - r["ci_lo"] for r in rows if r["condition"] == "baseline")
    base_hi = 100*next(r["ci_hi"] - r["em"] for r in rows if r["condition"] == "baseline")

    labels = ["Baseline", "Contrastive\n(MACH-1)", "Random\n(mean of 5)"]
    vals = [base_em, contr_em, rand_em]
    err_lo = [base_lo, contr_lo, 0]
    err_hi = [base_hi, contr_hi, rand_std]
    colors = ["#34495E", "#16A085", "#7F8C8D"]
    if rand_pq_em is not None:
        labels.append("Random\nper-query")
        vals.append(rand_pq_em)
        rpq_lo = 100*next(r["em"] - r["ci_lo"] for r in rows if r["condition"] == "random_per_query")
        rpq_hi = 100*next(r["ci_hi"] - r["em"] for r in rows if r["condition"] == "random_per_query")
        err_lo.append(rpq_lo); err_hi.append(rpq_hi)
        colors.append("#A0AEC0")
    bars = ax.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.errorbar(range(len(labels)), vals, yerr=[err_lo, err_hi], fmt="none", capsize=4, color="black", linewidth=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + max(err_hi[i], 0.5) + 0.3, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Exact Match (%)")
    delta_str = f"$\\Delta\\!=\\!{contr['delta_pp']:+.2f}$, p={contr['perm_p']:.3f}"
    ax.set_title(f"{name}\nn=500 paired; {delta_str}", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("cross_dataset_figure.png", dpi=140, bbox_inches="tight")
plt.close()
print("Wrote cross_dataset_figure.png")

Path("cross_dataset_inline.json").write_text(json.dumps(inline, indent=1))
print("\nKey inline numbers:")
for k, v in inline.items(): print(f"  {k}: {v}")
