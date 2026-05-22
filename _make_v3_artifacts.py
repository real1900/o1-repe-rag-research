"""Generate the v3 final artifacts: Self-RAG h2h tables, 8B headline tables,
and the new comparative figures.  Run AFTER all chain phases complete.

Inputs (all in CWD):
  statistical_summary_selfrag.json          -- Self-RAG-style on Llama-3.2-3B
  statistical_summary_actual_selfrag.json   -- selfrag/selfrag_llama2_7b
  statistical_summary_8b.json               -- Llama-3.1-8B 5-seed validation

Outputs:
  selfrag_h2h_table.tex             -- main 3-row h2h table
  llama3_8b_headline_table.tex      -- 5-seed validation table
  llama3_8b_sweep_table.tex         -- joint layer×α sweep heatmap
  comparison_figure.png             -- bar chart: baseline / MACH-1 / Self-RAG-style / actual Self-RAG
  inline_numbers.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load(name): return json.loads(Path(name).read_text())

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

# ---------------- Self-RAG h2h table ----------------------------------------
sr = load("statistical_summary_selfrag.json")
asr = load("statistical_summary_actual_selfrag.json") if Path("statistical_summary_actual_selfrag.json").exists() else None
mach1_layer7 = load("statistical_summary_layer7.json")  # already on disk
hl_row = next(r for r in mach1_layer7["rows"] if abs(r["alpha"] - 2.0) < 1e-9)
base_row = next(r for r in mach1_layer7["rows"] if abs(r["alpha"]) < 1e-9)

out = []
out.append(r"\begin{table*}[htbp]")
caption_pieces = [
    r"Head-to-head on $n\!=\!500$ paired HotpotQA distractor queries.",
    r"\textbf{Blind RAG} = no retrieval critique, single greedy decode.",
    r"\textbf{MACH-1 Layer-7} = unit-direction contrastive steering at layer 7, $\alpha\!=\!2$.",
    r"\textbf{Self-RAG-style (Llama-3.2-3B)} = three-stage autoregressive critique on the same base model.",
]
if asr is not None:
    caption_pieces.append(r"\textbf{Actual Self-RAG} = \texttt{selfrag/selfrag\_llama2\_7b} (Asai et al.\ 2024 ICLR published checkpoint).")
caption_pieces += [
    r"CI: bootstrap 95\% (10{,}000 resamples).",
    r"Lat ratio: latency relative to Blind RAG on the same base model.",
    r"$p$: paired permutation test (10{,}000 sign-flipped permutations) vs.\ Blind RAG.",
    r"Significance: $^{\ast}p<0.05$, $^{\dagger}p<0.10$.",
]
out.append(r"\caption{" + " ".join(caption_pieces) + r"}")
out.append(r"\label{tab:selfrag-h2h}")
out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
out.append(r"\begin{tabular}{l c r r r r}")
out.append(r"\toprule")
out.append(r"Method & Base model & EM (\%) & 95\% CI & Lat (ms) & Lat ratio \\")
out.append(r"\midrule")

# Blind RAG (Llama-3.2-3B) - from sr
out.append(f"Blind RAG               & Llama-3.2-3B-Instruct & "
           f"{100*sr['baseline']['em']:.2f} & [{100*sr['baseline']['ci_lo']:.2f}, {100*sr['baseline']['ci_hi']:.2f}] & "
           f"{1000*sr['baseline']['lat_mean']:.1f} & 1.00$\\times$ \\\\")

# MACH-1 Layer 7 (Llama-3.2-3B)
mach1_lat_ratio = hl_row["lat_mean"] / base_row["lat_mean"]
out.append(f"\\textbf{{MACH-1}} ($\\ell\\!=\\!7$, $\\alpha\\!=\\!2$) & Llama-3.2-3B-Instruct & "
           f"\\textbf{{{100*hl_row['em']:.2f}}} & [{100*hl_row['ci_lo']:.2f}, {100*hl_row['ci_hi']:.2f}] & "
           f"\\textbf{{{1000*hl_row['lat_mean']:.1f}}} & \\textbf{{{mach1_lat_ratio:.2f}$\\times$}} \\\\")

# Self-RAG-style on same base model
out.append(f"Self-RAG-style          & Llama-3.2-3B-Instruct & "
           f"{100*sr['selfrag']['em']:.2f} & [{100*sr['selfrag']['ci_lo']:.2f}, {100*sr['selfrag']['ci_hi']:.2f}] & "
           f"{1000*sr['selfrag']['lat_mean']:.1f} & {sr['latency_ratio']:.2f}$\\times$ \\\\")

if asr is not None:
    out.append(r"\midrule")
    # Actual Self-RAG (Llama-2-7B)
    out.append(f"Actual Self-RAG (Asai et al.) & \\texttt{{selfrag\\_llama2\\_7b}} & "
               f"{100*asr['selfrag']['em']:.2f} & [{100*asr['selfrag']['ci_lo']:.2f}, {100*asr['selfrag']['ci_hi']:.2f}] & "
               f"{1000*asr['selfrag']['lat_mean']:.1f} & {asr['latency_ratio']:.2f}$\\times$ \\\\")
    # Same model baseline
    out.append(f"\\quad (same model, no critique tokens) & \\texttt{{selfrag\\_llama2\\_7b}} & "
               f"{100*asr['baseline']['em']:.2f} & [{100*asr['baseline']['ci_lo']:.2f}, {100*asr['baseline']['ci_hi']:.2f}] & "
               f"{1000*asr['baseline']['lat_mean']:.1f} & 1.00$\\times$ \\\\")
out.append(r"\bottomrule\end{tabular}\end{table*}")
Path("selfrag_h2h_table.tex").write_text("\n".join(out) + "\n")
print("Wrote selfrag_h2h_table.tex")

# ---------------- Llama-3.1-8B headline table ----------------
inline = {}
if Path("statistical_summary_8b.json").exists():
    s8 = load("statistical_summary_8b.json")
    pooled = s8["pooled"]
    seeds = s8["seed_summary"]
    out = []
    out.append(r"\begin{table}[htbp]")
    out.append(rf"\caption{{\textbf{{Llama-3.1-8B-Instruct headline}}: 5-seed paired validation at the best layer/$\alpha$ "
               rf"identified by phase-A sweep (best layer $\ell\!=\!{s8['best_layer']}$, "
               rf"$\alpha\!=\!{s8['best_alpha']}$). $n\!=\!500$ paired queries per seed. "
               rf"Pooled $\Delta\!=\!{pooled['delta_pp']:+.2f}$\,pp, paired permutation $p\!=\!{pooled['perm_p']:.4f}$. "
               rf"Per-seed mean $\Delta\!=\!{pooled['seed_delta_mean']:+.2f}\!\pm\!{pooled['seed_delta_std']:.2f}$\,pp.}}")
    out.append(r"\label{tab:llama8b-headline}")
    out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
    out.append(r"\begin{tabular}{c r r r r}")
    out.append(r"\toprule")
    out.append(r"Seed & Baseline EM & Steered EM & $\Delta$ (pp) & Perm $p$ \\")
    out.append(r"\midrule")
    for r in seeds:
        out.append(f"{r['seed']} & {100*r['baseline_em']:.2f}\\% & {100*r['steered_em']:.2f}\\% & "
                   f"{r['delta_pp']:+.2f} & {fmt_p(r['perm_p'])}{sig(r['perm_p'])} \\\\")
    out.append(r"\midrule")
    out.append(f"\\textbf{{Pooled}} & \\textbf{{{100*pooled['baseline_em']:.2f}\\%}} & \\textbf{{{100*pooled['steered_em']:.2f}\\%}} & "
               f"\\textbf{{{pooled['delta_pp']:+.2f}}} & \\textbf{{{fmt_p(pooled['perm_p'])}{sig(pooled['perm_p'])}}} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    Path("llama3_8b_headline_table.tex").write_text("\n".join(out) + "\n")
    print("Wrote llama3_8b_headline_table.tex")
    inline["8b_best_layer"] = s8["best_layer"]
    inline["8b_best_alpha"] = s8["best_alpha"]
    inline["8b_pooled_baseline"] = round(100*pooled["baseline_em"], 2)
    inline["8b_pooled_steered"] = round(100*pooled["steered_em"], 2)
    inline["8b_pooled_delta_pp"] = round(pooled["delta_pp"], 2)
    inline["8b_pooled_p"] = pooled["perm_p"]
    inline["8b_seed_mean_delta"] = round(pooled["seed_delta_mean"], 2)
    inline["8b_seed_std_delta"] = round(pooled["seed_delta_std"], 2)

    # 8B sweep table (best 5 conditions)
    sweep = sorted([r for r in s8["sweep_rows"] if r["condition"] != "baseline"], key=lambda r: -r["delta_pp"])
    out = []
    out.append(r"\begin{table}[htbp]")
    out.append(rf"\caption{{Llama-3.1-8B phase-A joint sweep (top 5 conditions by $\Delta$, $n\!=\!200$). "
               rf"Layer $\ell$ is the steering hook position; $\alpha$ is the unit-direction coefficient.}}")
    out.append(r"\label{tab:llama8b-sweep}")
    out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
    out.append(r"\begin{tabular}{c c r r r r}")
    out.append(r"\toprule")
    out.append(r"$\ell$ & $\alpha$ & EM (\%) & 95\% CI & $\Delta$ (pp) & Perm $p$ \\")
    out.append(r"\midrule")
    base_row_8b = next(r for r in s8["sweep_rows"] if r["condition"] == "baseline")
    out.append(f"---  & 0  & {100*base_row_8b['em']:.2f} & [{100*base_row_8b['ci_lo']:.2f}, {100*base_row_8b['ci_hi']:.2f}] & --- & --- \\\\")
    for r in sweep[:5]:
        cond = r["condition"]; L = int(cond.split("_")[0][1:]); a = int(cond.split("_a")[1])
        out.append(f"{L} & {a} & {100*r['em']:.2f} & [{100*r['ci_lo']:.2f}, {100*r['ci_hi']:.2f}] & "
                   f"{r['delta_pp']:+.2f} & {fmt_p(r['perm_p'])}{sig(r['perm_p'])} \\\\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    Path("llama3_8b_sweep_table.tex").write_text("\n".join(out) + "\n")
    print("Wrote llama3_8b_sweep_table.tex")

# ---------------- Comparison bar figure ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
ax1, ax2 = axes

# EM comparison
labels = ["Blind RAG\n(L3.2-3B)", "MACH-1\n(L3.2-3B,\n$\\ell$=7,$\\alpha$=2)", "Self-RAG-style\n(L3.2-3B)"]
ems = [100*sr["baseline"]["em"], 100*hl_row["em"], 100*sr["selfrag"]["em"]]
errs_lo = [100*sr["baseline"]["em"] - 100*sr["baseline"]["ci_lo"],
           100*hl_row["em"] - 100*hl_row["ci_lo"],
           100*sr["selfrag"]["em"] - 100*sr["selfrag"]["ci_lo"]]
errs_hi = [100*sr["baseline"]["ci_hi"] - 100*sr["baseline"]["em"],
           100*hl_row["ci_hi"] - 100*hl_row["em"],
           100*sr["selfrag"]["ci_hi"] - 100*sr["selfrag"]["em"]]
colors = ["#34495E", "#16A085", "#C0392B"]
if asr is not None:
    labels.extend(["Actual Self-RAG\n(L2-7B)", "Same-model\nbaseline"])
    ems.extend([100*asr["selfrag"]["em"], 100*asr["baseline"]["em"]])
    errs_lo.extend([100*asr["selfrag"]["em"] - 100*asr["selfrag"]["ci_lo"],
                    100*asr["baseline"]["em"] - 100*asr["baseline"]["ci_lo"]])
    errs_hi.extend([100*asr["selfrag"]["ci_hi"] - 100*asr["selfrag"]["em"],
                    100*asr["baseline"]["ci_hi"] - 100*asr["baseline"]["em"]])
    colors.extend(["#8E44AD", "#566573"])

ax1.bar(range(len(labels)), ems, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
ax1.errorbar(range(len(labels)), ems, yerr=[errs_lo, errs_hi], fmt="none", capsize=4, color="black", linewidth=1.2)
for i, em in enumerate(ems):
    ax1.text(i, em + max(errs_hi[i], 0.5) + 0.3, f"{em:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax1.set_xticks(range(len(labels))); ax1.set_xticklabels(labels, fontsize=8)
ax1.set_ylabel("Exact Match (%)"); ax1.set_title(f"Exact Match (n=500 paired)")
ax1.grid(axis="y", alpha=0.3)

# Latency comparison
lats = [1000*sr["baseline"]["lat_mean"], 1000*hl_row["lat_mean"], 1000*sr["selfrag"]["lat_mean"]]
if asr is not None:
    lats.extend([1000*asr["selfrag"]["lat_mean"], 1000*asr["baseline"]["lat_mean"]])

ax2.bar(range(len(labels)), lats, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
for i, l in enumerate(lats):
    ax2.text(i, l + max(lats)*0.015, f"{l:.0f} ms", ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Avg Latency (ms)"); ax2.set_title("Per-Query Wall-Clock Latency")
ax2.grid(axis="y", alpha=0.3)
ax2.set_yscale("log")

plt.tight_layout()
plt.savefig("comparison_figure.png", dpi=140, bbox_inches="tight")
print("Wrote comparison_figure.png")
plt.close()

# Save inline numbers
inline.update({
    "selfrag_baseline_em":  round(100*sr["baseline"]["em"], 2),
    "selfrag_baseline_lat": round(1000*sr["baseline"]["lat_mean"], 1),
    "selfrag_em":           round(100*sr["selfrag"]["em"], 2),
    "selfrag_lat":          round(1000*sr["selfrag"]["lat_mean"], 1),
    "selfrag_lat_ratio":    round(sr["latency_ratio"], 1),
    "selfrag_delta_pp":     round(sr["delta_em_pp"], 2),
    "selfrag_p":            sr["delta_em_perm_p"],
    "mach1_em":             round(100*hl_row["em"], 2),
    "mach1_lat":            round(1000*hl_row["lat_mean"], 1),
    "mach1_lat_ratio":      round(mach1_lat_ratio, 2),
})
if asr is not None:
    inline.update({
        "actual_selfrag_em":     round(100*asr["selfrag"]["em"], 2),
        "actual_selfrag_lat":    round(1000*asr["selfrag"]["lat_mean"], 1),
        "actual_selfrag_ratio":  round(asr["latency_ratio"], 1),
        "actual_selfrag_delta_pp": round(asr["delta_em_pp"], 2),
        "actual_selfrag_p":       asr["delta_em_perm_p"],
    })

Path("inline_numbers_v3.json").write_text(json.dumps(inline, indent=1))
print("\nKey inline numbers:")
for k, v in inline.items(): print(f"  {k}: {v}")
