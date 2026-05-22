"""Generate the LaTeX tables and figures used by the rewritten paper.

Outputs:
  - alpha_sweep_table.tex      (mid-layer α-magnitude sweep, n=200)
  - layer_sweep_table.tex      (layer-position sweep at α=2, n=200)
  - layer7_validation_table.tex (layer-7 α-sweep, n=500)  -- headline
  - gpt2_ablation_table.tex    (6-condition paired ablation on GPT-2, n=500)
  - performance_metrics.png    (3-panel: layer sweep, α sweep, layer-7 validation)
  - relative_improvement.png   (focused: baseline vs layer-7 α=2 with 95% CI)
  - inline_numbers.json        (key numbers for inline use in paper)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(".")

def load(name):
    return json.loads((ROOT / name).read_text())

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

# ---------- 1. Layer-7 validation table (HEADLINE; n=500) -------------------
S = load("statistical_summary_layer7.json")
meta = S["meta"]; rows = S["rows"]
out = []
out.append(r"\begin{table}[htbp]")
out.append(r"\caption{Headline result: $\alpha$-magnitude sweep at layer "
           rf"$\ell^\star\!=\!{meta['target_layer']}$ "
           rf"(early-middle of 28-layer Llama-3.2-3B-Instruct), $n\!=\!{meta['n_queries']}$ paired queries (HotpotQA rows {meta['eval_start']}--{meta['eval_end']}). EM = Exact Match. "
           rf"CI: bootstrap 95\% (10{{,}}000 resamples). $p$: paired permutation test (10{{,}}000 sign-flipped permutations) vs.\ baseline ($\alpha\!=\!0$). "
           rf"Significance: $^{{\ast}}p<0.05$, $^{{\dagger}}p<0.10$.}}")
out.append(r"\label{tab:layer7-headline}")
out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
out.append(r"\begin{tabular}{c r r r r}")
out.append(r"\toprule")
out.append(r"$\alpha$ & EM (\%) & 95\% CI & $\Delta$ (pp) & Perm $p$ \\")
out.append(r"\midrule")
for r in rows:
    em = 100*r["em"]; lo = 100*r["ci_lo"]; hi = 100*r["ci_hi"]
    d = r["delta_pp"]; p = r["perm_p"]
    d_str = "---" if d is None else f"{d:+.2f}"
    p_str = fmt_p(p) + sig(p)
    bold_open = r"\textbf{" if abs(r["alpha"] - 2.0) < 1e-9 else ""
    bold_close = r"}" if abs(r["alpha"] - 2.0) < 1e-9 else ""
    out.append(f"{bold_open}{r['alpha']:.0f}{bold_close} & {bold_open}{em:.2f}{bold_close} & "
               f"{bold_open}[{lo:.2f}, {hi:.2f}]{bold_close} & "
               f"{bold_open}{d_str}{bold_close} & {bold_open}{p_str}{bold_close} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
Path("layer7_validation_table.tex").write_text("\n".join(out)+"\n")
print("Wrote layer7_validation_table.tex")
HEADLINE = next(r for r in rows if abs(r["alpha"] - 2.0) < 1e-9)
BASE = next(r for r in rows if abs(r["alpha"]) < 1e-9)
inline = {
    "headline_n": meta["n_queries"],
    "headline_layer": meta["target_layer"],
    "headline_alpha": 2.0,
    "headline_baseline_em": round(100*BASE["em"], 2),
    "headline_steered_em": round(100*HEADLINE["em"], 2),
    "headline_delta_pp": round(HEADLINE["delta_pp"], 2),
    "headline_delta_rel": round(100*(HEADLINE["em"] - BASE["em"]) / BASE["em"], 1) if BASE["em"] > 0 else None,
    "headline_p": HEADLINE["perm_p"],
    "headline_baseline_lat_ms": round(1000*BASE["lat_mean"], 1),
    "headline_steered_lat_ms": round(1000*HEADLINE["lat_mean"], 1),
}

# ---------- 2. α-magnitude sweep (mid-layer; n=200) -------------------------
S = load("statistical_summary_llama3_v3.json")
meta = S["meta"]; rows = S["rows"]
out = []
out.append(r"\begin{table}[htbp]")
out.append(r"\caption{Mid-layer $\alpha$-magnitude sweep: layer "
           rf"$\ell\!=\!{meta['target_layer']}$ (mid of {meta['num_layers']}), $n\!=\!{meta['n_queries']}$ paired queries on Llama-3.2-3B-Instruct. "
           rf"Steering monotonically degrades accuracy as $\alpha$ grows; the default mid-layer target is the wrong choice. "
           rf"Significance: $^{{\ast\ast}}p<0.01$, $^{{\dagger}}p<0.10$.}}")
out.append(r"\label{tab:alpha-sweep}")
out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
out.append(r"\begin{tabular}{c r r r r}")
out.append(r"\toprule")
out.append(r"$\alpha$ & EM (\%) & 95\% CI & $\Delta$ (pp) & Perm $p$ \\")
out.append(r"\midrule")
for r in rows:
    em = 100*r["em"]; lo = 100*r["ci_lo"]; hi = 100*r["ci_hi"]
    d = r["delta_pp"]; p = r["perm_p"]
    d_str = "---" if d is None else f"{d:+.2f}"
    p_str = fmt_p(p) + sig(p)
    out.append(f"{r['alpha']:.0f} & {em:.2f} & [{lo:.2f}, {hi:.2f}] & {d_str} & {p_str} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
Path("alpha_sweep_table.tex").write_text("\n".join(out)+"\n")
print("Wrote alpha_sweep_table.tex")
inline["alpha_sweep_layer"] = meta["target_layer"]
inline["alpha_sweep_n"] = meta["n_queries"]
inline["alpha_20_delta_pp"] = next(r["delta_pp"] for r in rows if abs(r["alpha"] - 20) < 1e-9)
inline["alpha_20_p"] = next(r["perm_p"] for r in rows if abs(r["alpha"] - 20) < 1e-9)

# ---------- 3. Layer-position sweep (α=2; n=200) ----------------------------
S = load("statistical_summary_layer_sweep.json")
meta = S["meta"]; rows = S["rows"]
out = []
out.append(r"\begin{table}[htbp]")
out.append(rf"\caption{{Layer-position sweep at fixed $\alpha\!=\!{meta['alpha']}$, $n\!=\!{meta['n_queries']}$ paired queries on Llama-3.2-3B-Instruct ($L\!=\!{meta['num_layers']}$). Only early layers ($\ell\!\leq\!7$) admit positive directional effects. Significance: $^{{\dagger}}p<0.10$.}}")
out.append(r"\label{tab:layer-sweep}")
out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
out.append(r"\begin{tabular}{c r r r r}")
out.append(r"\toprule")
out.append(r"Layer $\ell$ & EM (\%) & 95\% CI & $\Delta$ (pp) & Perm $p$ \\")
out.append(r"\midrule")
for r in rows:
    em = 100*r["em"]; lo = 100*r["ci_lo"]; hi = 100*r["ci_hi"]
    d = r["delta_pp"]; p = r["perm_p"]
    d_str = "---" if d is None else f"{d:+.2f}"
    p_str = fmt_p(p) + sig(p)
    label = "baseline (no steering)" if r["layer"] is None else f"$\\ell={r['layer']}$"
    out.append(f"{label} & {em:.2f} & [{lo:.2f}, {hi:.2f}] & {d_str} & {p_str} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
Path("layer_sweep_table.tex").write_text("\n".join(out)+"\n")
print("Wrote layer_sweep_table.tex")

# ---------- 4. GPT-2 ablation (n=500, 6 conditions) -------------------------
S = load("statistical_summary.json")
meta = S["meta"]; rows = S["rows"]
PRETTY = {
    "baseline":       r"Blind RAG (baseline)",
    "hybrid":         r"MACH-1 Hybrid",
    "probe_only":     r"MACH-1 Probe-Only",
    "static":         r"MACH-1 Static-$\alpha$",
    "no_gating":      r"MACH-1 NoGating",
    "no_contrastive": r"MACH-1 NoContrastive",
}
out = []
out.append(r"\begin{table}[htbp]")
out.append(rf"\caption{{Six-condition paired ablation on \textbf{{GPT-2}} (124M, layer {meta['target_layer']}/12), $n\!=\!{meta['n_queries']}$. All MACH-1 variants are statistically indistinguishable from the baseline ($p\!=\!1.000$ for every comparison; $\Delta\leq+0.20$\,pp $=$ 1 query out of 500). The non-instruction-tuned 124M model produces no measurable response to mid-layer contrastive steering on this task.}}")
out.append(r"\label{tab:gpt2-ablation}")
out.append(r"\centering\renewcommand{\arraystretch}{1.15}")
out.append(r"\begin{tabular}{l r r r r}")
out.append(r"\toprule")
out.append(r"Condition & EM (\%) & 95\% CI & $\Delta$ (pp) & Perm $p$ \\")
out.append(r"\midrule")
for r in rows:
    em = 100*r["em"]; lo = 100*r["ci_lo"]; hi = 100*r["ci_hi"]
    d = r["delta_pp"]; p = r["perm_p"]
    d_str = "---" if d is None else f"{d:+.2f}"
    p_str = fmt_p(p) + sig(p)
    out.append(f"{PRETTY[r['condition']]} & {em:.2f} & [{lo:.2f}, {hi:.2f}] & {d_str} & {p_str} \\\\")
out.append(r"\bottomrule\end{tabular}\end{table}")
Path("gpt2_ablation_table.tex").write_text("\n".join(out)+"\n")
print("Wrote gpt2_ablation_table.tex")
inline["gpt2_n"] = meta["n_queries"]
inline["gpt2_baseline_em"] = round(100*next(r["em"] for r in rows if r["condition"] == "baseline"), 2)
inline["gpt2_hybrid_em"] = round(100*next(r["em"] for r in rows if r["condition"] == "hybrid"), 2)

# ---------- 5. Three-panel figure --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
ax_l, ax_a, ax_h = axes

# Layer sweep panel
S = load("statistical_summary_layer_sweep.json"); rows = S["rows"]
xs = ["base"] + [str(r["layer"]) for r in rows[1:]]
ys = [100*r["em"] for r in rows]
err_lo = [100*r["em"] - 100*r["ci_lo"] for r in rows]
err_hi = [100*r["ci_hi"] - 100*r["em"] for r in rows]
colors_l = ["#34495E"] + ["#1ABC9C" if (r["delta_pp"] or 0) > 0 else "#E74C3C" if (r["delta_pp"] or 0) < 0 else "#95A5A6" for r in rows[1:]]
ax_l.bar(range(len(rows)), ys, color=colors_l, alpha=0.85, edgecolor="black", linewidth=0.7)
ax_l.errorbar(range(len(rows)), ys, yerr=[err_lo, err_hi], fmt="none", capsize=3, color="black", linewidth=1.2)
ax_l.set_xticks(range(len(rows))); ax_l.set_xticklabels(xs, fontsize=9)
ax_l.set_xlabel(f"Layer (of {S['meta']['num_layers']})"); ax_l.set_ylabel("Exact Match (%)")
ax_l.set_title(f"Layer-position sweep, α={S['meta']['alpha']} (n={S['meta']['n_queries']})", fontsize=10)
ax_l.grid(axis="y", alpha=0.3); ax_l.set_ylim([0, max(ys)+max(err_hi)+1.5])

# Alpha sweep panel
S = load("statistical_summary_llama3_v3.json"); rows = S["rows"]
xs = [r["alpha"] for r in rows]
ys = [100*r["em"] for r in rows]
err_lo = [100*r["em"] - 100*r["ci_lo"] for r in rows]
err_hi = [100*r["ci_hi"] - 100*r["em"] for r in rows]
ax_a.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o-", color="#C0392B", capsize=4, markersize=8, linewidth=1.5)
ax_a.axhline(y=100*rows[0]["em"], color="gray", linestyle="--", alpha=0.5, label="baseline")
ax_a.set_xlabel("Steering coefficient α (unit-direction)"); ax_a.set_ylabel("Exact Match (%)")
ax_a.set_title(f"Mid-layer α-sweep, ℓ={S['meta']['target_layer']} (n={S['meta']['n_queries']})", fontsize=10)
ax_a.grid(alpha=0.3); ax_a.legend(loc="lower left", fontsize=9)

# Layer-7 validation panel (HEADLINE)
S = load("statistical_summary_layer7.json"); rows = S["rows"]; meta = S["meta"]
xs = [r["alpha"] for r in rows]
ys = [100*r["em"] for r in rows]
err_lo = [100*r["em"] - 100*r["ci_lo"] for r in rows]
err_hi = [100*r["ci_hi"] - 100*r["em"] for r in rows]
ax_h.errorbar(xs, ys, yerr=[err_lo, err_hi], fmt="o-", color="#16A085", capsize=4, markersize=8, linewidth=1.5)
ax_h.axhline(y=100*rows[0]["em"], color="gray", linestyle="--", alpha=0.5, label="baseline")
hp = next(r for r in rows if abs(r["alpha"]-2.0) < 1e-9)
ax_h.annotate(f"+{hp['delta_pp']:.2f}pp\np={hp['perm_p']:.3f}", xy=(2, 100*hp["em"]), xytext=(2.4, 100*hp["em"]+1.4),
              fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="#16A085", lw=1.2))
ax_h.set_xlabel("Steering coefficient α (unit-direction)"); ax_h.set_ylabel("Exact Match (%)")
ax_h.set_title(f"HEADLINE: Layer-7 α-sweep (n={meta['n_queries']})", fontsize=10, fontweight="bold")
ax_h.grid(alpha=0.3); ax_h.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig("performance_metrics.png", dpi=140, bbox_inches="tight")
print("Wrote performance_metrics.png")
plt.close()

# Focused headline figure
fig, ax = plt.subplots(figsize=(7, 4.5))
xs = [0, 1]
ys = [100*BASE["em"], 100*HEADLINE["em"]]
errs = [
    [100*BASE["em"] - 100*BASE["ci_lo"], 100*HEADLINE["em"] - 100*HEADLINE["ci_lo"]],
    [100*BASE["ci_hi"] - 100*BASE["em"], 100*HEADLINE["ci_hi"] - 100*HEADLINE["em"]],
]
ax.bar(xs, ys, color=["#34495E", "#16A085"], alpha=0.9, edgecolor="black", linewidth=0.8, width=0.55)
ax.errorbar(xs, ys, yerr=errs, fmt="none", capsize=6, color="black", linewidth=1.5)
ax.set_xticks(xs); ax.set_xticklabels([f"Baseline\n(α=0)", f"MACH-1\n(ℓ={meta['target_layer']}, α=2)"], fontsize=11)
ax.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
delta = HEADLINE["em"] - BASE["em"]
delta_rel = 100*delta/BASE["em"] if BASE["em"] > 0 else 0
ax.set_title(f"Headline (Llama-3.2-3B, n={meta['n_queries']} paired): "
             f"Δ={100*delta:+.2f}pp ({delta_rel:+.1f}% rel.)  "
             f"paired-perm p={HEADLINE['perm_p']:.3f}", fontsize=10)
for x, y in zip(xs, ys):
    ax.text(x, y + 0.4, f"{y:.2f}%", ha="center", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("relative_improvement.png", dpi=140, bbox_inches="tight")
print("Wrote relative_improvement.png")
plt.close()

# ---------- Inline numbers ---------------------------------------------------
Path("inline_numbers.json").write_text(json.dumps(inline, indent=1))
print("\nInline numbers:")
for k, v in inline.items(): print(f"  {k}: {v}")
