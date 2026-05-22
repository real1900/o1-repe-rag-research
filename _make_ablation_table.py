"""Generate ablation_table.tex from statistical_summary.json.

Run after the notebook has produced statistical_summary.json.
"""
import json
from pathlib import Path

with open("statistical_summary.json") as f:
    S = json.load(f)
rows = S["rows"]
meta = S["meta"]

PRETTY = {
    "baseline":       r"Blind RAG (baseline)",
    "hybrid":         r"\textbf{MACH-1 Hybrid (production)}",
    "probe_only":     r"MACH-1 Probe-Only",
    "static":         r"MACH-1 Static-$\alpha$",
    "no_gating":      r"MACH-1 NoGating",
    "no_contrastive": r"MACH-1 NoContrastive",
}

def fmt_p(p):
    if p is None:
        return "---"
    if p < 1e-4:
        return r"$<\!10^{-4}$"
    return f"{p:.4f}"

def sig_marker(p):
    if p is None:
        return ""
    if p < 0.001:
        return r"$^{\ast\ast\ast}$"
    if p < 0.01:
        return r"$^{\ast\ast}$"
    if p < 0.05:
        return r"$^{\ast}$"
    return ""

caption = (
    rf"Six-condition paired ablation on $n = {meta['n_queries']}$ HotpotQA queries "
    rf"(rows {meta['eval_start']}--{meta['eval_end']}, OOD from the MLP-probe "
    rf"tuning subset). EM = Exact Match. CI: 95\% bootstrap (10{{,}}000 resamples). "
    rf"$\Delta$: percentage-point difference vs.\ baseline (paired). "
    rf"$p$: paired permutation test (10{{,}}000 sign-flipped permutations) vs.\ baseline. "
    rf"Significance: $^{{\ast}}p<0.05$, $^{{\ast\ast}}p<0.01$, $^{{\ast\ast\ast}}p<0.001$. "
    rf"Heuristic-$\alpha$ fallback fired on {meta['heuristic_trips']}/{meta['n_queries']} "
    rf"({100*meta['heuristic_trips']/meta['n_queries']:.1f}\%) of Hybrid queries."
)

lines = []
lines.append(r"\begin{table*}[htbp]")
lines.append(r"\caption{" + caption + r"}")
lines.append(r"\label{tab:ablation}")
lines.append(r"\centering")
lines.append(r"\renewcommand{\arraystretch}{1.15}")
lines.append(r"\begin{tabular}{l r r r r r}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Condition} & "
    r"\textbf{EM (\%)} & "
    r"\textbf{95\% CI} & "
    r"\textbf{$\Delta$ (pp)} & "
    r"\textbf{Perm $p$} & "
    r"\textbf{Latency (ms)} \\"
)
lines.append(r"\midrule")

for r in rows:
    em_pct = 100 * r["em"]
    ci_lo = 100 * r["ci_lo"]
    ci_hi = 100 * r["ci_hi"]
    delta = r["delta_pp"]
    p = r["perm_p"]
    lat_ms = 1000 * r["lat_mean"]
    delta_str = "---" if delta is None else f"{delta:+.2f}"
    p_str = fmt_p(p) + sig_marker(p)
    name = PRETTY.get(r["condition"], r["condition"])
    lines.append(
        f"{name} & {em_pct:.2f} & [{ci_lo:.2f}, {ci_hi:.2f}] & {delta_str} & {p_str} & {lat_ms:.1f} \\\\"
    )

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

Path("ablation_table.tex").write_text("\n".join(lines) + "\n")
print(f"Wrote ablation_table.tex ({len(rows)} rows).")

# Also produce a key-values shim for the abstract / inline numbers
hybrid = next(r for r in rows if r["condition"] == "hybrid")
baseline = next(r for r in rows if r["condition"] == "baseline")
probe_only = next(r for r in rows if r["condition"] == "probe_only")
static = next(r for r in rows if r["condition"] == "static")

inline = {
    "n_queries": meta["n_queries"],
    "baseline_em_pct": round(100*baseline["em"], 2),
    "hybrid_em_pct": round(100*hybrid["em"], 2),
    "hybrid_delta_pp": round(hybrid["delta_pp"], 2),
    "hybrid_delta_rel": round(100*(hybrid["em"] - baseline["em"]) / baseline["em"], 1) if baseline["em"] > 0 else None,
    "hybrid_perm_p": hybrid["perm_p"],
    "probe_only_em_pct": round(100*probe_only["em"], 2),
    "probe_only_delta_pp": round(probe_only["delta_pp"], 2),
    "probe_only_perm_p": probe_only["perm_p"],
    "static_em_pct": round(100*static["em"], 2),
    "static_delta_pp": round(static["delta_pp"], 2),
    "static_perm_p": static["perm_p"],
    "baseline_lat_ms": round(1000*baseline["lat_mean"], 1),
    "hybrid_lat_ms": round(1000*hybrid["lat_mean"], 1),
    "heuristic_pct": round(100*meta["heuristic_trips"]/meta["n_queries"], 1),
}
with open("inline_numbers.json", "w") as f:
    json.dump(inline, f, indent=1)
print("Wrote inline_numbers.json:")
for k, v in inline.items():
    print(f"  {k}: {v}")
