# Submission Versions

All four PDFs build clean from the same `taxonomy_paper.tex` source, adapted to each venue's style.

| Dir | PDF | Pages | Format | Status |
|---|---|---|---|---|
| `arxiv/` | `taxonomy_paper.pdf` | 11 | ACL style, **de-anonymized**, no `[review]` | Ready to post |
| `arr/` | `taxonomy_paper.pdf` | 11 | ACL style, **anonymized** (`[review]` flag) | Ready to submit |
| `tmlr/` | `taxonomy_paper.pdf` | 12 | TMLR single-column, anonymized | Ready to submit |
| `colm/` | `taxonomy_paper.pdf` | 12 | COLM 2026 style, anonymized | Ready to submit |

## Recommended order

### 1. arXiv (DO THIS FIRST — establishes priority on the LOO + SAE finding)

**Pre-check before posting:**
- Verify the GitHub URL in the paper (Reproducibility section) matches a real public repo you control:
  - Current placeholder: `https://github.com/real1900/steering-taxonomy`
  - If repo doesn't exist yet, create it before submitting OR edit the URL to point to the existing `o1-repe-rag-research` repo with a path
- Verify author name/affiliation/email at top of paper match what you want shown publicly

**Submit:**
1. Go to https://arxiv.org/submit (log in, or create account with your institutional email if you have one)
2. Category: **cs.CL** (primary) + **cs.LG** (cross-list)
3. Upload `submissions/arxiv/`'s contents:
   - `taxonomy_paper.tex`
   - `refs.bib`
   - `acl_style/` directory
   - `cross_model_4models.pdf`, `layer_sweep_heatmap.pdf`, `taxonomy_figure.pdf`
4. Fill in: title, authors, abstract (paste from paper §abstract), comments (e.g., "11 pages, 3 figures"), license (CC-BY-4.0 standard)
5. Submit → live on arXiv next business-day morning ET

### 2. ARR (OpenReview-hosted, funnel to ACL/EMNLP/NAACL Findings)

**Pre-check:**
- Anonymization already verified (no name/JHU/email leaks in tex or refs.bib)
- GitHub URL is the anonymized `https://github.com/anonymous/steering-taxonomy` — keep this for ARR

**Submit:**
1. Go to https://openreview.net, search for "ACL Rolling Review" (ARR)
2. Find the current submission cycle (cycles open monthly; check the official ARR page for deadline)
3. Upload `submissions/arr/taxonomy_paper.pdf` (just the PDF — source not required at submission)
4. Fill in: abstract, keywords (e.g., "activation steering, representation engineering, mechanistic interpretability")
5. Complete the **Responsible NLP Research Checklist** (~30 yes/no questions, takes ~20 min)
6. Review cycle: ~6 weeks. Then you "commit" the reviewed paper to ACL / EMNLP / NAACL Findings based on what deadline lines up next.

### 3. TMLR (highest accept probability, single-track journal)

**Pre-check:**
- TMLR has no page limit — the 12-page version is fine. Could potentially expand further with more detail in appendix.
- Anonymized via default `tmlr.sty` settings

**Submit:**
1. Go to https://jmlr.org/tmlr, click "Submit"
2. OpenReview workflow (same account as ARR if you have one)
3. Upload `submissions/tmlr/taxonomy_paper.pdf`
4. Fill in: abstract, keywords, suggested action editor (optional — can pick someone whose work overlaps; check editor list at https://jmlr.org/tmlr/editorial-board.html)
5. Review: ~1 week to action editor; ~6 weeks for reviewer reports; ~2 month total decision

### 4. COLM 2026

**Pre-check:**
- Anonymized via default `colm2026_conference.sty` (no `[final]` option)
- 12 pages currently — COLM allows 10 main + unlimited appendix; need to verify page count fits 10 main when references and appendix are separated (looks like it does since this PDF includes refs + appendix)

**Submit:**
1. Watch https://colmweb.org for the 2026 CFP (typically opens Feb–Mar, deadline in May)
2. OpenReview submission portal will be linked there
3. Upload `submissions/colm/taxonomy_paper.pdf`
4. Review: ~2 months; notification ~July

## Pre-submission checklist (apply to all anonymized versions)

Run before each submission:

```bash
# Search for name/affiliation/email leaks
grep -irnE "suleman|imdad|johns hopkins|jhu|me\.com|gmail|real1900" \
    submissions/<venue>/*.tex submissions/<venue>/refs.bib
# Should return nothing for anonymized versions (arr, tmlr, colm).
```

For arXiv only, verify the author block and GitHub URL look right.

## Estimated probabilities (my honest read)

| Venue | Accept probability |
|---|---|
| arXiv | 100% (no review) |
| TMLR | 85–90% (after one revision round) |
| ARR → ACL/EMNLP/NAACL Findings | 55–75% |
| COLM 2026 | 60–75% |
| ACL/EMNLP/NAACL Main | 35–50% (Findings is the safer aim) |

## What's in the paper that drives those probabilities

- 79.2% pooled out-of-sample LOO across 48 (model, task) pairs
- 75–83% cross-model CV per held-out model
- LRH-grounded formal derivation of the rule
- Probe baseline showing linear separability ≠ behavioral conditioning (r=−0.176 vs +0.436)
- Seed-stable Δ (max std 0.014 across 3 seeds × 7 tasks)
- SAE-decomposition mechanism on Gemma-Scope (r(n_features, σ) = +0.75)

What's NOT in it:
- Frontier-scale 70B+ verification (attempted, hit Apple Silicon memory walls; honestly flagged in §Limitations)
- LLM-as-judge scoring (uses substring/lexicon scorers; flagged in §Limitations)
