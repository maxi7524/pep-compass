# Evaluating TANDEM as a feasibility *filter* (not a ranker)

Why the current null is partly a measurement artefact, and how to test the real claim. All metrics
read only cached `(potential, distance, n_mut, pep)` from `_thesis_rq5_distances_pogs.parquet`,
`_thesis_rq7_scores_alt.parquet`, `_thesis_mutangplus_pogs*` — **no new geometry**.

## 0. The reframe
Spearman measures global monotone *ranking* over all candidates: bulk-dominated, every pair weighted
equally, blind to tails. TANDEM's claim is a *filter*: cut the far/outlier mutants, keep the rest.
A good filter can have Spearman ~0. Evaluate the decision boundary and the tail, not the ranking.

Two invariants for every metric below:
1. **Within a fixed mutation count** (counts confound distance) — aggregate per `(pep, n_mut)`.
2. **vs a size- AND count-matched random baseline** — keep a random subset of the same size; does
   the potential add information beyond keeping fewer candidates? This is the decisive control.

## 1. Cumulative-density operating curve  [§2 — implement first]
Selection = smallest set holding the top-`p` of `softmax(potential)` (nucleus), `p` on the 0.025 grid.
Per `(pep, n_mut, p)` report tail stats of the kept set, not the mean:
- **CVaR@10%** = mean of the worst (largest-distance) 10% kept — the cleanest "did you cut the tail".
- **P90 / P95 distance**, **max distance**.
- **retention %** (cost axis).
Plot the median-over-groups CVaR@10% and P90 of the kept set AND of the matched-random subset vs
retention. A real filter drops the upper-quantile/CVaR curve far faster than the median, and below
random. The CVaR-vs-retention knee = the recommended cumulative-density operating point.

## 2. Lift over matched-random baseline  [§4 — implement first]
For each `p`, per group: `lift = metric(random size-k subset) - metric(TANDEM top-k)` for
metric in {CVaR@10%, P90, median}. Aggregate the per-group lift distribution: median + 95% bootstrap
CI, **Wilcoxon signed-rank** across peptides (paired kept-vs-random), and report the best-`p`
operating point. Random baseline by Monte-Carlo (R~100 subsets) so size is matched exactly.
Conclusion form: "TANDEM-diff removes X% more tail mass than random at the same retention, p=…".

## 3. Direct outlier detection  [§3 — follow-up]
Label "bad" per peptide = far-distance tail (top decile, or `d > median + 3 MAD`; robust, relative).
- **AUPRC / AUROC of `-potential` predicting "bad"** (insensitive to bulk ordering; AUPRC for rare bad).
- **Enrichment / Jaccard** of {bottom-q potential} vs {top-q distance} over random-expected.
- **Tail-mass gap** `P(d>tau | rejected) - P(d>tau | kept)`.

## 4. If a single correlation is still wanted  [optional]
Tail-weighted rank stat (bottom-weighted Kendall / AP-correlation), or Somers' D restricted to pairs
with `|Δd| > δ` (ignore near-ties Spearman over-penalises). Report as robustness only.

## 5. What "success" looks like
A defensible conclusion either way: e.g. *"TANDEM-diff does not rank candidates (Spearman ≈ 0), but
at its cumulative-density operating point it removes far/outlier mutants significantly better than
size-matched random selection (CVaR lift = …, Wilcoxon p = …) — a usable feasibility filter,"* or the
honest negative if the lift is within random. The matched-random baseline is what makes it honest
(guards against the size/count confound that inflated the earlier selection curve).

## Implementation status
- §1+§2 (curve + lift) → `fig_pogs_filter_eval.py` (local; reads `_thesis_rq5_distances_pogs.parquet`,
  potentials `score_A_onehot`/`score_A_diff`, distance `dist_pogs`, within `n_mut in {2,3,4}`).
- §3, §4-optional pending after reviewing §1/§2 conclusions.
