# PoGS distance — how to finish the analysis (handoff)

Resume note for the PoGS (Potential-minimizing Geodesic Search) feasibility-distance task. If the
chat is closed before the Bury jobs end, this is everything needed to pick it up.

## What PoGS is here
PoGS distance = ambient **chord length** of an ADAM-optimised latent path between a parent `z` and
a mutant `z'`, with the activity potential switched off (`lambda = 0`), so it is a pure **geodesic
distance** (no metric `G` anywhere). Energy minimised:
`E = sum ||Dec(z_{k+1})-Dec(z_k)||^2 + lambda*sum Phi(X_k) + mu*sum ||z_{k+1}-z_k||^2`.
Implementation: `_pogs.py` (`pogs_distance_to_parent`). Defaults `N=8, mu=1e-2, lr=1e-3, 300-500
steps`. Methodology is written in the thesis: `chapters/models.tex`, paragraph `\label{par:pogs}`.

## Bury jobs (submitted from /home/kjurasz/pep-compass, branch rl_trials)
| Job | Script / wrapper | Output (in `results/data/all_in/_cache/` unless noted) | Feeds |
|-----|------------------|--------------------------------------------------------|-------|
| 7396 | `_exp_pogs_distance_cache.py` (`PGS_WHICH=rq5`) via `scripts/pogs_distance_gpu.sh` | `_thesis_rq5_distances_pogs.parquet` (385 pep, 138k cand) | **TANDEM-A table** |
| 7397 | `_exp_pogs_distance_cache.py` (`PGS_WHICH=mutangplus`) | `_thesis_mutangplus_pogs.parquet` (845 pep, 195k cand) | MUTANG+ fixed-count sanity |
| 7398 → full | `_exp_mutangplus_pogs.py` via `scripts/mutangplus_pogs_gpu.sh` | `rq5_mutangplus_pogs.pdf` (in `results/rq5_figs/`) + `_thesis_mutangplus_pogs_summary.csv` | **MUTANG+ tau-curve table** |

7398 is only a 5-parent smoke (`MPE_LIMIT=5`); after it passes, submit the full run:
`sbatch scripts/mutangplus_pogs_gpu.sh` (no MPE_LIMIT).

Check progress: `ssh kjurasz@bury.mimuw.edu.pl 'cd pep-compass && squeue -u kjurasz && tail logs/pogs_dist_7396.out'`

## What each output column is for
`_thesis_rq5_distances_pogs.parquet` (the important one) — the 385-peptide RQ3 candidate set, now
with a `dist_pogs` column **plus** the pre-existing `score_A_onehot`, `score_A_diff` (TANDEM-A
potentials), `dist_geo`, `dist_eucl`, `dist_maha`, `n_mut`, `pep`, `seq`. We test whether the
TANDEM-A coherence potential ranks candidates by PoGS-geodesic proximity (`-dist_pogs`), exactly
as the thesis already did for `-dist_geo`.

`_thesis_mutangplus_pogs.parquet` — same idea on the 845-peptide MUTANG+ set; used only for the
"does the geodesic still track Euclidean within a fixed mutation count" sanity.

`_thesis_mutangplus_pogs_summary.csv` / `rq5_mutangplus_pogs.pdf` — the MUTANG+ whitened-viability
filter tau-curve scored by `dist_pogs`: mean relative `dist_pogs` of the retained set and mean
positions changed vs threshold `tau`, for one-hot/diff x argmax/product.

## Steps to finish (each table goes in when its data lands)

### Table 1 — TANDEM-A under PoGS  [DONE: in results.tex as tab:rq5-pogs + fig:rq5-tandem-pogs]
`fig_dist_pogs_tandem.py` now does **soft-max top-$p$** selection (keep the smallest candidate set
holding the top-$p$ probability mass of the potential), dense grid step 0.025, single panel, and
marks the optimal operating point (min distance). Result: one-hot $p^\ast\approx0.03$ (-5.8%), diff
$p^\ast\approx0.48$ (-6.2%); fixed-count Spearman still $\approx0$, so the ~6% is the count confound.
To regenerate:
```
scp kjurasz@bury.mimuw.edu.pl:/home/kjurasz/pep-compass/results/data/all_in/_cache/_thesis_rq5_distances_pogs.parquet \
    results/data/all_in/_cache/
cd analysis/scripts/thesis_figures && python fig_dist_pogs_tandem.py   # uses local .venv
```
This prints the median fixed-count Spearman(TANDEM-A potential, `-dist_pogs`) for one-hot and diff
at `n_mut = 1..4` (with `-dist_geo` alongside) and writes `rq5_tandem_pogs.pdf`. Put those numbers
in a new table `tab:rq5-pogs` in `chapters/results.tex`, **right after `tab:rq5` /
`tab:rq5-fixedcount`** (sec:rq5-feasibility). One sentence: PoGS reproduces the graph-geodesic null
for TANDEM-A coherence (or whatever the numbers show — keep TANDEM framed constructively per the
`pepcompass-thesis` skill).

### Figure 5.17 — score-family alignment under PoGS  [DONE: fig:rq5-scorefamily-alignment-pogs]
`fig_rq5_scorefamily_pogs.py` (local): the analogue of fig 5.16 vs $-d_{\mathrm{PoGS}}$. Reuses the
saved 8-transform family scores in `_thesis_rq7_scores_alt.parquet` and attaches `dist_pogs` by an
exact merge on `(pep, n_mut, dist_geo)` (dist_geo is a unique per-candidate key; 100% match). No
bury re-run. Result: every cell in $[-0.02,+0.06]$ (max B/hinge/mean). Writes
`rq5_scorefamily_alignment_pogs.pdf`. 5.16 (d_geo) is kept for side-by-side comparison.

### Table 2 — MUTANG+ under PoGS (when the full `_exp_mutangplus_pogs.py` run ends)
```
scp kjurasz@bury.mimuw.edu.pl:/home/kjurasz/pep-compass/results/rq5_figs/rq5_mutangplus_pogs.pdf \
    "C:/Users/Karol/Desktop/Magisterka/magisterka-karola-txt/figures/"
scp kjurasz@bury.mimuw.edu.pl:/home/kjurasz/pep-compass/results/data/all_in/_cache/_thesis_mutangplus_pogs_summary.csv \
    results/data/all_in/_cache/
```
Read the summary CSV (`enc, method, tau, rel, pos`) and add `tab:rq5-mutangplus-pogs` in
`results.tex` **right after `tab:rq5-mutangplus`** (sec:rq5-mutangplus), mirroring its layout:
positions changed and, in parentheses, mean `dist_pogs` relative to the full MUTANG set, at the
selected `tau`. Reference `rq5_mutangplus_pogs.pdf`.

### Verify
`cd magisterka-karola-txt && latexmk -pdf -interaction=nonstopmode main.tex` — confirm the two new
tables + the `par:pogs` methodology compile and `kingma2015adam` resolves.

## Ready-to-fill table LaTeX (drop into results.tex, replace `??`)

`fig_dist_pogs_tandem.py` was dry-run-validated on the 5-parent smoke parquet: all columns present,
`dist_pogs` 100% finite, the diff encoding is correctly undefined at `n=1`. Fill `??` from the full
run's stdout.

**Table 1 — after `tab:rq5` / `tab:rq5-fixedcount` (sec:rq5-feasibility):**
```latex
\begin{table}[H]
  \centering\small
  \caption[TANDEM-A alignment with the PoGS geodesic]{Median per-peptide
  Spearman of the TANDEM-A coherence potential with feasibility within a fixed mutation count,
  over the $385$-peptide RQ3 set, under the PoGS geodesic $-d_{\mathrm{PoGS}}$ ($\lambda=0$) and
  the graph geodesic $-d_{\mathrm{geo}}$. The diff encoding is constant within single mutants
  (undefined for $n{=}1$). As under $d_{\mathrm{geo}}$, TANDEM-A coherence stays $\approx 0$.}
  \label{tab:rq5-pogs}
  \begin{tabular}{lrrrr}
    \toprule
    TANDEM-A coherence vs distance & $n{=}1$ & $n{=}2$ & $n{=}3$ & $n{=}4$ \\
    \midrule
    one-hot, $-d_{\mathrm{PoGS}}$ & ?? & ?? & ?? & ?? \\
    diff, $-d_{\mathrm{PoGS}}$    & --- & ?? & ?? & ?? \\
    one-hot, $-d_{\mathrm{geo}}$  & ?? & ?? & ?? & ?? \\
    diff, $-d_{\mathrm{geo}}$     & --- & ?? & ?? & ?? \\
    \bottomrule
  \end{tabular}
\end{table}
```

**Table 2 — after `tab:rq5-mutangplus` (sec:rq5-mutangplus); rows = `tau` from the summary CSV.**
Note: PoGS is an **absolute** distance, so report absolute mean `d_pogs` (column `dabs` in the CSV),
NOT a MUTANG=1 ratio. The script prints `full-MUTANG absolute mean d_pogs` as the reference value
(put it in the caption).
```latex
\begin{table}[H]
  \centering\small
  \caption[MUTANG+ under the PoGS geodesic]{MUTANG+ whitened viability filter at selected
  thresholds $\tau$, scored by the PoGS geodesic distance ($\lambda=0$): mean number of positions
  changed and, in parentheses, the absolute mean $d_{\mathrm{PoGS}}$ of the retained set (ambient
  chord units; the full MUTANG set averages $d_{\mathrm{PoGS}}={??}$), for the two encodings
  $\times$ two selections. Because PoGS is an absolute distance, no per-peptide MUTANG${}=1$
  normalisation is needed (contrast \autoref{tab:rq5-mutangplus} under the cloud-relative
  $d_{\mathrm{geo}}$).}
  \label{tab:rq5-mutangplus-pogs}
  \begin{tabular}{lcccc}
    \toprule
    $\tau$ & one-hot / argmax & one-hot / product & diff / argmax & diff / product \\
           & pos ($d_{\mathrm{PoGS}}$) & pos ($d_{\mathrm{PoGS}}$) & pos ($d_{\mathrm{PoGS}}$) & pos ($d_{\mathrm{PoGS}}$) \\
    \midrule
    $-0.2$ & ?? & ?? & ?? & ?? \\
    $0.0$  & ?? & ?? & ?? & ?? \\
    % ... remaining tau rows from _thesis_mutangplus_pogs_summary.csv (cols: enc,method,tau,dabs,pos) ...
    \bottomrule
  \end{tabular}
\end{table}
```

## Files created for this task
- `_pogs.py`, `_exp_pogs_distance_cache.py`, `_exp_mutangplus_pogs.py`, `fig_dist_pogs_tandem.py`,
  `_smoke_pogs.py` (local), `POGS_CONTINUE.md` (this file) — in `analysis/scripts/thesis_figures/`
- `scripts/pogs_distance_gpu.sh`, `scripts/mutangplus_pogs_gpu.sh`
- notebook `analysis/notebooks/wildeast/mutang/202601-geodesics-biharmonics-pogs.ipynb`
- thesis: `chapters/models.tex` (`par:pogs`), `bibliography.bib` (`kingma2015adam`)
- still TODO: the two `results.tex` tables above.
