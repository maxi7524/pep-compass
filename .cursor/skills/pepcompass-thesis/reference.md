# PepCompass Thesis Reference

## Stable Paths

- PepCompass repo: `C:/Users/Karol/Desktop/PepCompass/pep-compass`.
- Thesis repo: `C:/Users/Karol/Desktop/Magisterka/magisterka-karola-txt`.
- Thesis chapters: `chapters/introduction.tex`, `chapters/models.tex`, `chapters/data.tex`, `chapters/results.tex`, `chapters/conclusions.tex`.
- Thesis notes: `notes/pepcompass_summary.md`, `notes/contributions.md`.
- Thesis figures: `figures/`.
- Figure scripts: `analysis/scripts/thesis_figures/`.
- Bury repo: `/home/kjurasz/pep-compass`.

## Branches

- Main thesis-analysis branch: `kjxpp/mutang_analysis_wip`.
- Comparison branch: `rl_trials`.
- If the user writes `rl_trails`, verify whether this is a typo before creating or switching branches.

## Thesis Narrative

- Contribution 1: empirical and theoretical analysis of MUTANG as a mutation-generation strategy.
- Contribution 2: TANDEM, a potentials-based extension of MUTANG, plus MUTANG+ as a similarity-driven viability filter.
- Current empirical framing is qualified: TANDEM is a useful diagnostic extension, but the pairwise-similarity potential does not improve latent-feasibility alignment.
- Constructive conclusion: latent feasibility depends on net latent displacement magnitude rather than pairwise tangent-direction coherence.

## Research Questions

- RQ1: Does MUTANG produce mutations that reflect biologically grounded distribution patterns?
- RQ2: What are MUTANG's structural weaknesses?
- RQ3: Does potential-based weighting of the single-position mutation similarity matrix improve alignment with latent-geometry feasibility?
- RQ4: Does TANDEM show superior alignment with the feasible mutation space compared to MUTANG?

## Key Results To Preserve

- APEX-predicted `Delta log MIC` signatures recover a charge-driven, lysine-centred, Gram-agnostic AMP activity pattern.
- MUTANG proposal frequencies are essentially uncorrelated with biological benefit.
- MUTANG weaknesses: Cartesian-product explosion, threshold instability, and finite-difference Jacobian sensitivity.
- TANDEM pairwise potentials are near-null after controlling for mutation count, across geodesic, first-order pullback, and Euclidean distances.
- TANDEM-induced AMP-BLOSUM agreement is modest and almost unchanged by selection level.
- MUTANG+ gives a smooth threshold knob but is only a weak feasibility selector.

## Figure And Analysis Scripts

- `fig_rq1_mutang_proposal.py`: MUTANG proposal signatures.
- `fig_rq1_biological_signature.py`: `Delta log MIC` biological signatures.
- `fig_rq1_wasserstein.py`: Wasserstein clustering figures and cluster table stdout.
- `fig_rq1_proposal_vs_benefit.py`: proposal frequency versus activity benefit.
- `fig_rq2_stability.py`: MUTANG weakness and stability figures.
- `_common.py`: shared plotting helpers and thesis output directory.

## Recommended Helper Scripts To Add Later

- `scripts/dev/check_bury_branches.sh`: verify remote branch status and dirty tree on Bury.
- `scripts/dev/worktree_branch.sh`: create clean sibling worktrees for branch comparisons.
- `scripts/thesis/run_figures.py`: run selected thesis figure scripts with configurable output directory.
- `scripts/thesis/check_thesis_consistency.py`: scan TODO/FIXME, stale RQ references, missing figures, and broken labels.
- `scripts/thesis/branch_diff_summary.py`: summarize thesis-relevant differences between analysis branches.

## Common Consistency Checks

- Compare claims in `chapters/introduction.tex` against `chapters/abstract.tex` and `chapters/conclusions.tex`; introduction may be more optimistic than the final null-result framing.
- Check `notes/contributions.md` before using it as source material; it may contain older RQ numbering or stale TANDEM expectations.
- Do not convert the TANDEM null result into a positive improvement claim.
