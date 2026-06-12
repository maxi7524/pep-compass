---
name: pepcompass-thesis
description: Guides PepCompass thesis writing, figure regeneration, branch comparison, and Bury workflow. Use when working on magisterka-karola-txt, thesis chapters, MUTANG/TANDEM results, thesis figures, Bury jobs, or branches kjxpp/mutang_analysis_wip and rl_trials.
---

# PepCompass Thesis

## Quick Start

When the user asks about `magisterka-karola-txt`, thesis writing, thesis figures, MUTANG/TANDEM results, or Bury branch workflow:

1. Read `reference.md`.
2. Inspect the relevant thesis chapter or analysis script before editing.
3. Keep all claims aligned with `chapters/results.tex`, `chapters/conclusions.tex`, and current generated figures.
4. Preserve the formal scientific tone required by the thesis `CLAUDE.md`.

## Core Constraints

- Treat APEX activity as predicted MIC, not measured MIC.
- Do not overstate TANDEM. The current results show that pairwise similarity weighting does not improve latent-feasibility alignment after controlling for Hamming distance.
- Frame the TANDEM null result constructively: feasibility depends more on net latent displacement magnitude than on pairwise tangent-direction coherence.
- Keep sequences one per line when listing peptide sequences.
- Preserve LaTeX labels, citations, and cross-references unless the user asks for structural edits.

## Thesis Workflow

- Thesis repo: `C:/Users/Karol/Desktop/Magisterka/magisterka-karola-txt`.
- PepCompass repo: `C:/Users/Karol/Desktop/PepCompass/pep-compass`.
- Thesis figures are generated from `analysis/scripts/thesis_figures/`.
- The current local figure output path is configured in `analysis/scripts/thesis_figures/_common.py`.
- Before editing thesis text, check whether `introduction.tex`, `results.tex`, `abstract.tex`, and `conclusions.tex` agree on the same claim.

## Bury Workflow

- SSH target: `kjurasz@bury.mimuw.edu.pl`.
- Expected remote repo: `/home/kjurasz/pep-compass`.
- Verify branches with:

```bash
ssh kjurasz@bury.mimuw.edu.pl 'cd /home/kjurasz/pep-compass && git status --short --branch && git branch -a | egrep "kjxpp/mutang_analysis_wip|rl_trials|rl_trails"'
```

- Existing GPU job scripts assume `/home/kjurasz/pep-compass/.venv/bin/python` and should refuse CPU fallback.

## Drafting Prompts

Use this prompt shape for thesis edits:

```text
Use the pepcompass-thesis skill. Edit only the requested LaTeX section. Keep formal scientific language. Reconcile every claim with results.tex and conclusions.tex. Do not overstate TANDEM: pairwise similarity did not improve latent feasibility; the constructive finding is that net latent displacement matters.
```

Use this prompt shape for branch comparison:

```text
Compare branches kjxpp/mutang_analysis_wip and rl_trials for thesis-relevant changes only: analysis scripts, notebooks, generated figure assumptions, and LaTeX-facing results. Do not edit files. Return a risk-ranked summary and exact files to inspect.
```
