# Thesis workflow (Magisterka_Karola)

This file documents where the Master's thesis lives, how it is wired to the
`pep-compass` analysis pipeline, and how to compile and edit it without breaking
the build.

## Locations

| Resource                 | Path                                                                         |
|--------------------------|------------------------------------------------------------------------------|
| Thesis repo (LaTeX)      | `C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt`                    |
| Thesis figures directory | `C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt\figures`            |
| Thesis style class       | `C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt\pracamgr.cls`       |
| Build artefacts          | `C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt\build\` and root    |
| PepCompass figure scripts| `analysis\scripts\thesis_figures\` (this repo)                               |
| Figure-pipeline helper   | `analysis\scripts\thesis_figures\_common.py` (writes directly to thesis dir) |
| Local Python environment | `.venv\Scripts\python.exe`                                                   |
| Cache used by RQ scripts | `results\data\all_in\_cache\`                                                |
| Project rules / skills   | `.cursor\rules\pepcompass.mdc`, `.cursor\skills\pepcompass-thesis\`          |
| Thesis-side rules        | `magisterka-karola-txt\.cursor\rules\thesis-agent.mdc`                       |
| Thesis writing style     | `magisterka-karola-txt\CLAUDE.md`                                            |

## Repository layout (thesis side)

```
magisterka-karola-txt/
  main.tex                    # master document (\include's the chapters)
  pracamgr.cls                # UW magisterska class (DO NOT EDIT)
  bibliography.bib            # bibliography (BibTeX)
  chapters/
    abstract.tex              # streszczenie i abstract (PL + EN)
    introduction.tex          # motivation + RQs (rq:signatures, rq:weaknesses, ...)
    background.tex            # AMP background
    models.tex                # HydrAMP + APEX + decoder geometry
    data.tex                  # datasets, tools, methods (incl. tool-rankcorr)
    results.tex               # all RQ sections (5.x)
    conclusions.tex           # answers RQ-by-RQ
  figures/                    # PDFs produced by pep-compass scripts go here
  notes/                      # scratch summaries, NOT compiled into the thesis
    contributions.md
    pepcompass_summary.md
  CLAUDE.md                   # short content-style notes
  .cursor/rules/thesis-agent.mdc
```

## Compile

The thesis uses TeX Live 2025 with `latexmk`.

```powershell
cd C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For a forced full rebuild (useful after editing labels or figures):

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error -g main.tex
```

To clean intermediate files (keeps `main.pdf`):

```powershell
latexmk -c
```

The successful build prints `Output written on main.pdf (NN pages, ...)`. At the
time of writing the document is 71 pages.

## Figure pipeline (pep-compass --> thesis)

`analysis/scripts/thesis_figures/_common.py` hard-codes the output directory:

```python
THESIS_FIG = Path(r"C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt\figures")
```

Every script in `analysis/scripts/thesis_figures/` that calls `save(fig, "name.pdf")`
writes a vector PDF directly into the thesis `figures/` folder, so re-running a
figure script is the only step needed to update a figure in the thesis. The
thesis side does not need a git pull from `pep-compass`; the PDF just appears.

Typical workflow to refresh a figure:

```powershell
cd C:\Users\Karol\Desktop\PepCompass\pep-compass
.\.venv\Scripts\python.exe analysis\scripts\thesis_figures\fig_rq1_proposal_vs_benefit.py
.\.venv\Scripts\python.exe analysis\scripts\thesis_figures\fig_rq1_proposal_vs_benefit_diagnostics.py
```

then recompile the thesis. Each figure script reads from
`results\data\all_in\_cache\*.parquet`. If the cache is missing, run the
upstream signature/proposal scripts first (`fig_rq1_mutang_proposal.py`,
`fig_rq1_biological_signature.py`), or set `REGEN_COUNTS=1` to recount from the
14 GB `peptides_mutants_apex.csv`.

## Editing rules (must follow)

These are enforced by `magisterka-karola-txt\CLAUDE.md` and
`magisterka-karola-txt\.cursor\rules\thesis-agent.mdc`:

1. **Formal scientific English**. No conversational filler. Avoid "we just",
   "essentially", "blind"-style metaphors unless they are precise.
2. **One sequence per line** when listing peptide sequences in text/tables.
3. **Preserve notation**. Use the PepCompass conventions:
   - decoder `Dec`, pullback metric `G_{Dec}(z)=J_{\hat{Dec}}(z)^\top J_{\hat{Dec}}(z)`
   - stable manifold `M_z^\kappa`, ambient tangent basis `U^\kappa(z)`
   - mutation threshold `\theta_{mut}`, eigenvalue/SV threshold `\kappa`
   - production setting `\kappa=10^{-3}`, `\theta_{mut}=10^{-6}`, `\varepsilon=5\times10^{-2}`
4. **Do not reintroduce retired symbols**: `\Sigma`, `\mathcal{A}_i`,
   `\mathcal{Q}`, `\omega`, `s^{(0)}`.
5. **Two kappa conventions**: the upstream paper thresholds *squared* singular
   values, the implementation thresholds singular values directly. State which
   you mean whenever you write a number.
6. **APEX = predicted MIC**, never measured. Always qualify activity claims as
   APEX-predicted.
7. **TANDEM**: do not overstate. The thesis position is that pairwise tangent
   coherence does NOT improve latent-feasibility alignment after Hamming
   control; net latent displacement does. `abstract.tex`, `results.tex` and
   `conclusions.tex` are aligned; if `introduction.tex` reads optimistic, fix
   it there, not in the results.
8. **Preserve LaTeX labels and cross-references**. Do not rename a `\label{}`
   without updating every `\autoref{}` / `\ref{}` that targets it.
9. **Chapter order in `main.tex` is fixed**: Introduction, Antimicrobial
   peptides, Models, Data and Experimental Setup, Results, Conclusions.
10. **Do not vendor large datasets or compiled PDFs into the thesis repo.** The
    only artefacts allowed under `figures/` are the PDFs/CSVs the thesis cites,
    plus the small `blosum_dbaasp.csv` reference.

## Common compile failures

| Symptom                                  | Fix                                                                 |
|------------------------------------------|---------------------------------------------------------------------|
| `Nothing to do for 'main.tex'`           | Pass `-g` to force a rebuild, or `latexmk -c` then build.            |
| `LaTeX Error: File 'foo.pdf' not found`  | Run the matching figure script in `analysis\scripts\thesis_figures`.|
| `Undefined control sequence \autoref`    | `hyperref` not loaded; do not strip the preamble in `main.tex`.      |
| `Reference X undefined`                  | Build twice (latexmk handles this automatically; if not, `-g`).      |
| `pdflatex: not found`                    | Add `C:\texlive\2025\bin\windows` to PATH.                           |
| Citations missing                        | Re-run `bibtex main` then pdflatex x2 (latexmk does this on `-g`).   |

## Where to put new analyses

- Drop a new `fig_rqX_*.py` into `analysis\scripts\thesis_figures\`. It will
  automatically write into the thesis `figures/` directory via `_common.save`.
- Persist intermediate matrices into `results\data\all_in\_cache\` as
  `_thesis_*.parquet` / `_thesis_*.csv` so downstream scripts can reuse them.
- Reference the new figure from `chapters\results.tex` with
  `\includegraphics{name.pdf}` + a `\label{fig:...}` so the cross-refs stay
  intact.
- Companion diagnostic scripts (extra robustness checks, sensitivity sweeps)
  follow the same naming pattern, e.g.
  `fig_rq1_proposal_vs_benefit_diagnostics.py`.

## Sanity-check after edits

After any text edit in `chapters/`:

```powershell
cd C:\Users\Karol\Desktop\Magisterka\magisterka-karola-txt
latexmk -pdf -interaction=nonstopmode -halt-on-error -g main.tex
```

After any figure-script change in this repo:

```powershell
cd C:\Users\Karol\Desktop\PepCompass\pep-compass
.\.venv\Scripts\python.exe analysis\scripts\thesis_figures\fig_xxx.py
# then recompile the thesis as above
```

If the LaTeX log shows `Overfull \hbox`, that is a warning, not an error; the
PDF is still produced.
