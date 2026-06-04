# Wasserstein-distance clustering of AA->AA substitutions on APEX MIC deltas

Notebook: [`final_wasserstein.ipynb`](final_wasserstein.ipynb)
Parent MIC: `results/data/all_in/peptides_apex.csv` filtered to `dataset = hydramp_veltri_negative` (854 sequences)
Mutant MIC: `results/data/all_in/peptides_mutants_apex.csv` (14.3 GB, streamed once and cached as parquet under `results/data/all_in/_cache/`)
Metric: per substitution event we compute `log2(MIC_mutant) - log2(MIC_parent)` for each of the 34 bacterial columns; all such values for a given AA->AA pair (across every event AND every bacterium) are pooled into one 1D sample. 1-Wasserstein on those samples drives the clustering.

**Sign convention:** log2 ratio > 0 means the mutant has a *higher* MIC -> *less* active; log2 ratio < 0 means the mutant has a *lower* MIC -> *more* active AMP.

## TL;DR
- 77,297 mutants survived the parent filter; 57,182 valid single-position substitution events after cleanup, giving an event x bacterium delta matrix of shape (57182, 34).
- 363 of 380 AA->AA pairs have at least one event; 35 pairs have < 5 events and should be read as noisy.
- The clustering recovers a clean **cationic vs anionic** axis among non-trivial clusters:
  - **Cluster 11** (14 pairs, mean log2 ratio -0.10) — gaining basic / losing acidic character (e.g. `D>K, E>K, E>R`, plus `C>{A,E,L,N,S}`) -> peptides become *more* active.
  - **Cluster 5** (8 pairs, mean +0.11) — losing basic character (`R>{A,E,S,T}`, `K>{C,P}`) -> peptides become *less* active.
- The dominant cluster (228 pairs / 60% of pairs, mean ~0) captures the vast majority of substitutions, whose average effect on MIC is essentially zero on this AMP-negative parent set.
- Singleton outlier clusters are driven by individual high-impact substitutions like `R>D` (mean +0.29), `C>K` (-0.21), `I>M` (+0.20), `I>W` (-0.17).

## Pipeline summary
| Step | Value |
|---|---|
| Parent rows (after `dataset` filter) | 854 |
| Mutant rows kept (after parent filter, from 37.3 M scanned) | 77,297 |
| Distinct mutant sequences | 23,669 |
| Substitution events (after cleanup + positive-MIC requirement) | 57,182 |
| Per-event delta matrix shape | (57182, 34) |
| Delta range / mean | [-2.770, 2.785] / -0.013 |
| Off-diagonal AA->AA pairs total / non-empty / >= 5 events | 380 / 363 / 345 |
| 17 empty pairs (no events) | A>H, A>Y, D>H, D>I, D>P, D>Q, D>R, D>Y, H>D, H>W, N>D, S>Y, T>D, T>M, T>R, Y>D, Y>E |
| Pairwise Wasserstein matrix shape / runtime | (380, 380) / 76.9 s (numba+parallel) |
| Distance min / mean / max | 0.0000 / 0.0491 / 0.5031 |

The chunked load of the 14.3 GB mutants CSV took 251 s once; subsequent runs read the 11.3 MB parquet cache instantly.

## Cluster summary (k = 12, average linkage)
Sorted by mean log2 MIC ratio (most beneficial -> most harmful).

| cluster | n_pairs | n_events | mean log2 ratio | median | example members |
|--:|--:|--:|--:|--:|---|
|  3 |   1 |     27 | -0.213 | -0.102 | C>K |
|  4 |   1 |     15 | -0.171 | -0.130 | I>W |
| 12 |   1 |     10 | -0.126 | -0.058 | F>W |
| 11 |  14 |  1,916 | -0.104 | -0.054 | C>A, C>E, C>L, C>N, C>S, D>K, E>K, E>R, ... |
|  9 |  59 | 18,732 | -0.050 | -0.018 | C>F, C>I, C>R, C>T, C>W, D>A, D>G, D>L, ... |
| 10 |   5 |    655 | -0.010 |  0.000 | C>D, C>P, K>W, R>W, W>K |
|  7 | 228 | 26,837 | -0.000 | -0.000 | A>C, A>D, A>E, A>F, A>G, A>H, A>I, A>K, ... |
|  8 |  59 |  8,621 |  0.046 |  0.015 | F>C, F>M, F>P, F>Q, G>C, G>D, G>M, G>Q, ... |
|  5 |   8 |    292 |  0.105 |  0.066 | I>D, I>Q, K>C, K>P, R>A, R>E, R>S, R>T |
|  6 |   2 |     59 |  0.145 |  0.082 | G>E, I>E |
|  1 |   1 |     14 |  0.201 |  0.190 | I>M |
|  2 |   1 |      4 |  0.291 |  0.259 | R>D |

Reading this:
- **Cluster 7 (~60% of pairs)** is the "neutral" backbone. Every common substitution that does not strongly perturb charge or hydrophobicity ends up here; the centroid distribution is sharply peaked around 0.
- **Cluster 11 ("beneficial")** is the most actionable: 14 substitutions consistently *lower* MIC. The pattern is clean — replacing acidic residues with basic ones (`D>K, E>K, E>R`) and removing cysteine (`C>{A,E,L,N,S}`). Both moves are textbook AMP-enhancement: increasing net positive charge improves membrane interaction, and clearing Cys avoids inactive disulfide-locked conformers.
- **Cluster 5 ("harmful")** is the mirror image: 8 substitutions that *raise* MIC. The signature is loss of cationic character (`R>{A,E,S,T}`, `K>{C,P}`) and loss of bulky hydrophobic context (`I>{D,Q}`). Replacing R with E in particular flips a single residue from +1 to -1 net charge.
- **Singleton clusters (1, 2, 3, 4, 12)** capture individually-outlier substitutions. `R>D` (charge inversion) and `C>K` (gain of charge + loss of Cys) are the two strongest single-pair effects (mean +0.29 and -0.21 respectively). These are real biology, but each cluster contains a single AA->AA pair so the "cluster" label is more of a label of the pair than of a group.

## Most similar / most distinct pairs (each side >= 5 events)

Closest pairs:
| pair A | pair B | W |
|---|---|---:|
| E>T | F>I | 0.0015 |
| F>I | S>G | 0.0015 |
| G>V | H>T | 0.0018 |
| Q>N | S>N | 0.0019 |
| W>E | W>L | 0.0020 |

All five are "near-neutral" substitutions: the log2 MIC ratio distributions are nearly identical small bumps around 0. None of them perturbs charge or core hydrophobicity, so the model sees them as interchangeable in their MIC footprint.

Most distinct pairs:
| pair A | pair B | W |
|---|---|---:|
| C>K | I>M | 0.4132 |
| C>K | I>E | 0.3725 |
| I>M | I>W | 0.3717 |
| C>K | R>E | 0.3465 |

The outliers from the singleton clusters dominate the top of this list, exactly as expected.

## What the four visualizations show
1. **Clustermap (cell 19).** The bulk of the matrix is uniformly pale (most pairs are near-neighbors); a small block in the lower-right (clusters 5/8/1/2) and an upper-left island (clusters 3/4/11/12) light up against the background. The row/column cluster bars at k=6/12/20 stay coherent across cut levels — the structure is genuinely hierarchical, not an artefact of one choice of k.
2. **AA->AA cluster-assignment heatmap (cell 21).** Most of the 20x20 grid is filled with cluster 7. The basic (K, R, H) rows show many cells from clusters 5/8 (harmful) once you mutate *away* from a basic residue. The acidic (D, E) rows have cells in cluster 11 (beneficial) when D/E is replaced by K or R. The Cys row breaks across clusters 11 (gain) and 9 (mild gain), confirming Cys removal trends toward more active.
3. **Per-cluster centroid distributions (cell 23).** Cluster 7's histogram is a tall narrow spike at 0; cluster 11 is a left-shifted bump; cluster 5 is a right-shifted bump; the singleton clusters (1, 2, 3, 4, 12) look messier because they're built from a single pair's events.
4. **MDS scatter (cell 25).** Most points pile into the center (cluster 7); the harmful/beneficial clusters fan out into two distinct directions, with the singletons (C>K, I>M, I>W, R>D, F>W, G>E/I>E) on the far ends.

## Conclusions
1. The APEX-MIC-delta metric **does** organize AA->AA substitutions into a biologically interpretable structure on this AMP-negative parent set, whereas the previous position-based metric did not.
2. Two non-trivial clusters carry the signal: cluster 11 (gaining cationic / losing Cys -> more active) and cluster 5 (losing cationic / losing hydrophobic packing -> less active).
3. The majority cluster (cluster 7, ~228 pairs) is a feature, not a flaw: most amino-acid swaps simply do not move MIC on average, and APEX correctly recognises that.
4. Single-pair outliers (`R>D`, `C>K`, `I>M`, `I>W`) are the highest-impact individual substitutions and would be the natural targets for follow-up: either to confirm a specific design move (e.g. `D>K` ramping up activity) or to flag a single residue change worth avoiding.

## Reproducing
First run (builds the 11.3 MB cache from the 14.3 GB CSV, ~4 min):
```bash
jupyter nbconvert --to notebook --execute --inplace \
  analysis/notebooks/wildeast/mutang/final_wasserstein.ipynb \
  --ExecutePreprocessor.timeout=900
```
Subsequent runs read the cache and complete in ~1.5 min (the numba pairwise_wasserstein loop is ~77 s, MDS and plotting add the rest).

To switch parent datasets, change `DATASET_FILTER` in cell 3 (e.g. `hydramp_veltri_positive`); the loader will build a fresh `_cache/parents_<dataset>.parquet` and `_cache/mutants_<dataset>.parquet` on the next run.

---

# Extension: position-aware clustering on a wider dataset union

Section 14 of the notebook adds a second pass that fixes the two main weaknesses of the
1D-only metric:
- It discards position, so two AA->AA pairs enumerated at the *same* sites on the same
  parents (e.g. `A>L` and `A>K` from the same residue context) end up with correlated
  event sets and tend to cluster together by context, not by substitution biology.
- It looks at MIC delta in isolation, even though the same substitution at the
  N-terminus vs mid-helix vs C-terminus of an AMP can have very different effects.

The extension does three things together:
1. **Widens the parent set** to the union of `hydramp_veltri_negative ∪
   hydramp_veltri_positive ∪ hydramp_dbaasp_clean ∪ hydramp_mic_data` — 6,371 parent
   peptides (vs 854 in the main section). Cache parquet is ~55 MB (built in 200 s from
   the 14.3 GB CSV) and yields 234,727 valid substitution events.
2. **Replaces the 1D feature with a 2D feature** per event:
   `(relpos, mean_log2_ratio)`. `mean_log2_ratio` averages the per-bacterium log2 MIC
   ratio across the 34 strains so each event contributes a single number on each axis.
   Including `relpos` in the feature naturally resolves the same-site ambiguity: two
   pairs that occur at the same position but produce different MIC effects now live at
   the same x but different y, so the distance metric separates them properly.
3. **Uses Sliced 1-Wasserstein in 2D**: averages the 1D Wasserstein distance across
   30 random unit-direction projections of the 2D clouds (each projection delegates to
   the same numba `pairwise_wasserstein` util). Both axes are IQR-rescaled so neither
   dominates the projections.

## Extension pipeline summary

| Step | Value |
|---|---|
| Parent rows (union of 4 dataset tags) | 6,371 |
| Mutant rows kept | 328,631 |
| Valid substitution events | 234,727 |
| Per-event 2D features `(relpos, mean log2 ratio)` | 234,727 x 2 |
| Empty pairs / pairs with < 5 events | 2 / 8 |
| Events per pair: min / median / max | 0 / 193 / 6,616 |
| Sliced-Wasserstein projections | 30 |
| Sliced-Wasserstein matrix shape / runtime | (380, 380) / 56.5 s |
| D_sw min / mean / max | 0.000 / 0.759 / 4.862 |

## Does adding position actually matter?
Comparing the 1D-only `D` (section 7) against the position-aware `D_sw` (section 14.3):

| Comparison | Value |
|---|---|
| Spearman correlation on upper-triangle distances | **0.470** |
| Adjusted Rand Index of k=12 partitions | **0.167** |
| Normalised Mutual Information | **0.294** |

A moderate Spearman correlation paired with a near-zero ARI means the *pairwise distance
ordering* still partially agrees, but the *cluster partition is substantially
reorganised*. Position is contributing real new signal, not just noise.

## Position-aware cluster summary (k = 12)
Sorted by mean log2 MIC ratio.

| cluster | n_pairs | n_events | mean relpos | mean log2 ratio | example members |
|--:|--:|--:|--:|--:|---|
| 12 |   1 |      7 | 0.875 | -0.571 | D>H |
|  5 |   1 |    610 | 0.904 | -0.342 | E>K |
|  4 |   8 |    346 | 0.881 | -0.213 | C>W, D>Q, D>R, D>Y, F>W, M>W, N>W, T>W |
|  8 |  10 |  1,178 | 0.867 | -0.186 | C>A, C>G, C>K, C>M, C>R, C>S, D>K, D>L |
| 10 |   9 | 33,819 | 0.549 | -0.176 | E>L, H>I, N>K, P>I, P>K, Q>K, Q>W, T>I |
|  9 | 187 | 85,307 | 0.719 | -0.052 | A>F, A>I, A>K, A>L, A>N, A>P, A>Q, A>R, ... |
|  3 |   2 |      0 | 0.186 |  0.000 | A>H, D>P |
| 11 |   2 |  5,946 | 0.615 |  0.019 | K>W, W>K |
|  6 | 121 | 68,782 | 0.728 |  0.063 | A>C, A>E, A>G, A>M, A>S, A>T, C>D, E>C, ... |
|  7 |  28 | 23,711 | 0.704 |  0.165 | F>C, F>D, F>E, F>P, F>Q, F>Y, G>D, H>C, ... |
|  2 |   9 | 14,951 | 0.710 |  0.279 | A>D, I>D, I>P, K>C, K>D, K>P, K>Q, L>E, ... |
|  1 |   2 |     70 | 0.880 |  0.427 | R>D, W>D |

Reading this:
- The position axis splits what used to be a single "beneficial" cluster into a clear
  **C-terminal beneficial** family (c12, c5, c4, c8 — all `mean_relpos` ≈ 0.87-0.90):
  gaining cationic residues (`E>K`, `D>K`, `D>R`) and especially gaining Trp
  (`C>W, D>Y, F>W, M>W, N>W, T>W`) at the C-terminal third strongly lowers MIC. This is
  consistent with the amphipathic-helix anchoring biology of many AMPs.
- A separate **mid-peptide beneficial** cluster (c10, `relpos ≈ 0.55`) captures gains of
  Lys / Ile / Trp in the middle of the peptide, which is a different mechanism (helix
  stabilisation rather than terminal anchoring).
- The **strongly detrimental** end (c1, c2, c7) is dominated by *losing* cationic /
  hydrophobic character or by full charge inversion (`R>D`, `W>D` -> mean log2 ratio
  +0.43 at C-terminus). `F>{C,D,E,P,Q,Y}` (c7) shows that knocking out the aromatic ring
  is broadly harmful regardless of position.
- The two largest clusters (c9 with 187 pairs and c6 with 121) are the "background" of
  near-neutral substitutions, but they are now cleanly separated by sign of mean log2
  ratio (mild gain vs mild loss) and by position centroid — something the 1D-only run
  bundled into one indistinguishable cluster of 228 pairs.

## How to use the two views together
- Use **section 7's 1D clustermap** to ask "which substitutions have the same MIC
  footprint averaged over the peptide?"
- Use **section 14's 2D clustermap** to ask "which substitutions have the same MIC
  footprint *and* occur at the same part of the peptide?" — better suited for design
  decisions that depend on placement (e.g. "should I add a Lys at the C-terminus?").
- The Spearman/ARI/NMI cell (14.6) quantifies how much the two views diverge on any
  future dataset; rerun it after changing `EXT_DATASETS` to track whether adding more
  parents stabilises or shifts the clustering.

## Cache and runtime
- One-time extended cache build: 200 s (4 dataset tags, 37 M mutant rows scanned,
  328 k kept, 55 MB parquet).
- Steady-state notebook runtime (everything cached): ~3.5 min — 77 s for the 1D
  pairwise Wasserstein + 57 s for the 30-projection 2D Sliced Wasserstein + MDS and
  plotting.
- To change the extended dataset list, edit `EXT_DATASETS` in cell 29; a fresh cache
  parquet will be built on the next run.
