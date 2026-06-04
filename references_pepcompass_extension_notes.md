# Literature notes — PepCompass extension (Karol's MSc thesis)

Companion to `references_pepcompass_extension.bib`. For each entry, one or two
lines on **why it earns its slot in the lit-review**: what claim it supports,
which baseline it provides, or which gap it lets you frame.

Tags:
- **[core]** — essential, cite in introduction or methods
- **[baseline]** — direct comparator for your experiments
- **[framing]** — used to motivate the gap or set up vocabulary
- **[support]** — backs a specific technical claim
- **[forward]** — discussion / future-work material

---

## [01] PepCompass core dependencies

- **`szymczak2023hydramp`** **[core]** — HydrAMP is the encoder–decoder you
  operate on. Cite when introducing the 64-dim latent space and whenever you
  reference the decoder Jacobian. Establishes that the latent space you
  navigate is already biologically meaningful.
- **`nijkamp2023apex`** **[core]** — APEX is your oracle/reward signal. The
  log₂MIC values driving both LE-BO and the RL extensions come from here;
  must be cited in methods.
- **`torres2024discovery`** **[framing]** — modern review of AMP discovery
  bottlenecks. Use in introduction to position PepCompass against
  brute-force/in-silico screens.
- **`wang2016apd3`**, **`pirtskhalava2021dbaasp`** **[support]** — the AMP
  databases that define the universe of training data. Cite where you discuss
  data sources or coverage of sequence space.

## [02] VAEs & latent-space generative models

- **`kingma2014vae`** **[core]** — foundational VAE paper. One-line cite when
  introducing the encoder–decoder.
- **`higgins2017betavae`** **[support]** — disentanglement framing; useful
  if you discuss what latent dimensions encode and whether the local geometry
  has an interpretable basis (relevant to the MUTANG SVD-direction story).
- **`gomezbombarelli2018automatic`** **[framing]** — the canonical
  "optimize in continuous latent space instead of discrete sequence space"
  paper. Cite as the lineage your work belongs to.
- **`bowman2016generating`** **[support]** — earliest demonstration that
  generative latent spaces have meaningful geometry along interpolations;
  motivates why geodesics-on-decoders is a sensible idea at all.

## [03] Riemannian geometry of deep generative models

- **`arvanitidis2018latent`** **[core]** — *the* paper that introduces the
  pullback metric `G = J^T J` on a VAE latent. This is the mathematical seed
  of PepCompass. Cite in the methods section where you define `G_Dec(z)`.
- **`shao2018riemannian`** **[support]** — extends the pullback story with
  numerical methods for geodesics. Useful when discussing why PoGS optimizes
  waypoints rather than solving a geodesic ODE directly.
- **`chen2018metrics`** **[support]** — alternative metric formulations on
  generative latents; cite for completeness in the related-work section.
- **`tosi2014metrics`** **[support]** — earlier work on probabilistic
  manifolds; situates the VAE pullback metric within a broader programme.
- **`detlefsen2022reliable`** **[support]** — addresses stability of estimating
  geometric quantities; relevant when justifying the κ-stability threshold
  (singular-value floor).
- **`arvanitidis2021geometrically`** **[forward]** — generalises latent
  geometries beyond pullback. Use in discussion / future work where you
  speculate on richer metric choices.

## [04] Sub-Riemannian geometry

- **`montgomery2002subriemannian`** **[core]** — standard reference for the
  horizontal/vertical decomposition used in `SubRiemannianTangentSpace`. Cite
  when introducing the horizontal subspace concept in MUTANG++.
- **`agrachev2019sub`** **[support]** — modern, more accessible textbook
  reference. Cite alongside Montgomery, especially for explicit definitions of
  bracket-generating distributions and admissible curves.
- **`bellaiche1996tangent`** **[support]** — defines the sub-Riemannian
  tangent space rigorously. Useful as the explicit anchor for the term
  "tangent space" as you use it.

## [05] Geodesics & Brownian motion on manifolds

- **`hsu2002stochastic`** **[core]** — Riemannian Brownian motion is what
  SORBES is approximating. Cite when stating the convergence theorem
  (ε → 0 gives Riemannian BM).
- **`kalatzis2020brownian`** **[support]** — VAE-side analogue: ML paper that
  uses Riemannian BM as a prior. Strengthens the case that BM-on-decoders is a
  studied object, not an idiosyncratic choice.
- **`do1992riemannian`** **[support]** — go-to textbook for Christoffel
  symbols. Cite when introducing the second-order correction Γ(z)[v,v].

## [06] Bayesian optimization (high-dim, latent-space, Tanimoto, ROBOT)

- **`shahriari2016taking`** **[support]** — broad BO review. Cite once when
  introducing BO terminology (acquisition, GP surrogate, EI).
- **`eriksson2019turbo`** **[support]** — TuRBO trust regions; local BO
  philosophy that LE-BO inherits.
- **`eriksson2021saasbo`** **[baseline]** — SAASBO is your stated comparator
  ("8× faster than SAASBO"). Must be cited in the experiments section.
- **`maus2022local`** **[baseline]** — LOL-BO is the closest published cousin
  of LE-BO. Cite explicitly when explaining what LE-BO does differently
  (geometric neighborhoods via MUTANG++ rather than gradient-descent
  trust regions).
- **`maus2023robot`** **[core]** — ROBOT diversity scheme used inside LE-BO.
  Must be cited in methods.
- **`hvarfner2024vanilla`** **[support]** — counterpoint argument that vanilla
  high-dim BO can be competitive. Cite if a reviewer asks why you need a
  bespoke approach — and to honestly position your work.
- **`griffiths2020constrained`** **[support]** — Tanimoto kernel + VAE
  combination. Cite when introducing the Tanimoto kernel choice.
- **`ralaivola2005graph`** **[support]** — original Tanimoto kernel derivation
  for cheminformatics. One-line cite alongside `griffiths2020constrained`.

## [07] Molecular fingerprints

- **`capecchi2020map4`** **[core]** — MAP4 is the fingerprint feeding the GP
  surrogate inside LE-BO. Cite explicitly in methods.

## [08] BLOSUM & substitution matrices

- **`henikoff1992blosum`** **[core]** — the original BLOSUM paper. Cite when
  you introduce position-specific BLOSUM scores from `blosum/` (your new
  BLOSUM-based potential / scoring extension).
- **`eddy2004blosum62`** **[support]** — explains the information-theoretic
  derivation of BLOSUM62. Useful for justifying *which* BLOSUM variant you use
  and what its score units actually represent.
- **`altschul1991amino`** **[support]** — earlier information-theoretic
  framing of substitution matrices. Cite when arguing your BLOSUM extension
  is a principled scoring (not just an ad-hoc prior).
- **`styczynski2008blosum62`** **[support]** — notes well-known numerical
  quirks in BLOSUM62. Worth a footnote if you discuss reproducibility / the
  exact matrix you load.

## [09] Epistasis & deep mutational scanning

- **`olson2014comprehensive`** **[core]** — direct empirical evidence for
  **pairwise epistasis** at the protein scale. This is the central biological
  motivation for MUTANG++: positions are not independent, and pairwise
  similarity captures that.
- **`fowler2014deep`** **[framing]** — DMS overview; cite to frame the
  experimental landscape your scoring is implicitly fitting.
- **`starr2016epistasis`** **[support]** — review of epistasis in protein
  evolution. Cite to argue that combinatorial scoring is necessary, not
  optional.
- **`riesselman2018deep`** **[support]** — DeepSequence: generative-model
  mutation effects. Useful as a baseline view of "use a generative model
  to score variants," contrasted against your geometric approach.
- **`hopf2017mutation`** **[support]** — EVmutation, a co-variation-based
  predictor. A natural compare-and-contrast for the pairwise-similarity idea.
- **`stiffler2015evolvability`** **[support]** — empirical fitness-landscape
  data; cite when motivating why local navigation matters.

## [10] Reinforcement learning foundations

- **`sutton2018reinforcement`** **[core]** — single textbook citation
  covering ε-greedy, Q-learning, TD(0), epsilon decay schedules. Cite once,
  prominently, in the RL methods section.
- **`watkins1992qlearning`** **[core]** — original Q-learning paper. Cite at
  the exact moment you write down the TD(0) update in `eps_greedy_learn`.
- **`auer2002bandit`** **[support]** — finite-time bandit analysis; supports
  the choice of ε-greedy over UCB and the exponential decay schedule.
- **`mnih2015human`** **[support]** — DQN; cite when discussing why you chose
  a **tabular** Q-table (state space = peptide strings; deep value
  approximation is unnecessary at MUTANG++-filtered action-set sizes).
- **`williams1992simple`** **[forward]** — REINFORCE / policy gradient.
  Cite in future-work where you propose a policy-network successor.

## [11] RL for biological / molecular sequence design

- **`angermueller2020dyna`** **[baseline]** — DyNA-PPO: the canonical RL
  baseline for biological-sequence design. Must compare against in the RL
  chapter — explain why your MUTANG++-structured action space is preferable
  to its proposal distribution.
- **`zhou2019moldqn`** **[baseline]** — MolDQN: Q-learning over molecule
  actions. The closest published analog of `eps_greedy_learn`. Comparison is
  essential.
- **`olivecrona2017molecular`** **[support]** — early RL-for-de-novo-design;
  positions your work historically.
- **`popova2018deep`** **[support]** — ReLeaSE: actor–critic for molecules.
  Cite as related-work coverage.
- **`jain2022biological`** **[support]** — GFlowNets for biological sequences;
  a structurally different approach with a similar goal. Mention as related,
  non-RL paradigm.
- **`stanton2022accelerating`** **[support]** — BO + denoising AEs for
  sequences. Useful bridge between your BO and RL chapters.

## [12] Generative methods for AMP design

- **`muller2018recurrent`** **[baseline]** — RNN-based AMP generation;
  earliest neural baseline.
- **`tucs2020ampgan`**, **`vanoort2021ampgan`** **[baseline]** — AMPGAN
  v1/v2; GAN-based AMP design. Direct comparators in the AMP-discovery
  related-work section.
- **`das2021accelerated`** **[baseline]** — CLaSS controlled generation +
  MD validation. Strong baseline that combined generation with biophysics.

## [13] Antimicrobial resistance — clinical motivation

- **`oneill2016tackling`** **[framing]** — the headline AMR projection
  (10M deaths/year by 2050). Standard first-paragraph cite.
- **`murray2022global`** **[framing]** — current empirical AMR burden
  (4.95M deaths attributed to bacterial AMR in 2019). Use alongside O'Neill
  to anchor the clinical problem.

## [14] Protein language models — forward-looking extensions

- **`rives2021biological`** **[forward]** — ESM-1b; protein language models.
  Cite in future-work where you propose replacing/augmenting the HydrAMP
  latent with an ESM embedding.
- **`lin2023evolutionary`** **[forward]** — ESM-2 / ESMFold. Forward-work
  reference for incorporating structural priors.
- **`meier2021language`** **[forward]** — ESM-based zero-shot variant scoring.
  Natural successor to the BLOSUM extension and a candidate alternative
  scoring potential.

---

## Suggested reading order before writing the lit-review chapter

1. **Geometric backbone** (already familiar): `arvanitidis2018latent`,
   `montgomery2002subriemannian`, `hsu2002stochastic`.
2. **Direct competitors to LE-BO**: `eriksson2021saasbo`, `maus2022local`,
   `maus2023robot`.
3. **Direct competitors to ε-greedy / Q-learn extensions**:
   `angermueller2020dyna`, `zhou2019moldqn`.
4. **Biological motivation for MUTANG++**: `olson2014comprehensive`,
   `starr2016epistasis`.
5. **BLOSUM extension grounding**: `henikoff1992blosum`, `eddy2004blosum62`.

That set covers ~80% of the citations you'll actually need in the body of
the thesis; the rest are one-line supporting cites in their respective
sections.

---

## Possible additional papers to chase if the chapter feels thin

- **Geodesic shooting on VAEs** — Beik-Mohammadi et al., 2021, "Learning
  Riemannian Manifolds for Geodesic Motion Skills" (transfer of pullback-
  metric ideas to control; useful framing for PoGS).
- **Specific ε-greedy ablations** — Tokic 2010, "Adaptive ε-greedy exploration
  in reinforcement learning based on value differences." Use if you tune or
  ablate the decay schedule.
- **A protein-RL survey** — Wang et al., 2024 (recent surveys on RL for protein
  design appear most years; pick a current one).
- **A geometric DL textbook** — Bronstein et al., 2021, "Geometric Deep
  Learning: Grids, Groups, Graphs, Geodesics, and Gauges." Useful for the
  introduction if you want a single broad reference for "geometry-aware ML."
