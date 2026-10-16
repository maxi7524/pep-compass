# Task conformance

The acceptance objective is the conjunction of R1–R11. No weighted score can hide
a failed mandatory condition. Evidence refers to the final source tree and
executed checks on `dev_max_integrated`.

| Criterion | Verdict | Evidence and reason |
| --- | --- | --- |
| R1 Direct public components under strategies | PASS | `tests/core/test_direct_api.py`: public exports are actual definitions; imports and constructors remain model-free. |
| R2 Remove registries and old compatibility paths | PASS | Source/import audit and wheel audit; no `registry`, `data`, `runtime`, old `optimization/components`, old declaration grammar or legacy result reader. |
| R3 Wrapper/builder/schema separation | PASS | Composition, immutable snapshots, stable edit IDs, atomic scopes and independent compilation regression tests. |
| R4 Explicit configuration and Step lifecycle | PASS | Component constructors, `Step.contract`, centralized lifecycle and lazy model initialization tests. |
| R5 Preserve computation and run semantics | PASS | Exact real reference; retained scientific tests; complete-batch thresholds, distinct terminal/archive data, seeded retention invariance and multiple replay boundaries. This is regression verification for the tested CPU paths. |
| R6 Same objects for Python and Hydra | PASS | Direct target workflow tests, actual CLI plan/run and isolated-wheel Hydra composition; YAML groups and ordinary preset functions. |
| R7 Common memory/SQLite result contract | PASS | Storage/export equality, lossless tensor/mask tests, checksums, lazy projected scans and collection reader tests. |
| R8 Separate new notebook; preserve existing notebooks | PASS | `wrapper_new.ipynb` executed end to end; original notebook hashes match the baseline and were excluded from writes. |
| R9 Reduce construction duplication and empty classes | PASS | Removed managers/config forwarding classes/family markers; shared configuration, lifecycle and artifact handling; AST audit found only the intentional data-unavailable exception without methods/fields. See the decision report for retained interfaces. |
| R10 Test requirements independently | PASS | 432 tests; frozen reference, analytical compositions, exhaustive field-transfer model, structural import checks, replay, storage and Hydra workflows; notebook execution. |
| R11 Apply and report coherent work | PASS | Changes applied to the actual repository after validation; scope audit and chronological explanation in `docs/refactor-report.md`. |

Overall task conformance: **PASS**.

# Algorithm verification

Scope: the new symbolic field-flow validator and the scientific regressions
preserved by this refactor. Pretrained model training and exhaustive scientific
validation of external models are outside this claim.

- Formal model: each field transfer is `f(I) = (I ∩ K) ∪ P`, with required inputs.
  Full-row guarantees and partially present fields are tracked separately.
- Input-domain checks: PASS for tested required fields, nullable permissions,
  unique IDs, ownership cycles, invalid children and constructor bounds.
- Composition/reference comparison: PASS, exhaustive over 4096 transfer pairs
  and four input subsets in a two-field domain.
- Branch invariants: PASS, complete fields require every branch; partial presence
  may come from any branch.
- Loop invariant: PASS; feedback requirements use the idempotent transfer summary.
  Zero-iteration loops do not promise body outputs.
- Structural complexity regression: PASS; nested loop contracts are visited once.
- Scientific reference: PASS; exact sequence order and evaluation values for the
  frozen real simple run; existing numerical tolerances were not relaxed.
- Retention/replay metamorphic checks: PASS; recording does not change the seeded
  result, and replay restores supported RNG/state boundaries.
- Numerical payload integrity: PASS for tested dtype, shape, masks, shared mutation
  containers and byte-exact tensor serialization.
- Mutation testing: NOT RUN.
- GPU, Slurm submission, all optional external model inference: NOT VERIFIED;
  no claim is made for these environments.

Overall algorithm verification within the stated scope: **PASS**.

Complexity: one graph traversal, O((N+E)F) field-set work, where F is the number of
relevant field names. Model loading and numerical component work are separate.
This avoids exponential revisiting of nested loops; no global optimality claim.

# Final verification

- Complete target-repository suite: **432 passed**.
- Scoped Ruff `F,E9,I`: PASS; format check: PASS for **196 Python files**.
- New Jupyter notebook: PASS; executed copy in `/tmp/wrapper_new.complete.ipynb`.
- Wheel build: PASS; **323 Python files** compared byte-for-byte to current source;
  removed modules absent.
- Isolated wheel import, actual seeded computation and Hydra composition: PASS.
- CLI plan/run and `git diff --check`: PASS.
- Existing notebook preservation: PASS by SHA-256.

No dependency upgrades, branch switch, commit or push was performed. The wide
vendor-code lint is not clean; scoped lint results do not claim otherwise. See
`docs/refactor-report.md` for commands, decisions and verification limits.

# Concurrent change after verification

After the complete passing test run, a separate edit appeared in
`assets/tools/proteolysis_data_manager/src/proteolysis_data_manager/acquisition/half_life.py`.
Its leading backtick causes `SyntaxError` on line 1. This file was not modified by
this refactor. The 432-test result predates that edit; the full current suite
cannot be reported as passing until that separate syntax error is removed.
Task conformance above refers to the requested refactor's change set.
