# Poranna walidacja funkcjonalna

Najpierw wykonaj `--dry-run` dla każdego pliku YAML w tym katalogu. Następnie
uruchamiaj konfiguracje w kolejności:

1. `latent_trust_region_smoke.yaml` — lekki oracle, pełna pętla i tracking;
2. `esm_smoke.yaml` — filtr decision model ESM;
3. `mutation_filters_smoke.yaml` — LPBEBO, LAMS, TANDEM, MOVE i random controls;
4. `parallel_smoke.yaml` — dwie współbieżne gałęzie i merge;
5. `oracle_smoke.yaml` — wszystkie rzeczywiste modele oracle.

Przykładowe polecenia:

```bash
.venv/bin/python scripts/runner/run_composable_optimization.py \
  --config configs/optimization/validation/latent_trust_region_smoke.yaml

.venv/bin/python scripts/runner/run_composable_optimization.py \
  --config configs/optimization/validation/mutation_filters_smoke.yaml
```

`oracle_smoke.yaml` materializuje indeksy:

- `0`: Hydrophobicity;
- `1`: APEX;
- `2`: BattleAMP;
- `3`: ToxiPep;
- `4`: EIPred;
- `5`: MBC-Attention.

Można uruchomić pojedynczy model:

```bash
.venv/bin/python scripts/runner/run_composable_optimization.py \
  --config configs/optimization/validation/oracle_smoke.yaml \
  --run-index 0
```

Po testach użyj agregatora:

```bash
.venv/bin/python scripts/runner/aggregate_composable_results.py \
  configs/optimization/validation/results/validation/oracles
```

Błędy wag, zależności lub importów konkretnego oracle należy wpisać do jego
README walidacyjnego. Te konfiguracje rozdzielają takie problemy od działania
core, pętli, trackingu, backendów i agregacji.
