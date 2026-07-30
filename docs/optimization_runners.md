# Konfigurowalne eksperymenty optymalizacyjne

Nowy runner wykonuje drzewo kroków opisane w YAML. Nie wybiera gotowego
„optymalizatora” po nazwie: walker, generator mutacji, filtry i oracle są
niezależnymi krokami, które można składać w dowolnej kolejności.

## Uruchomienie

```bash
.venv/bin/python scripts/runner/run_composable_optimization.py \
  --config configs/optimization/composable_example.yaml
```

Wymagane są tylko sekwencje wejściowe oraz encoder-decoder. Walker, generator
mutacji, filtry, pętla i oracle są opcjonalne. Eksperyment bez oracle zapisuje
końcowy batch, a pola `best_score`, `objective_name` i `objective_direction`
mają wartość `null`.

## Input

Sekwencje można podać bezpośrednio:

```yaml
experiment:
  input:
    sequences:
      - FLYKWWIRIGRLKL
      - KLLKLLKKLLKLLK
```

albo w CSV:

```yaml
experiment:
  input:
    csv:
      path: peptides/input.csv
      sequence_column: sequence
      repetitions_column: repetitions
```

Ścieżka względna jest liczona od pliku YAML. Kolumna `repetitions` jest
opcjonalna i domyślnie ma wartość `1`. Na obecnym etapie powtórzenia rozwijają
wejściowy batch; niezależne taski eksperymentalne i grid parametrów będą osobną
warstwą wykonawczą.

## Kroki

Każdy element `optimization.steps` ma dokładnie jeden klucz operacji:
`walker`, `mutation_generator`, `filter`, `oracle`, `loop` albo `parallel`.

```yaml
optimization:
  limits:
    oracle_calls: null
    generated_candidates: 100000
  steps:
    - loop:
        iterations: 10
        steps:
          - walker:
              method: sorbes
              parameters: {}
          - mutation_generator:
              method: mutang
              parameters: {}
          - filter:
              method: deduplicate
              parameters:
                key: sequence_and_latent
```

`CandidateBatch` zawsze przenosi kolumnowo sekwencje i ich `latent_origin`.
Dodatkowe pola obliczeniowe są wyrównane do kandydatów. Tracking jest osobnym
strumieniem danych i nie zanieczyszcza batcha.

## Równoległe gałęzie

`parallel` przekazuje ten sam wejściowy batch do każdej gałęzi. Gałąź może
zawierać dowolne kroki, również własne pętle. Wszystkie gałęzie muszą zakończyć
się `CandidateBatch`; merger sprowadza wyniki do jednego batcha, aby kolejne
kroki mogły działać bez znajomości topologii drzewa.

```yaml
- parallel:
    execution: concurrent
    merge: concatenate
    branches:
      - name: local_geometry
        steps: []
      - name: alternative_generator
        steps: []
```

Dostępne jest deterministyczne `concatenate`. Funkcje `interleave`,
`select_best` i `weighted_sample` są pozostawione jako jawne punkty rozszerzeń
z `TODO`, dopóki nie zostaną ustalone ich kontrakty.

## Tracking

`short` zapisuje oracle, `normal` zapisuje podsumowania kroków, a `all` także
kandydatów. `max_depth` ogranicza głębokość zbierania bez zmiany wykonania.
Indeksy iteracji i nazwy gałęzi są zapisywane osobno. `store_latents` i
`store_fields` włączają kosztowniejsze dane tylko wtedy, gdy są potrzebne.
