# Walidacja modeli decyzyjnych

## Zweryfikować

### ESM perplexity

Model językowy białek używany do odrzucania mało prawdopodobnych mutacji.

Kod: [`models/esm/esm2_ppl.py`](../../../models/esm/esm2_ppl.py),
[`optimization/baselines/random_mutation.py`](../../../optimization/baselines/random_mutation.py).

#### Co sprawdzić

- dokładną wersję i wariant modelu ESM;
- sposób tokenizacji, maskowania i normalizacji perplexity;
- znaczenie progu oraz zachowanie przy przekroczeniu limitu resamplingu;
- zgodność CPU/GPU i deterministyczność;
- różnice względem referencyjnej implementacji ESM.

#### Walidacja

Należy wyjaśnić użytkownikowi, że wynik modelu językowego nie jest bezpośrednią
oceną aktywności biologicznej, oraz udokumentować znaczenie skonfigurowanego progu.
