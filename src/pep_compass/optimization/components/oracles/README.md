# Oracles — zakres zmian do przekazania

Instrukcje dla osoby przejmującej porządkowanie tego komponentu, bez konieczności samodzielnego
czytania każdej metody w `strategies/`.

## Scope

Ten dokument opisuje wyłącznie strukturę i spójność kodu adapterów (`oracles/strategies/*/oracle.py`),
nie poprawność naukową predykcji poszczególnych modeli (wagi, preprocessing, interpretacja wyniku).
Stan walidacji naukowej opisuje [`docs/developer/REFACTORING_HANDOFF_PL.md`](../../../../../docs/developer/REFACTORING_HANDOFF_PL.md),
sekcja "Modele predykcyjne oracle" — modele czekają na osobną walidację przez ich właścicieli.

## Warstwa wspólna — nie zmieniać

`Oracle` (`base.py`) → `BlackBoxOracle` (`strategies/black_box.py`) → rejestracja przez
`_black_box_oracle()` w `strategies/__init__.py`. Ta warstwa jest spójna dla wszystkich sześciu
zarejestrowanych oracle'i (`apex`, `battleamp`, `eipred`, `hydrophobicity`, `mbc_attention`,
`toxipep`): limit wywołań (`context.state.remaining_oracle_calls()`), chunkowanie po
`evaluation_batch_size`, zapis obserwacji (`record_observations`), konwencja pól
`oracle.<name>.score` / `.direction` / `.name`. Żaden pojedynczy oracle tego nie duplikuje. Każdy
`OracleManager.register(...)` występuje dokładnie raz — brak duplikatu rejestracji, w
przeciwieństwie do analogicznych bugów opisanych dla `walkers`/`mutation_generators` w planie
restrukturyzacji tamtych komponentów.

## APEX — zweryfikowany, bez zmian

`strategies/apex/oracle.py`, klasa `APEXBlackBox`: poprawnie zarejestrowana
(`strategies/__init__.py:33-42`), poprawnie owinięta przez `BlackBoxOracle`, `self.maximize = False`
ustawione jawnie, brak martwych zależności w ścieżce wykonania. Jedyna zmiana dotycząca tego pliku to
usunięcie nieużywanej klasy opisanej w punkcie 4 poniżej — nie zmienia to działania `APEXBlackBox`.

Nieprzetestowane uruchomieniowo w ramach tego przeglądu (sprawdzone tylko czytaniem kodu): ładowanie
wag przez `PredictorAPEX`/`APEXUnpickler` (`apex/APEX_predictor.py:16-24`, customowy unpickler
remapujący stare nazwy modułów) i poprawność liczbowa predykcji.

## Zmiany do wprowadzenia w pozostałych pięciu oracle'ach

### 1. Brakujące `self.maximize`

`BlackBoxOracle._attach_result` czyta `getattr(self.black_box, "maximize", False)` — brak
przypisania w `__init__` oznacza cichy fallback na `"minimize"`, bez jawnej decyzji w kodzie.

Dotyczy `strategies/eipred/oracle.py` (`EIPredBlackBox.__init__`) i
`strategies/mbc_attention/oracle.py` (`MBCAttentionBlackBox.__init__`) — obie klasy nie ustawiają
`self.maximize`. Oba modele dziś zwracają `log2(...)`, więc kierunek jest prawdopodobnie `False`, ale
wymaga potwierdzenia merytorycznego z właścicielem modelu przed dodaniem jawnego przypisania.

### 2. Sprzeczny opis kierunku w `ToxiPepBlackBox`

`strategies/toxipep/oracle.py:27-32` — docstring klasy opisuje wyższy score jako bezpieczniejszy
peptyd (sugeruje maksymalizację). Komentarz przy `self.peptide_scorer` (linia 67) mówi
"for minimization", a `self.maximize = False` jest ustawione na sztywno (linia 73). Do wyjaśnienia
merytorycznie z właścicielem modelu, który z dwóch opisów jest poprawny, i poprawienia drugiego.

### 3. Nieużywany `self.cache`

Występuje w `apex/oracle.py`, `battleamp/oracle.py`, `eipred/oracle.py`, `hydrophobicity/oracle.py`,
`mbc_attention/oracle.py`, `toxipep/oracle.py` (ten ostatni ma dodatkowo gettery
`get_cache_size`/`get_cached_results`). Nic poza samym plikiem oracle'a nigdy go nie czyta; rośnie bez
ograniczeń przez cały przebieg optymalizacji. Do usunięcia z wszystkich sześciu plików, albo — jeśli
ma wartość diagnostyczną — do scentralizowania w jednym miejscu zamiast kopiowania w każdym adapterze.

### 4. Nieużywana klasa `HydrAMPAPEXBlackBox`

`apex/oracle.py:68-149`. Duplikuje logikę `APEXBlackBox`, dodatkowo dekoduje z latentu przez
`HydrampAutoencoder`, ma własne `shift`/`cache`. Nigdzie nieimportowana ani niezarejestrowana —
`strategies/__init__.py` importuje z tego modułu tylko `APEXBlackBox`. Do usunięcia albo przeniesienia
poza ścieżkę importową `strategies/__init__.py` z jawnym oznaczeniem jako nieużywany/eksperymentalny.

### 5. Ręczne smoke-testy w plikach produkcyjnych

`strategies/hydrophobicity/oracle.py` (linie 65-111) i `strategies/toxipep/oracle.py` (linie 132-182)
zawierają bloki `if __name__ == "__main__":` z ręcznymi testami i `print()`. Do usunięcia albo
przeniesienia do `tests/` jako właściwe testy jednostkowe.

### 6. Niespójne źródło `AbstractBlackBox`

`apex/oracle.py` i `battleamp/oracle.py` importują `AbstractBlackBox` z `poli.core.abstract_black_box`;
`eipred/oracle.py`, `hydrophobicity/oracle.py`, `mbc_attention/oracle.py`, `toxipep/oracle.py`
importują z `poli_baselines.core.abstract_solver`. Do wyjaśnienia: czy rozjazd jest zamierzony (dwie
różne biblioteki dla różnych klas modeli), czy przypadkowy i wymaga ujednolicenia.

### 7. Ręczna izolacja procesowa tylko w BattleAMP

`battleamp/oracle.py:47-52` uruchamia predyktor w osobnym procesie CPU przy `device=cuda` przez
ręczny `ProcessPoolExecutor` (komentarz w kodzie: unika konfliktu CUDA/cuSOLVER z PyTorchem
wykonującym LE-BO na GPU). Wszystkie sześć oracle'i przyjmują już `force_isolation` jako parametr POLI
(`_COMMON` w `strategies/__init__.py:10-13`). Do wyjaśnienia: czy wbudowany mechanizm POLI faktycznie
nie wystarcza tutaj (i wtedy ten wzorzec wymaga udokumentowania i ewentualnego powtórzenia tam, gdzie
jest potrzebny), czy to obejście możliwe do zastąpienia samym `force_isolation=True`.

## Powiązane dokumenty

- [`docs/developer/REFACTORING_HANDOFF_PL.md`](../../../../../docs/developer/REFACTORING_HANDOFF_PL.md)
  — stan walidacji naukowej modeli i kontekst poprzedniej restrukturyzacji pakietu.
