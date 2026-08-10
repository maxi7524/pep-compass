# Przeniesienie wcześniejszej warstwy `models`

Ten dokument wskazuje zespołowi autorów wcześniejszych implementacji, gdzie
znajduje się obecnie ich kod i które interfejsy zostały zmienione podczas
refaktoryzacji. Nie opisuje poprawności naukowej poszczególnych metod; ta będzie
walidowana osobno przez ich właścicieli.

## Autoencoder HydrAMP

Wcześniejszy pakiet `src/pep_compass/encoder_decoder/` został zastąpiony przez
`src/pep_compass/autoencoder/`. Odpowiedniki najważniejszych plików są następujące:

```text
encoder_decoder/base.py
  -> autoencoder/base.py

encoder_decoder/manager.py
  -> autoencoder/registry.py
  -> autoencoder/factory.py

encoder_decoder/strategies/hydramp/model.py
  -> autoencoder/strategies/hydramp/architecture.py

encoder_decoder/strategies/hydramp/adapter.py
  -> autoencoder/strategies/hydramp/adapter.py

encoder_decoder/strategies/hydramp/weights/*.pickle
  -> autoencoder/strategies/hydramp/models/article_25/*.pickle
```

Model wybierany jest teraz jawnie przez parę `method: hydramp` oraz
`model: article_25`. Rejestr przechowuje nazwane warianty modelu, a factory
łączy parametry wariantu z parametrami uruchomienia. Import samego pakietu nie
ładuje wag; rzeczywista architektura i wagi są ładowane dopiero podczas budowy
workflow.

Adapter udostępnia wspólny kontrakt `Autoencoder`: kodowanie sekwencji,
dekodowanie, wymiar latentny i ambientowy oraz operacje potrzebne do geometrii
dekodera. Komponenty optymalizacyjne nie tworzą własnej kopii modelu — gotowa
instancja jest wstrzykiwana przez `PipelineBuilder`.

## Geometria latentna

Obliczenia Jacobianu i SVD, które wcześniej występowały w kilku miejscach,
zostały skupione w `src/pep_compass/autoencoder/geometry.py`. Jest to pomocnicza
funkcjonalność autoencodera, używana między innymi przez MUTANG, gdy walker nie
przekazał już obliczonej geometrii w polach batcha.

Kod analiz lokalności nie jest częścią autoencodera. Został wydzielony do
`src/pep_compass/analysis/`, ponieważ konsumuje zapisane wyniki, zamiast
uczestniczyć w pętli wykonawczej.

## SORBES, MUTANG i filtry decyzyjne

Dawne pakiety najwyższego poziomu zostały przeniesione pod wspólną warstwę
komponentów:

```text
walkers/
  -> optimization/components/walkers/

mutation_generators/
  -> optimization/components/mutation_generators/

filters/
  -> optimization/components/filters/

oracles/
  -> optimization/components/oracles/
```

Klasy bazowe dziedziczą teraz po wspólnym `Step`. Implementacja komponentu
powinna dostarczać `_execute(batch, context)`. Publicznego `__call__` nie należy
nadpisywać, ponieważ odpowiada on za pomiary czasu i pamięci, tracking oraz
obsługę błędów.

Argumenty wejściowe i wyniki nie są już przekazywane jako luźne słowniki.
Komponent otrzymuje `CandidateBatch`, który przechowuje sekwencje, pozycje
latentne i typowane pola wyrównane względem kandydatów. MUTANG zachowuje indeks
i sekwencję rodzica, a SORBES zapisuje geometrię styczną w polach
`walker.*`, dzięki czemu kolejne komponenty mogą ponownie wykorzystać wynik.

## Modele predykcyjne oracle

Pliki modeli APEX, BattleAMP, EIPred, MBC-Attention i ToxiPep znajdują się teraz
pod `src/pep_compass/optimization/components/oracles/strategies/`. Zewnętrzne
implementacje zostały opakowane adapterami zgodnymi z `Oracle` i są wybierane
przez `OracleManager`.

Nie wykonano jeszcze pełnej walidacji naukowej tych modeli. W szczególności
należy osobno sprawdzić preprocessing, interpretację wyniku, urządzenie,
wersje bibliotek i zgodność dołączonych wag. Sam fakt rejestracji oznacza tylko,
że model może zostać odnaleziony i włączony do pipeline.

## `decision_models` nie jest rejestrem modeli

`optimization/components/filters/strategies/decision_models/` zawiera funkcje
oceniające kandydatów dla filtrów, między innymi ESM oraz potencjały MUTANG,
TANDEM i MOVE. Nie zastępuje rejestru autoencoderów ani oracle. Są to elementy
wewnętrzne filtrów i nie powinny być ładowane bezpośrednio przez runtime.

## Zmiana sposobu uruchamiania

Wcześniejsze skrypty składające ręcznie pełne eksperymenty zostały rozdzielone:

```text
runtime/       wejście, warianty, plan wykonania, CLI i zapis wyników
core/          walidacja deklaracji i budowa wykonywalnego grafu
optimization/  komponenty naukowe oraz operacje wykonawcze
analysis/      odczyt wyników i analizy wykonywane po eksperymencie
```

Nowa operacja `LocalEnumeration` utrzymuje osobno punkt kontynuujący trajektorię
SORBES oraz pulę kandydatów MUTANG. Kandydaci MUTANG nie stają się automatycznie
punktami początkowymi następnego kroku SORBES. Lista lokalnych filtrów jest
wykonywana po każdym MUTANG, natomiast globalny wybór, deduplikacja, oracle i
Bayesian optimization pozostają zwykłymi krokami po Local Enumeration.
