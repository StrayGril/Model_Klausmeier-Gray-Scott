# Projekt: Model Klausmeiera-Graya-Scotta

## Badanie powstawania formacji roślinnych na terenach pustynniejących

### 1. Opis.

Projekt zawarty w repozytorium ma na celu badanie modelu Klausmeiera-Graya-Scotta, czyli układu nieliniowych równań różniczkowych cząstkowych. Analizowane są zmiany biomasy i wody w zależności od opadów.

### 2. Zawartość.

\pipeline - pliki .py z kodem z funkcjami do poszczególnych etapów analizy modelu

\notebooks - notatniki .ipynb (Jupyter) korzystające z funkcji z \pipeline do wykonywania symulacji

\folder_na_modele - pliki do modelu machine learning do dalszego etapu analizy na podstawie wykonanych symulacji

\data - dane wygenerowane przez model, zapisane w plikach .npz (do oetykietowania) lub .csv

\figury - wykresy.

### 3. Obsługa.

Repozytorium można pobrać i otworzyć jako projekt w dowolnym programie obsługującym język Python oraz odczytującym notatniki Jupyter, aby przetestować funkcje dla własnych danych lub przeglądnąć kod oraz notatniki (z wykonanym kodem) w przeglądarce.

### 4. Wykorzystywane pakiety.

Część symulacyjna korzysta z następujących pakietów pythonowskich:

- matplotlib,
- numpy,
- os,
- panda,
- scipy,
- sys,
- tqdm.

Ponadto część modelowa używa:

...

