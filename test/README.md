# test/ — proof of concept i testy automatyczne

Wymaga przygotowanych danych (`cd src && python data.py`). Testy, które potrzebują danych, są pomijane (skip), gdy danych brak.

## 1. Proof of concept (`poc.py`)

Cały pipeline projektu w skali mini, w ~15 s na Apple M4 (MPS). Na CPU trwa to około kilku minut.

```bash
python test/poc.py                     # 10 klas, 3 epoki, urządzenie auto (CUDA > MPS > CPU)
python test/poc.py --classes 20 --epochs 5 --device cpu
```

Co robi:
- bierze N pierwszych klas z **tego samego** podziału 80/10/10 (`splits/split_seed6102.csv`, seed 6102); przy 10 klasach to 418 / 52 / 52 zdjęcia,
- trenuje ResNet-18 z wagami ImageNet w dwóch wariantach z eksperymentu: **zamrożone ciało** (baseline) i **odmrożony ostatni blok**,
- liczy accuracy, macro-F1 i top-3 (**własny licznik `TopKCounter`**) na teście,
- zapisuje w `test/poc_output/` pliki `poc_results.json`, `poc_curves.png` (val accuracy w kolejnych epokach) i `poc_predictions.jpg` (predykcje top-3).

Używa tych samych funkcji co pełny projekt (`src/data.py`, `src/models.py`, `src/metrics.py`, `src/train.py`), więc działający PoC potwierdza, że działa też główny kod.

Przykładowy wynik (10 klas, 3 epoki, MPS):

| wariant | trenowane param. | czas | accuracy | macro-F1 | top-3 |
|---|---:|---:|---:|---:|---:|
| zamrożone ciało | 5 130 | 5 s | 0,904 | 0,847 | 1,000 |
| ostatni blok | 8,4 M | 4 s | 0,981 | 0,975 | 1,000 |

Już w mini-skali widać ten sam efekt co w pełnym eksperymencie: odmrożenie ostatniego bloku wyraźnie poprawia wynik względem zamrożonych cech ImageNet. Test zawiera tylko 52 zdjęcia (1 zdjęcie ≈ 1,9 punktu procentowego), a augmentacja w procesach roboczych i operacje na MPS nie są w pełni deterministyczne. Dlatego liczby mogą się różnić o kilka punktów procentowych między uruchomieniami. Kierunek wyniku (ostatni blok > zamrożone) jest stabilny.

## 2. Testy (`pytest`)

```bash
python -m pytest test -q      # 18 testów, ~15–20 s, CPU
```

| plik | co sprawdza |
|---|---|
| `test_metrics.py` | własny `TopKCounter`: ręcznie policzony przykład, akumulacja po batchach daje to samo co jeden batch, zgodność z niezależną implementacją brute-force (ranking prawdziwej klasy), k = liczba klas → 1.0, `reset()` |
| `test_data.py` | podział 6551/819/819, rozłączność części, każda ze 102 klas w każdej części, odtworzenie splitu z seed 6102, kształt próbek z `FlowersDataset` |
| `test_models.py` | liczba trenowanych parametrów dla `frozen` / `last_block` / `full` (ResNet-18 i MobileNetV2), kształt wyjścia [B, 102] |
| `test_train_smoke.py` | kilka kroków treningu: loss maleje, wagi i statystyki BatchNorm zamrożonych warstw się nie zmieniają, zamrożone moduły są w trybie `eval` |

Testy modeli i treningu używają `pretrained=False`, więc nie pobierają wag z internetu.
