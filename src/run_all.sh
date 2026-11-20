#!/usr/bin/env bash
# Trening wszystkich wariantów na tym samym podziale (seed 6102), potem ewaluacja.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
for run in r18_frozen r18_last_block r18_full mnv2_frozen mnv2_full; do
  "$PY" train.py --run "$run"
done
"$PY" evaluate.py
