#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/ecg_baseline_wander}"

echo "Project root: $ROOT_DIR"
echo "Data root:    $DATA_DIR"
echo

echo "Required data files:"
for file in train.npz val.npz test.npz; do
  path="$DATA_DIR/processed/$file"
  if [[ -f "$path" ]]; then
    echo "  OK      $path"
  else
    echo "  MISSING $path"
  fi
done

echo
echo "Python syntax check:"
cd "$ROOT_DIR"
python3 -X pycache_prefix=/tmp/rl_exp_pycache -m py_compile \
  src/models/mecg_e.py \
  src/models/mambattention.py \
  src/models/pc_scfm.py \
  src/models/pc_scfm_components.py \
  src/models/eddm.py \
  src/models/drnn.py \
  src/models/fcn_dae.py \
  src/models/deepfilter.py \
  src/models/descod_ecg.py \
  src/models/classical_filters.py \
  src/experiment_suite.py \
  src/train_supervised.py \
  src/inference.py
echo "  OK"
