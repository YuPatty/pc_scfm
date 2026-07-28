#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"

MODEL="${1:-pc_scfm}"
shift || true

case "$MODEL" in
  pc_scfm)
    CONFIG="configs/ecg_baseline_wander_pc_scfm.yaml"
    ;;
  mecg_e)
    CONFIG="configs/ecg_baseline_wander_mecg_e.yaml"
    ;;
  mambattention|mambattention_ecg)
    CONFIG="configs/ecg_baseline_wander_mambattention.yaml"
    ;;
  *)
    echo "Unknown model: $MODEL" >&2
    echo "Expected one of: pc_scfm, mecg_e, mambattention" >&2
    exit 2
    ;;
esac

cd "$APP_DIR"
python3 train_supervised.py --config "$CONFIG" "$@"
