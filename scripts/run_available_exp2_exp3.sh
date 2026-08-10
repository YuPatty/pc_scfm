#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
RUN_ROOT="$ROOT_DIR/runs/ecg_baseline_wander"
DATA_ROOT="$ROOT_DIR/data/ecg_baseline_wander"
PTBXL_ROOT="$DATA_ROOT/raw/PTBXL"
NOISE_DIR="$DATA_ROOT/raw/NSTDB"

BATCH_SIZE="64"
DEVICE=""
FORCE=1
RUN_EXP2=1
RUN_EXP3=1
LIMIT=""
ALPHA_VALUES="0.05,0.1,0.2,0.3,0.5"
FREQUENCIES_HZ="0.05,0.1,0.2,0.3,0.5,0.8,1.0"
FREQUENCY_ALPHA="0.2"
EXP2_BASELINE_KIND="nstdb"
EXP3_BASELINE_KIND="sinusoidal"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_available_exp2_exp3.sh [options]

Runs Experiment 2 and Experiment 3 for every model in scripts/experiment_models.sh
when the required checkpoint and PTB-XL raw data are available.

Default behavior overwrites same-name Experiment 2/3 outputs.

Options:
  --skip-existing       Skip an experiment when its summary.csv already exists
  --only-exp2           Run Experiment 2 only
  --only-exp3           Run Experiment 3 only
  --batch-size N        Inference batch size. Default: 64
  --device DEVICE       Passed to experiment_suite.py, e.g. cuda:0 or cpu. Default: auto
  --limit N             Limit generated test windows, useful for smoke tests
  --alpha-values CSV    Exp2 alpha values. Default: 0.05,0.1,0.2,0.3,0.5
  --frequencies-hz CSV  Exp3 frequencies. Default: 0.05,0.1,0.2,0.3,0.5,0.8,1.0
  --frequency-alpha A   Exp3 alpha value. Default: 0.2
  -h, --help            Show this help

Required files:
  data/ecg_baseline_wander/raw/PTBXL/records100/
  data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv

Experiment 2 uses NSTDB by default and requires:
  data/ecg_baseline_wander/raw/NSTDB/

Experiment 3 uses synthetic sinusoidal baseline by default and does not require NSTDB.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-existing)
      FORCE=0
      shift
      ;;
    --only-exp2)
      RUN_EXP2=1
      RUN_EXP3=0
      shift
      ;;
    --only-exp3)
      RUN_EXP2=0
      RUN_EXP3=1
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --alpha-values)
      ALPHA_VALUES="$2"
      shift 2
      ;;
    --frequencies-hz)
      FREQUENCIES_HZ="$2"
      shift 2
      ;;
    --frequency-alpha)
      FREQUENCY_ALPHA="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

source "$ROOT_DIR/scripts/experiment_models.sh"

checkpoint_for() {
  local exp_name="$1"
  local model_dir="$2"
  printf '%s\n' "$RUN_ROOT/checkpoint/$exp_name/$model_dir/best_pcc_model.pt"
}

have_ptbxl_raw() {
  [[ -d "$PTBXL_ROOT/records100" && -f "$PTBXL_ROOT/ptbxl_database.csv" ]]
}

have_nstdb() {
  find "$NOISE_DIR" -type f -print -quit 2>/dev/null | grep -q .
}

common_args() {
  local config="$1"
  local checkpoint="$2"
  local output_root="$3"

  local args=(
    --config "$config"
    --input-dir "$PTBXL_ROOT/records100"
    --metadata-csv "$PTBXL_ROOT/ptbxl_database.csv"
    --checkpoint "$checkpoint"
    --output-root "$output_root"
    --batch-size "$BATCH_SIZE"
  )

  if [[ -n "$DEVICE" ]]; then
    args+=(--device "$DEVICE")
  fi
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi

  printf '%s\n' "${args[@]}"
}

run_exp2_if_available() {
  local model_key="$1"
  local config="$2"
  local checkpoint="$3"
  local output_root="$RUN_ROOT/controlled_tests/$model_key"
  local summary="$output_root/exp2_strength/summary.csv"

  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP exp2: $model_key: missing checkpoint $checkpoint"
    return 0
  fi
  if ! have_ptbxl_raw; then
    echo "SKIP exp2: $model_key: missing PTB-XL records100 or ptbxl_database.csv under $PTBXL_ROOT"
    return 0
  fi
  if [[ "$EXP2_BASELINE_KIND" == "nstdb" ]] && ! have_nstdb; then
    echo "SKIP exp2: $model_key: missing NSTDB files under $NOISE_DIR"
    return 0
  fi
  if [[ "$FORCE" -eq 0 && -f "$summary" ]]; then
    echo "SKIP exp2: $model_key: existing $summary"
    return 0
  fi

  echo "RUN exp2: $model_key"
  mapfile -t args < <(common_args "$config" "$checkpoint" "$output_root")
  if [[ "$EXP2_BASELINE_KIND" == "nstdb" ]]; then
    args+=(--noise-dir "$NOISE_DIR")
  fi
  python3 experiment_suite.py exp2-strength \
    "${args[@]}" \
    --baseline-kind "$EXP2_BASELINE_KIND" \
    --alpha-values "$ALPHA_VALUES"
}

run_exp3_if_available() {
  local model_key="$1"
  local config="$2"
  local checkpoint="$3"
  local output_root="$RUN_ROOT/controlled_tests/$model_key"
  local summary="$output_root/exp3_frequency/summary.csv"

  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP exp3: $model_key: missing checkpoint $checkpoint"
    return 0
  fi
  if ! have_ptbxl_raw; then
    echo "SKIP exp3: $model_key: missing PTB-XL records100 or ptbxl_database.csv under $PTBXL_ROOT"
    return 0
  fi
  if [[ "$FORCE" -eq 0 && -f "$summary" ]]; then
    echo "SKIP exp3: $model_key: existing $summary"
    return 0
  fi

  echo "RUN exp3: $model_key"
  mapfile -t args < <(common_args "$config" "$checkpoint" "$output_root")
  python3 experiment_suite.py exp3-frequency \
    "${args[@]}" \
    --baseline-kind "$EXP3_BASELINE_KIND" \
    --alpha-value "$FREQUENCY_ALPHA" \
    --frequencies-hz "$FREQUENCIES_HZ"
}

cd "$APP_DIR"

for item in "${EXPERIMENT_MODELS[@]}"; do
  IFS="|" read -r model_key config exp_name model_dir <<< "$item"
  checkpoint="$(checkpoint_for "$exp_name" "$model_dir")"

  if [[ "$RUN_EXP2" -eq 1 ]]; then
    run_exp2_if_available "$model_key" "$config" "$checkpoint"
  fi
  if [[ "$RUN_EXP3" -eq 1 ]]; then
    run_exp3_if_available "$model_key" "$config" "$checkpoint"
  fi
done

echo "Done."
