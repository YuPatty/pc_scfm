#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/src"
RUN_ROOT="$ROOT_DIR/runs/ecg_baseline_wander"
DATA_DIR="$ROOT_DIR/data/ecg_baseline_wander/processed"

BASELINE_MODEL="mecge"
BATCH_SIZE="64"
DEVICE=""
FORCE=1
SKIP_INFERENCE=0
SKIP_STATS=0
CORRECTION="holm"
METRICS="SSD,MAD,PRD,CosSim,SNR_Improvement_dB,LF_Reduction_dB,R_Peak_Timing_Error_ms,RR_Interval_MAE_ms,QRS_Amplitude_Error"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_available_inference_and_stats.sh [options]

Runs inference for every model in scripts/experiment_models.sh when both:
  1. best_pcc_model.pt exists
  2. the requested processed dataset NPZ exists

Then runs paired statistics against the baseline model for every dataset where
both models have metrics_per_window.csv.

Options:
  --baseline NAME       Baseline model key from experiment_models.sh. Default: mecge
  --batch-size N        Inference batch size. Default: 64
  --device DEVICE       Passed to inference.py, e.g. cuda:0 or cpu. Default: auto
  --force              Re-run inference even if metrics_per_window.csv exists. Default behavior
  --skip-existing      Skip inference outputs that already have metrics_per_window.csv
  --skip-inference      Only run statistics from existing inference outputs
  --skip-stats          Only run inference
  --correction METHOD   none, bonferroni, or holm. Default: holm
  --metrics CSV         Metrics for paired-stats
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      BASELINE_MODEL="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --skip-existing)
      FORCE=0
      shift
      ;;
    --skip-inference)
      SKIP_INFERENCE=1
      shift
      ;;
    --skip-stats)
      SKIP_STATS=1
      shift
      ;;
    --correction)
      CORRECTION="$2"
      shift 2
      ;;
    --metrics)
      METRICS="$2"
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

DATASETS=(
  "ptbxl_fold10|$DATA_DIR/test.npz"
  "mit_bih|$DATA_DIR/mit_bih.npz"
  "chapman|$DATA_DIR/chapman.npz"
  "cpsc|$DATA_DIR/cpsc.npz"
)

checkpoint_for() {
  local exp_name="$1"
  local model_dir="$2"
  printf '%s\n' "$RUN_ROOT/checkpoint/$exp_name/$model_dir/best_pcc_model.pt"
}

inference_dir_for() {
  local model_key="$1"
  local dataset_key="$2"
  printf '%s\n' "$RUN_ROOT/inference/${model_key}_${dataset_key}"
}

run_inference_if_available() {
  local model_key="$1"
  local config="$2"
  local exp_name="$3"
  local model_dir="$4"
  local dataset_key="$5"
  local dataset_path="$6"

  local checkpoint
  local output_dir
  checkpoint="$(checkpoint_for "$exp_name" "$model_dir")"
  output_dir="$(inference_dir_for "$model_key" "$dataset_key")"

  if [[ ! -f "$checkpoint" ]]; then
    echo "SKIP inference: $model_key / $dataset_key: missing checkpoint $checkpoint"
    return 0
  fi
  if [[ ! -f "$dataset_path" ]]; then
    echo "SKIP inference: $model_key / $dataset_key: missing dataset $dataset_path"
    return 0
  fi
  if [[ "$FORCE" -eq 0 && -f "$output_dir/metrics_per_window.csv" ]]; then
    echo "SKIP inference: $model_key / $dataset_key: existing $output_dir/metrics_per_window.csv"
    return 0
  fi

  echo "RUN inference: $model_key / $dataset_key"
  mkdir -p "$output_dir"

  local cmd=(
    python3 inference.py
    --config "$config"
    --checkpoint "$checkpoint"
    --input "$dataset_path"
    --output-dir "$output_dir"
    --batch-size "$BATCH_SIZE"
  )
  if [[ -n "$DEVICE" ]]; then
    cmd+=(--device "$DEVICE")
  fi
  "${cmd[@]}"
}

run_stats_if_available() {
  local candidate_key="$1"
  local dataset_key="$2"

  if [[ "$candidate_key" == "$BASELINE_MODEL" ]]; then
    return 0
  fi

  local baseline_csv
  local candidate_csv
  local output_csv
  baseline_csv="$(inference_dir_for "$BASELINE_MODEL" "$dataset_key")/metrics_per_window.csv"
  candidate_csv="$(inference_dir_for "$candidate_key" "$dataset_key")/metrics_per_window.csv"
  output_csv="$RUN_ROOT/statistics/${BASELINE_MODEL}_vs_${candidate_key}_${dataset_key}.csv"

  if [[ ! -f "$baseline_csv" ]]; then
    echo "SKIP stats: $BASELINE_MODEL vs $candidate_key / $dataset_key: missing $baseline_csv"
    return 0
  fi
  if [[ ! -f "$candidate_csv" ]]; then
    echo "SKIP stats: $BASELINE_MODEL vs $candidate_key / $dataset_key: missing $candidate_csv"
    return 0
  fi

  echo "RUN stats: $BASELINE_MODEL vs $candidate_key / $dataset_key"
  mkdir -p "$(dirname "$output_csv")"
  python3 result_analysis.py paired-stats \
    --baseline "$baseline_csv" \
    --candidate "$candidate_csv" \
    --output "$output_csv" \
    --metrics "$METRICS" \
    --correction "$CORRECTION"
}

cd "$APP_DIR"

if [[ "$SKIP_INFERENCE" -eq 0 ]]; then
  for item in "${EXPERIMENT_MODELS[@]}"; do
    IFS="|" read -r model_key config exp_name model_dir <<< "$item"
    for dataset in "${DATASETS[@]}"; do
      IFS="|" read -r dataset_key dataset_path <<< "$dataset"
      run_inference_if_available "$model_key" "$config" "$exp_name" "$model_dir" "$dataset_key" "$dataset_path"
    done
  done
fi

if [[ "$SKIP_STATS" -eq 0 ]]; then
  for item in "${EXPERIMENT_MODELS[@]}"; do
    IFS="|" read -r model_key _config _exp_name _model_dir <<< "$item"
    for dataset in "${DATASETS[@]}"; do
      IFS="|" read -r dataset_key _dataset_path <<< "$dataset"
      run_stats_if_available "$model_key" "$dataset_key"
    done
  done
fi

echo "Done."
