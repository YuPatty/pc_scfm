#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT_DIR"

python3 - <<'PY'
from pathlib import Path
import sys
import yaml


CONFIG_DIR = Path("src/configs")
CONFIGS = sorted(CONFIG_DIR.glob("ecg_baseline_wander*.yaml"))

EXPECTED = {
    ("data_dir",): "../data/ecg_baseline_wander",
    ("root_dir",): "../runs/ecg_baseline_wander",
    ("checkpoint_dir",): "${root_dir}/checkpoint",
    ("results_dir",): "${root_dir}/results",
    ("log_dir",): "${root_dir}/log",
    ("seed",): 42,
    ("training", "batch_size"): 32,
    ("training", "train_iterations"): 20000,
    ("training", "eval_every"): 100,
    ("training", "validation_metrics_every"): 500,
    ("training", "save_every"): 5000,
    ("training", "lr"): 1.0e-4,
    ("training", "early_stopping_patience"): 25,
    ("training", "early_stopping_min_delta"): 1.0e-4,
    ("training", "resume"): False,
    ("training", "resume_checkpoint"): None,
    ("evaluation", "low_frequency_high_hz"): 0.5,
    ("dataset", "name"): "ecg_baseline_wander",
    ("dataset", "data_dir"): "${data_dir}",
    ("dataset", "train_dataset"): "ptbxl",
    ("dataset", "external_test_datasets"): ["mit_bih", "chapman", "cpsc"],
    ("dataset", "lead"): "II",
    ("dataset", "split", "strategy"): "ptbxl_official_folds",
    ("dataset", "split", "patient_wise"): True,
    ("dataset", "split", "train_folds"): [1, 2, 3, 4, 5, 6, 7, 8],
    ("dataset", "split", "validation_fold"): 9,
    ("dataset", "split", "test_fold"): 10,
    ("dataset", "split", "ratio"): [0.8, 0.1, 0.1],
    ("dataset", "clean_reference", "filter_type"): "butterworth",
    ("dataset", "clean_reference", "filter_order"): 4,
    ("dataset", "clean_reference", "zero_phase"): True,
    ("dataset", "clean_reference", "bandpass_hz"): [0.05, 40.0],
    ("dataset", "resample_hz"): 250,
    ("dataset", "window_size"): 512,
    ("dataset", "overlap_ratio"): 0.5,
    ("dataset", "normalization"): "z_score",
    ("dataset", "baseline_wander", "train_source"): "nstdb",
    ("dataset", "baseline_wander", "alpha_mode"): "peak_to_peak_ratio",
    ("dataset", "baseline_wander", "alpha_values"): [0.05, 0.1, 0.2, 0.3, 0.5],
    ("dataset", "baseline_wander", "controlled_frequencies_hz"): [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
    ("dataset", "baseline_wander", "test_types"): [
        "nstdb",
        "sinusoidal",
        "multi_sine",
        "random_low_frequency_drift",
    ],
    ("dataset", "eps"): 1.0e-10,
    ("model", "fea"): "pha",
    ("model", "norm"): False,
    ("model", "compress_factor"): 0.3,
    ("model", "num_tscblocks"): 4,
    ("model", "d_state"): 16,
    ("model", "d_conv"): 4,
    ("model", "expand"): 4,
    ("model", "norm_epsilon"): 1.0e-5,
    ("model", "sampling_rate"): 250,
    ("model", "n_fft"): 64,
    ("model", "hop_size"): 8,
    ("model", "win_size"): 64,
    ("model", "beta"): 2,
}

EXPECTED_BY_MODEL = {
    "mecg_e": {
        ("model", "dense_channel"): 32,
        ("model", "fmamba"): True,
        ("model", "loss_fn"): "time+com+con",
    },
    "mambattention_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con",
    },
    "mambattention_stfrft_ecg": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "loss_fn"): "time+com+con",
        ("model", "time_frequency_transform"): "stfrft",
        ("model", "learnable_frft_order"): True,
        ("model", "frft_order_init"): 0.9,
        ("model", "frft_order_min"): 0.05,
        ("model", "frft_order_max"): 1.95,
    },
    "pc_scfm": {
        ("model", "dense_channel"): 64,
        ("model", "attention_heads"): 8,
        ("model", "attention_dropout"): 0.0,
        ("model", "attention_position"): "before_mamba",
        ("model", "use_time_attention"): True,
        ("model", "use_freq_attention"): True,
        ("model", "pcscfm_enabled"): True,
    },
    "eddm": {
        ("model", "dense_channel"): 64,
        ("model", "timesteps"): 50,
        ("model", "inference_steps"): 10,
        ("model", "base_channels"): 64,
        ("model", "channel_mults"): [1, 2, 4],
        ("model", "time_dim"): 256,
        ("model", "dropout"): 0.0,
        ("model", "gaussian_scale"): 0.2,
        ("model", "ecg_noise_weight"): 1.0,
        ("model", "gaussian_noise_weight"): 0.1,
        ("model", "clean_weight"): 1.0,
        ("model", "pcc_weight"): 0.1,
        ("model", "pool_scales"): [2, 4, 8],
    },
    "fir_filter": {
        ("model", "dense_channel"): 64,
        ("model", "filter_kind"): "fir",
        ("model", "cutoff_hz"): 0.5,
        ("model", "fir_order"): 56,
        ("model", "fir_window"): "kaiser",
        ("model", "kaiser_beta"): 8.6,
        ("model", "zero_phase"): True,
    },
    "iir_filter": {
        ("model", "dense_channel"): 64,
        ("model", "filter_kind"): "iir",
        ("model", "cutoff_hz"): 0.5,
        ("model", "iir_order"): 1,
        ("model", "iir_method"): "butterworth",
        ("model", "zero_phase"): True,
    },
    "drnn": {
        ("model", "dense_channel"): 64,
        ("model", "input_size"): 1,
        ("model", "hidden_size"): 64,
        ("model", "lstm_layers"): 1,
        ("model", "dense_layers"): [64, 64],
        ("model", "dropout"): 0.0,
        ("model", "residual"): False,
        ("model", "loss_fn"): "mse",
    },
}

EXPECTED_MAMBATTENTION_VARIANTS = {
    "ecg_baseline_wander_mambattention.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_no_time_attention.yaml": ("before_mamba", False, True),
    "ecg_baseline_wander_mambattention_no_freq_attention.yaml": ("before_mamba", True, False),
    "ecg_baseline_wander_mambattention_post_attention.yaml": ("after_mamba", True, True),
    "ecg_baseline_wander_mambattention_post_no_time_attention.yaml": ("after_mamba", False, True),
    "ecg_baseline_wander_mambattention_post_no_freq_attention.yaml": ("after_mamba", True, False),
    "ecg_baseline_wander_mambattention_stfrft.yaml": ("before_mamba", True, True),
    "ecg_baseline_wander_mambattention_stfrft_no_time_attention.yaml": ("before_mamba", False, True),
    "ecg_baseline_wander_mambattention_stfrft_no_freq_attention.yaml": ("before_mamba", True, False),
    "ecg_baseline_wander_mambattention_stfrft_no_attention.yaml": ("before_mamba", False, False),
}

EXPECTED_PCSCFM_VARIANTS = {
    "ecg_baseline_wander_pc_scfm.yaml": {
        ("model", "phase_representation"): "raw",
        ("model", "use_flow_proposal"): None,
        ("model", "flow_nfe"): 4,
        ("model", "flow_samples"): 4,
        ("model", "loss_fn"): "time+com+con+lf+morph+flow+bc+value+risk",
        ("model", "lambda_flow"): 0.01,
    },
    "ecg_baseline_wander_pc_scfm_rl_no_flow.yaml": {
        ("model", "phase_representation"): "raw",
        ("model", "use_flow_proposal"): False,
        ("model", "flow_nfe"): 1,
        ("model", "flow_samples"): 1,
        ("model", "loss_fn"): "time+com+con+lf+morph+bc+value+risk",
        ("model", "lambda_flow"): 0.0,
    },
}


def get_value(data, path):
    value = data
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def check_value(errors, data, path, expected, config_name):
    actual = get_value(data, path)
    if actual != expected:
        errors.append(
            f"{config_name}: {'.'.join(path)} expected {expected!r}, got {actual!r}"
        )


def check_optional_value(errors, data, path, expected, config_name):
    actual = get_value(data, path)
    if actual not in {None, expected}:
        errors.append(
            f"{config_name}: {'.'.join(path)} expected omitted or {expected!r}, got {actual!r}"
        )


errors = []
if not CONFIGS:
    errors.append("No configs found.")

for path in CONFIGS:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.name
    model_name = data.get("model_name")

    for key_path, expected in EXPECTED.items():
        check_value(errors, data, key_path, expected, name)

    for key_path, expected in EXPECTED_BY_MODEL.get(model_name, {}).items():
        check_value(errors, data, key_path, expected, name)

    if model_name in {"mambattention_ecg", "mambattention_stfrft_ecg"}:
        expected_variant = EXPECTED_MAMBATTENTION_VARIANTS.get(name)
        if expected_variant is None:
            errors.append(f"{name}: unknown MambAttention variant config name.")
        else:
            position, use_time, use_freq = expected_variant
            check_value(errors, data, ("model", "attention_position"), position, name)
            check_value(errors, data, ("model", "use_time_attention"), use_time, name)
            check_value(errors, data, ("model", "use_freq_attention"), use_freq, name)

    if model_name == "pc_scfm":
        expected_variant = EXPECTED_PCSCFM_VARIANTS.get(name)
        if expected_variant is None:
            errors.append(f"{name}: unknown PC-SCFM variant config name.")
        else:
            for key_path, expected in expected_variant.items():
                if key_path == ("model", "use_flow_proposal") and expected is None:
                    check_optional_value(errors, data, key_path, True, name)
                else:
                    check_value(errors, data, key_path, expected, name)
            loss_terms = set(str(get_value(data, ("model", "loss_fn"))).split("+"))
            uses_flow = get_value(data, ("model", "use_flow_proposal"))
            if uses_flow is False and "flow" in loss_terms:
                errors.append(f"{name}: no-flow PC-SCFM must not include 'flow' in model.loss_fn.")

if errors:
    print("Experiment config check failed:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print(f"Experiment config check OK ({len(CONFIGS)} configs).")
PY
