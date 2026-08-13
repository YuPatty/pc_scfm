import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from models import get_model
from utils_ecg import (
    centered_cosine_similarity,
    cosine_similarity,
    low_frequency_power_reduction,
    maximum_absolute_distance,
    nanmean_or_nan,
    prd,
    qrs_amplitude_error,
    r_peak_timing_error_ms,
    rr_interval_mae_ms,
    snr_db,
    snr_improvement_db,
    ssd,
)


METRIC_DISPLAY = {
    "SSD": ("SSD", "lower"),
    "MAD": ("MAD", "lower"),
    "PRD": ("PRD %", "lower"),
    "CosSim": ("CosSim", "higher"),
    "Output_SNR_dB": ("Output SNR dB", "higher"),
    "SNR_Improvement_dB": ("SNR Improvement dB", "higher"),
    "LF_Reduction_dB": ("Low-frequency Reduction dB", "higher"),
    "R_Peak_Timing_Error_ms": ("R-peak Timing Error ms", "lower"),
    "RR_Interval_MAE_ms": ("RR Interval MAE ms", "lower"),
    "RMSE": ("RMSE", "lower"),
    "Centered_CosSim": ("Centered Cosine Similarity", "higher"),
    "QRS_Amplitude_Error": ("QRS Amplitude Error", "lower"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECG baseline-wander-removal inference.")
    parser.add_argument("--config", default="configs/ecg_baseline_wander_mecg_e.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Input NPZ with noisy_ecg/input and optional clean_reference/target.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-frequency-high-hz", type=float, default=None)
    return parser.parse_args()


def as_single_lead(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError(f"Expected ECG shaped [N, T] or [N, 1, T], got {array.shape}.")
    return array


def load_npz(path):
    data = np.load(path)
    noisy_key = "noisy_ecg" if "noisy_ecg" in data else "input"
    if noisy_key not in data:
        raise KeyError(f"{path} must contain `noisy_ecg` or `input`.")

    clean_key = None
    if "clean_reference" in data:
        clean_key = "clean_reference"
    elif "target" in data:
        clean_key = "target"

    noisy = as_single_lead(data[noisy_key])
    clean = as_single_lead(data[clean_key]) if clean_key else None
    if clean is not None and len(clean) != len(noisy):
        raise ValueError(f"clean length {len(clean)} does not match noisy length {len(noisy)}.")
    return noisy, clean


def run_inference(model, noisy, device, batch_size):
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(noisy), batch_size):
            batch = torch.from_numpy(noisy[start:start + batch_size]).to(device)
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def load_model_checkpoint(model, checkpoint_path, device):
    if str(checkpoint_path).lower() in {"none", "null", "__classical_filter_no_checkpoint__"}:
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def metric_rows(noisy, restored, clean, fs, low_freq_hz, eps):
    noisy_2d = np.squeeze(noisy, axis=1)
    restored_2d = np.squeeze(restored, axis=1)
    lf_reduction = low_frequency_power_reduction(
        noisy_2d, restored_2d, fs=fs, high_hz=low_freq_hz, eps=eps
    )

    rows = [{"window_index": idx, "LF_Reduction_dB": float(value)} for idx, value in enumerate(lf_reduction)]
    if clean is None:
        summary = {
            "LF_Reduction_dB": {
                "mean": float(np.mean(lf_reduction)),
                "std": float(np.std(lf_reduction)),
                "count": int(len(lf_reduction)),
            }
        }
        return rows, summary, False

    clean_2d = np.squeeze(clean, axis=1)
    metric_values = {
        "SSD": ssd(clean_2d, restored_2d),
        "MAD": maximum_absolute_distance(clean_2d, restored_2d),
        "PRD": prd(clean_2d, restored_2d, eps=eps),
        "CosSim": cosine_similarity(clean_2d, restored_2d, eps=eps),
        "Output_SNR_dB": snr_db(clean_2d, restored_2d, eps=eps),
        "SNR_Improvement_dB": snr_improvement_db(clean_2d, noisy_2d, restored_2d, eps=eps),
        "LF_Reduction_dB": lf_reduction,
        "R_Peak_Timing_Error_ms": r_peak_timing_error_ms(clean_2d, restored_2d, fs=fs),
        "RR_Interval_MAE_ms": rr_interval_mae_ms(clean_2d, restored_2d, fs=fs),
        "RMSE": np.sqrt(np.mean((clean_2d - restored_2d) ** 2, axis=-1)),
        "Centered_CosSim": centered_cosine_similarity(clean_2d, restored_2d, eps=eps),
        "QRS_Amplitude_Error": qrs_amplitude_error(clean_2d, restored_2d, fs=fs),
    }

    rows = []
    for idx in range(len(noisy_2d)):
        row = {"window_index": idx}
        for key, values in metric_values.items():
            row[key] = float(values[idx]) if not np.isnan(values[idx]) else ""
        rows.append(row)

    summary = {}
    for key, values in metric_values.items():
        values = np.asarray(values, dtype=np.float64)
        valid = values[~np.isnan(values)]
        summary[key] = {
            "mean": nanmean_or_nan(values),
            "std": float(np.std(valid)) if valid.size else float("nan"),
            "count": int(valid.size),
        }
    return rows, summary, True


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output_dir, noisy, restored, clean, rows, summary, has_reference):
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"noisy_ecg": noisy, "restored_ecg": restored}
    if clean is not None:
        arrays["clean_reference"] = clean
    np.savez(output_dir / "restored_ecg.npz", **arrays)

    fieldnames = ["window_index"] + [key for key in METRIC_DISPLAY if key in rows[0]]
    write_csv(output_dir / "metrics_per_window.csv", rows, fieldnames)

    summary_rows = []
    for key, stats in summary.items():
        display, direction = METRIC_DISPLAY.get(key, (key, ""))
        summary_rows.append(
            {
                "metric": key,
                "display_name": display,
                "direction": direction,
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
            }
        )
    write_csv(
        output_dir / "metrics_summary.csv",
        summary_rows,
        ["metric", "display_name", "direction", "mean", "std", "count"],
    )
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "has_clean_reference": has_reference,
                "metrics": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(summary, has_reference):
    if not has_reference:
        print("clean_reference not found; only no-reference metrics were computed.")
    print(f"{'Metric':>30} | {'Mean':>12} | {'Std':>12} | {'N':>8} | Direction")
    print("-" * 86)
    for key, stats in summary.items():
        display, direction = METRIC_DISPLAY.get(key, (key, ""))
        print(
            f"{display:>30} | {stats['mean']:>12.4f} | {stats['std']:>12.4f} | "
            f"{stats['count']:>8} | {direction}"
        )


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)

    noisy, clean = load_npz(args.input)
    model = get_model(cfg.model_name, **OmegaConf.to_container(cfg.model, resolve=True)).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    restored = run_inference(model, noisy, device=device, batch_size=args.batch_size)

    fs = cfg.dataset.get("resample_hz", cfg.model.get("sampling_rate", 250))
    low_freq_hz = (
        args.low_frequency_high_hz
        if args.low_frequency_high_hz is not None
        else cfg.get("evaluation", {}).get("low_frequency_high_hz", 0.5)
    )
    eps = cfg.dataset.get("eps", 1e-10)
    rows, summary, has_reference = metric_rows(noisy, restored, clean, fs, low_freq_hz, eps)
    write_outputs(output_dir, noisy, restored, clean, rows, summary, has_reference)
    print_summary(summary, has_reference)
    print(f"saved restored ECG and metric tables to: {output_dir}")


if __name__ == "__main__":
    main()
