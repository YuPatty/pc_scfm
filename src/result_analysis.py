import argparse
import csv
import math
from pathlib import Path

import yaml


METRIC_DIRECTIONS = {
    "SSD": "lower",
    "MAD": "lower",
    "PRD": "lower",
    "CosSim": "higher",
    "Output_SNR_dB": "higher",
    "SNR_Improvement_dB": "higher",
    "LF_Reduction_dB": "higher",
    "R_Peak_Timing_Error_ms": "lower",
    "RR_Interval_MAE_ms": "lower",
    "RMSE": "lower",
    "Centered_CosSim": "higher",
    "QRS_Amplitude_Error": "lower",
}


def require_numpy():
    import numpy as np

    return np


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate ECG experiment results and reports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate = subparsers.add_parser("aggregate", help="Collect paper-ready mean ± std tables.")
    aggregate.add_argument("--results-root", required=True)
    aggregate.add_argument("--output", required=True)

    cases = subparsers.add_parser("select-cases", help="Plot best/median/worst cases from inference outputs.")
    cases.add_argument("--inference-dir", required=True)
    cases.add_argument("--output-dir", required=True)
    cases.add_argument("--metric", default="PRD")
    cases.add_argument("--direction", choices=["lower", "higher"], default=None)
    cases.add_argument("--fs", type=float, default=250.0)

    paired = subparsers.add_parser("paired-stats", help="Paired model comparison with correction.")
    paired.add_argument("--baseline", required=True, help="Baseline metrics_per_window.csv.")
    paired.add_argument("--candidate", required=True, help="Candidate metrics_per_window.csv.")
    paired.add_argument("--output", required=True)
    paired.add_argument("--metrics", default=None, help="Comma-separated metrics. Defaults to shared numeric metrics.")
    paired.add_argument("--correction", choices=["holm", "bonferroni", "none"], default="holm")
    paired.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_yaml_metrics(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        key: value
        for key, value in data.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(results_root, output):
    results_root = Path(results_root)
    rows = []

    for summary_path in sorted(results_root.rglob("metrics_summary.csv")):
        summary_rows = read_csv_rows(summary_path)
        context = infer_context(summary_path, results_root)
        for row in summary_rows:
            metric = row["metric"]
            mean = parse_float(row.get("mean"))
            std = parse_float(row.get("std"))
            count = row.get("count", "")
            rows.append(
                {
                    **context,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "count": count,
                    "mean_std": format_mean_std(mean, std),
                    "direction": row.get("direction") or METRIC_DIRECTIONS.get(metric, ""),
                    "source": str(summary_path),
                }
            )

    for summary_path in sorted(results_root.rglob("summary.csv")):
        if summary_path.name == "metrics_summary.csv":
            continue
        for row in read_csv_rows(summary_path):
            context = infer_context(summary_path, results_root)
            for key, value in row.items():
                if not key.endswith("_mean"):
                    continue
                metric = key[: -len("_mean")]
                mean = parse_float(value)
                std = parse_float(row.get(f"{metric}_std"))
                count = row.get(f"{metric}_count", "")
                rows.append(
                    {
                        **context,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "count": count,
                        "mean_std": format_mean_std(mean, std),
                        "direction": METRIC_DIRECTIONS.get(metric, ""),
                        "source": str(summary_path),
                }
            )

    for summary_path in sorted(results_root.rglob("metrics_*.yaml")):
        context = infer_context(summary_path, results_root)
        context["dataset_or_split"] = summary_path.stem.removeprefix("metrics_")
        for metric, value in read_yaml_metrics(summary_path).items():
            mean = parse_float(value)
            rows.append(
                {
                    **context,
                    "metric": metric,
                    "mean": mean,
                    "std": "",
                    "count": "",
                    "mean_std": format_mean_std(mean, float("nan")),
                    "direction": METRIC_DIRECTIONS.get(metric, ""),
                    "source": str(summary_path),
                }
            )

    for summary_path in sorted(results_root.rglob("complexity_summary.yaml")):
        context = infer_context(summary_path, results_root)
        context["dataset_or_split"] = "complexity"
        for metric, value in read_yaml_metrics(summary_path).items():
            mean = parse_float(value)
            rows.append(
                {
                    **context,
                    "metric": metric,
                    "mean": mean,
                    "std": "",
                    "count": "",
                    "mean_std": format_mean_std(mean, float("nan")),
                    "direction": METRIC_DIRECTIONS.get(metric, ""),
                    "source": str(summary_path),
                }
            )

    fieldnames = [
        "experiment",
        "model_or_condition",
        "checkpoint",
        "dataset_or_split",
        "metric",
        "mean",
        "std",
        "count",
        "mean_std",
        "direction",
        "source",
    ]
    write_csv(output, rows, fieldnames)
    print(f"saved aggregate table -> {output}")


def infer_context(path, root):
    rel = path.relative_to(root)
    parts = rel.parts
    experiment = parts[0] if parts else ""
    model_or_condition = ""
    dataset_or_split = ""
    checkpoint = ""

    if "exp2_strength" in parts or "exp3_frequency" in parts:
        experiment = "Experiment 2" if "exp2_strength" in parts else "Experiment 3"
        idx = parts.index("exp2_strength") if "exp2_strength" in parts else parts.index("exp3_frequency")
        model_or_condition = parts[idx + 1] if idx + 1 < len(parts) else ""
        dataset_or_split = "ptbxl_fold10_test"
    elif "best_pcc" in parts or "best_loss" in parts:
        ckpt = "best_pcc" if "best_pcc" in parts else "best_loss"
        idx = parts.index(ckpt)
        experiment = parts[idx - 2] if idx >= 2 else experiment
        model_or_condition = parts[idx - 1] if idx >= 1 else ""
        checkpoint = ckpt
        dataset_or_split = ckpt
    elif path.name == "summary.csv" and "exp7_ablation" in parts:
        experiment = "Experiment 7"

    return {
        "experiment": experiment,
        "model_or_condition": model_or_condition,
        "checkpoint": checkpoint,
        "dataset_or_split": dataset_or_split,
    }


def select_cases(inference_dir, output_dir, metric, direction, fs):
    np = require_numpy()
    inference_dir = Path(inference_dir)
    output_dir = Path(output_dir)
    rows = read_csv_rows(inference_dir / "metrics_per_window.csv")
    arrays = np.load(inference_dir / "restored_ecg.npz")
    noisy = squeeze_lead(arrays["noisy_ecg"])
    restored = squeeze_lead(arrays["restored_ecg"])
    clean = squeeze_lead(arrays["clean_reference"]) if "clean_reference" in arrays else None

    values = []
    for row in rows:
        value = parse_float(row.get(metric))
        if not math.isnan(value):
            values.append((int(row["window_index"]), value))
    if not values:
        raise ValueError(f"No numeric values found for metric {metric}.")

    direction = direction or METRIC_DIRECTIONS.get(metric, "lower")
    values.sort(key=lambda item: item[1], reverse=(direction == "higher"))
    selected = [
        ("best", values[0]),
        ("median", values[len(values) // 2]),
        ("worst", values[-1]),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for label, (idx, value) in selected:
        out_path = output_dir / f"{label}_{metric}_window_{idx}.png"
        plot_case(out_path, idx, value, metric, noisy[idx], restored[idx], clean[idx] if clean is not None else None, fs)
        report_rows.append({"case": label, "window_index": idx, "metric": metric, "value": value, "plot": str(out_path)})

    write_csv(output_dir / "selected_cases.csv", report_rows, ["case", "window_index", "metric", "value", "plot"])
    print(f"saved selected cases -> {output_dir}")


def squeeze_lead(array):
    np = require_numpy()
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected [N, 1, T] or [N, T], got {array.shape}.")


def plot_case(path, idx, value, metric, noisy, restored, clean, fs):
    np = require_numpy()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(noisy)) / fs
    nrows = 4 if clean is not None else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, noisy, color="tab:gray", linewidth=1.0)
    axes[0].set_ylabel("Input")
    if clean is not None:
        axes[1].plot(t, clean, color="tab:blue", linewidth=1.0, label="Clean")
        axes[1].plot(t, restored, color="tab:red", linewidth=1.0, alpha=0.85, label="Restored")
        axes[1].legend(loc="best")
        axes[1].set_ylabel("ECG")
        axes[2].plot(t, clean - restored, color="tab:purple", linewidth=1.0)
        axes[2].set_ylabel("Residual")
        spec_ax = axes[3]
    else:
        axes[1].plot(t, restored, color="tab:red", linewidth=1.0)
        axes[1].set_ylabel("Restored")
        spec_ax = axes[2]
    spec_ax.psd(noisy, Fs=fs, color="tab:gray", label="Input")
    spec_ax.psd(restored, Fs=fs, color="tab:red", label="Restored")
    spec_ax.set_ylabel("PSD")
    spec_ax.legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Second")
    axes[0].set_title(f"Window {idx} | {metric}={value:.4f}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def paired_stats(baseline_path, candidate_path, output, metrics, correction, alpha):
    np = require_numpy()
    from scipy import stats

    baseline_rows = keyed_rows(read_csv_rows(baseline_path))
    candidate_rows = keyed_rows(read_csv_rows(candidate_path))
    shared_indices = sorted(set(baseline_rows) & set(candidate_rows))
    if not shared_indices:
        raise ValueError("No shared window_index values between baseline and candidate files.")

    metric_names = (
        [metric.strip() for metric in metrics.split(",") if metric.strip()]
        if metrics
        else infer_shared_numeric_metrics(baseline_rows, candidate_rows, shared_indices)
    )

    rows = []
    for metric in metric_names:
        baseline_values, candidate_values = [], []
        for idx in shared_indices:
            b = parse_float(baseline_rows[idx].get(metric))
            c = parse_float(candidate_rows[idx].get(metric))
            if not math.isnan(b) and not math.isnan(c):
                baseline_values.append(b)
                candidate_values.append(c)
        if len(baseline_values) < 2:
            continue
        b = np.asarray(baseline_values, dtype=np.float64)
        c = np.asarray(candidate_values, dtype=np.float64)
        diff = c - b
        t_stat, t_p = stats.ttest_rel(c, b, nan_policy="omit")
        try:
            w_stat, w_p = stats.wilcoxon(c, b)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        rows.append(
            {
                "metric": metric,
                "direction": METRIC_DIRECTIONS.get(metric, ""),
                "n": len(diff),
                "baseline_mean": float(np.mean(b)),
                "candidate_mean": float(np.mean(c)),
                "mean_difference_candidate_minus_baseline": float(np.mean(diff)),
                "paired_t_stat": float(t_stat),
                "paired_t_p": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
            }
        )

    corrected = correct_pvalues([row["wilcoxon_p"] for row in rows], correction)
    for row, corrected_p in zip(rows, corrected):
        row["corrected_p"] = corrected_p
        row["significant"] = bool(corrected_p <= alpha) if not math.isnan(corrected_p) else False
        row["correction"] = correction
        row["alpha"] = alpha

    fieldnames = [
        "metric",
        "direction",
        "n",
        "baseline_mean",
        "candidate_mean",
        "mean_difference_candidate_minus_baseline",
        "paired_t_stat",
        "paired_t_p",
        "wilcoxon_stat",
        "wilcoxon_p",
        "corrected_p",
        "significant",
        "correction",
        "alpha",
    ]
    write_csv(output, rows, fieldnames)
    print(f"saved paired statistics -> {output}")


def keyed_rows(rows):
    return {int(row["window_index"]): row for row in rows if row.get("window_index", "").strip() != ""}


def infer_shared_numeric_metrics(baseline_rows, candidate_rows, indices):
    keys = set()
    for idx in indices[: min(10, len(indices))]:
        keys.update(baseline_rows[idx].keys())
        keys.update(candidate_rows[idx].keys())
    keys.discard("window_index")
    metrics = []
    for key in sorted(keys):
        for idx in indices:
            b = parse_float(baseline_rows[idx].get(key))
            c = parse_float(candidate_rows[idx].get(key))
            if not math.isnan(b) and not math.isnan(c):
                metrics.append(key)
                break
    return metrics


def correct_pvalues(pvalues, correction):
    np = require_numpy()
    pvalues = np.asarray(pvalues, dtype=np.float64)
    if correction == "none":
        return pvalues.tolist()
    valid = ~np.isnan(pvalues)
    corrected = np.full_like(pvalues, np.nan)
    if not np.any(valid):
        return corrected.tolist()
    valid_indices = np.flatnonzero(valid)
    valid_p = pvalues[valid]
    if correction == "bonferroni":
        corrected[valid] = np.minimum(valid_p * len(valid_p), 1.0)
        return corrected.tolist()

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    m = len(sorted_p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, pvalue in enumerate(sorted_p, start=1):
        running = max(running, min((m - rank + 1) * pvalue, 1.0))
        adjusted[rank - 1] = running
    for sorted_pos, original_pos in enumerate(order):
        corrected[valid_indices[original_pos]] = adjusted[sorted_pos]
    return corrected.tolist()


def parse_float(value):
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_mean_std(mean, std):
    if math.isnan(mean):
        return ""
    if math.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def main():
    args = parse_args()
    if args.command == "aggregate":
        aggregate_results(args.results_root, args.output)
    elif args.command == "select-cases":
        select_cases(args.inference_dir, args.output_dir, args.metric, args.direction, args.fs)
    elif args.command == "paired-stats":
        paired_stats(args.baseline, args.candidate, args.output, args.metrics, args.correction, args.alpha)


if __name__ == "__main__":
    main()
