import argparse
import csv
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiment 2/3/7 workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    exp2 = subparsers.add_parser("exp2-strength", help="Baseline strength robustness sweep.")
    add_generation_args(exp2)
    add_inference_args(exp2)
    exp2.add_argument("--baseline-kind", default="nstdb")
    exp2.add_argument("--alpha-values", default=None)

    exp3 = subparsers.add_parser("exp3-frequency", help="Baseline frequency robustness sweep.")
    add_generation_args(exp3)
    add_inference_args(exp3)
    exp3.add_argument("--baseline-kind", default="sinusoidal")
    exp3.add_argument("--frequencies-hz", default=None)
    exp3.add_argument("--alpha-value", default="0.2")

    exp7 = subparsers.add_parser("exp7-ablation", help="Generate and optionally run PC-SCFM ablation configs.")
    exp7.add_argument("--config", default="configs/ecg_baseline_wander_pc_scfm.yaml")
    exp7.add_argument("--output-root", required=True)
    exp7.add_argument("--run-train", action="store_true")
    exp7.add_argument("--train-iterations", type=int, default=None)
    exp7.add_argument("--eval-every", type=int, default=None)
    exp7.add_argument("--summarize-only", action="store_true")
    exp7.add_argument("--include-best-loss", action="store_true")
    return parser.parse_args()


def add_generation_args(parser):
    parser.add_argument("--config", default="configs/ecg_baseline_wander_mecg_e.yaml")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--noise-dir", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-fs", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)


def add_inference_args(parser):
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--low-frequency-high-hz", type=float, default=None)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def comma_values(raw, default_values):
    if raw is None:
        return [str(value) for value in default_values]
    return [value.strip() for value in raw.split(",") if value.strip()]


def label_number(value):
    return str(value).replace(".", "p").replace("-", "m")


def run_command(cmd):
    print(" ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True, cwd=SCRIPT_DIR)


def run_preprocess(args, output_dir, baseline_kind, alpha_values, frequencies_hz=None):
    cmd = [
        sys.executable,
        SCRIPT_DIR / "preprocess_ecg.py",
        "--config",
        args.config,
        "--input-dir",
        args.input_dir,
        "--metadata-csv",
        args.metadata_csv,
        "--dataset-name",
        "ptbxl",
        "--output-dir",
        output_dir,
        "--baseline-kind",
        baseline_kind,
        "--alpha-values",
        alpha_values,
        "--splits",
        "test",
        "--seed",
        args.seed,
    ]
    if args.noise_dir:
        cmd.extend(["--noise-dir", args.noise_dir])
    if args.source_fs is not None:
        cmd.extend(["--source-fs", args.source_fs])
    if args.limit is not None:
        cmd.extend(["--limit", args.limit])
    if frequencies_hz is not None:
        cmd.extend(["--frequencies-hz", frequencies_hz])
    run_command(cmd)


def run_inference(args, input_npz, output_dir):
    cmd = [
        sys.executable,
        SCRIPT_DIR / "inference.py",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--input",
        input_npz,
        "--output-dir",
        output_dir,
        "--batch-size",
        args.batch_size,
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.low_frequency_high_hz is not None:
        cmd.extend(["--low-frequency-high-hz", args.low_frequency_high_hz])
    run_command(cmd)


def read_metrics_summary(path):
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["metric"]] = row
    return rows


def write_sweep_summary(path, sweep_rows):
    metrics = sorted({metric for row in sweep_rows for metric in row["metrics"]})
    fieldnames = ["experiment", "condition_name", "condition_value"]
    for metric in metrics:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_count"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep_rows:
            out = {
                "experiment": row["experiment"],
                "condition_name": row["condition_name"],
                "condition_value": row["condition_value"],
            }
            for metric, stats in row["metrics"].items():
                out[f"{metric}_mean"] = stats["mean"]
                out[f"{metric}_std"] = stats["std"]
                out[f"{metric}_count"] = stats["count"]
            writer.writerow(out)


def run_exp2_strength(args):
    cfg = load_yaml(SCRIPT_DIR / args.config if not Path(args.config).is_absolute() else args.config)
    default_alphas = cfg["dataset"]["baseline_wander"].get("alpha_values", [0.05, 0.1, 0.2, 0.3, 0.5])
    alphas = comma_values(args.alpha_values, default_alphas)
    root = Path(args.output_root)
    summary_rows = []

    for alpha in alphas:
        condition = f"alpha_{label_number(alpha)}"
        condition_dir = root / "exp2_strength" / condition
        processed_dir = condition_dir / "processed"
        inference_dir = condition_dir / "inference"
        run_preprocess(args, processed_dir, args.baseline_kind, alpha)
        run_inference(args, processed_dir / "test.npz", inference_dir)
        metrics = read_metrics_summary(inference_dir / "metrics_summary.csv")
        summary_rows.append(
            {
                "experiment": "Experiment 2 - Baseline Strength",
                "condition_name": "alpha",
                "condition_value": alpha,
                "metrics": metrics,
            }
        )

    write_sweep_summary(root / "exp2_strength" / "summary.csv", summary_rows)


def run_exp3_frequency(args):
    cfg = load_yaml(SCRIPT_DIR / args.config if not Path(args.config).is_absolute() else args.config)
    default_freqs = cfg["dataset"]["baseline_wander"].get(
        "controlled_frequencies_hz", [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    )
    freqs = comma_values(args.frequencies_hz, default_freqs)
    root = Path(args.output_root)
    summary_rows = []

    for freq in freqs:
        condition = f"freq_{label_number(freq)}hz"
        condition_dir = root / "exp3_frequency" / condition
        processed_dir = condition_dir / "processed"
        inference_dir = condition_dir / "inference"
        run_preprocess(args, processed_dir, args.baseline_kind, args.alpha_value, frequencies_hz=freq)
        run_inference(args, processed_dir / "test.npz", inference_dir)
        metrics = read_metrics_summary(inference_dir / "metrics_summary.csv")
        summary_rows.append(
            {
                "experiment": "Experiment 3 - Baseline Frequency",
                "condition_name": "frequency_hz",
                "condition_value": freq,
                "metrics": metrics,
            }
        )

    write_sweep_summary(root / "exp3_frequency" / "summary.csv", summary_rows)


def ablation_variants(base_cfg, output_root):
    variants = []

    def add(name, description, updates):
        cfg = clone_yaml(base_cfg)
        cfg["exp_name"] = f"{base_cfg['exp_name']}_{name}"
        cfg["root_dir"] = str(Path(output_root) / "exp7_ablation" / "runs")
        cfg["checkpoint_dir"] = "${root_dir}/checkpoint"
        cfg["results_dir"] = "${root_dir}/results"
        cfg["log_dir"] = "${root_dir}/log"
        for path, value in updates.items():
            set_nested(cfg, path, value)
        variants.append((name, description, cfg))

    add("full", "Full PC-SCFM-MambAttention-ECG.", {})
    add(
        "one_shot",
        "MambAttention one-shot baseline without PC-SCFM controller.",
        {
            "model_name": "mambattention_ecg",
            "model.pcscfm_enabled": False,
            "model.loss_fn": "time+com+con",
        },
    )
    add(
        "fixed_multistep",
        "Fixed supervised multi-step baseline without flow/policy losses.",
        {
            "model.policy_mode": "fixed_multistep",
            "model.t_max": 4,
            "model.flow_nfe": 1,
            "model.flow_samples": 1,
            "model.stop_threshold": 1.1,
            "model.reject_threshold": 1.1,
            "model.loss_fn": "time+com+con+lf+morph",
        },
    )
    add(
        "no_flow",
        "Remove flow matching loss and use a deterministic one-sample flow branch.",
        {
            "model.flow_nfe": 1,
            "model.flow_samples": 1,
            "model.loss_fn": "time+com+con+lf+morph+bc+value+risk",
        },
    )
    add(
        "no_reject",
        "Disable reject decisions and reject regularization.",
        {
            "model.reject_threshold": 1.1,
            "model.reject_uncertainty_threshold": 1.0e9,
            "model.reject_disagreement_threshold": 1.0e9,
            "model.lambda_reject": 0.0,
        },
    )
    add(
        "no_safety",
        "Relax morphology safety projection budget.",
        {
            "model.safety_kappa": 1.0e9,
        },
    )
    add(
        "phase_sincos",
        "Use sin/cos phase representation instead of raw wrapped phase.",
        {
            "model.phase_representation": "sincos",
        },
    )

    return variants


def clone_yaml(value):
    return yaml.safe_load(yaml.safe_dump(value))


def set_nested(cfg, dotted_path, value):
    current = cfg
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def run_exp7_ablation(args):
    root = Path(args.output_root)
    base_cfg = load_yaml(SCRIPT_DIR / args.config if not Path(args.config).is_absolute() else args.config)
    if base_cfg.get("model_name") != "pc_scfm" or not base_cfg.get("model", {}).get("pcscfm_enabled", False):
        raise ValueError(
            "exp7-ablation is PC-SCFM-specific. Use configs/ecg_baseline_wander_pc_scfm.yaml "
            "or another config with model_name=pc_scfm and model.pcscfm_enabled=true."
        )
    variants = ablation_variants(base_cfg, root)
    config_dir = root / "exp7_ablation" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    for name, description, cfg in variants:
        if args.train_iterations is not None:
            cfg["training"]["train_iterations"] = args.train_iterations
        if args.eval_every is not None:
            cfg["training"]["eval_every"] = args.eval_every
        config_path = config_dir / f"{name}.yaml"
        save_yaml(config_path, cfg)
        plan_rows.append({"ablation": name, "description": description, "config": str(config_path)})

    write_plan_csv(root / "exp7_ablation" / "ablation_plan.csv", plan_rows)

    if args.run_train and not args.summarize_only:
        for row in plan_rows:
            run_command([sys.executable, SCRIPT_DIR / "train_supervised.py", "--config", row["config"]])

    summarize_ablation(root, plan_rows, include_best_loss=args.include_best_loss)


def write_plan_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "description", "config"])
        writer.writeheader()
        writer.writerows(rows)


def summarize_ablation(root, plan_rows, include_best_loss=False):
    summary_path = root / "exp7_ablation" / "summary.csv"
    rows = []
    checkpoint_names = ["best_pcc", "best_loss"] if include_best_loss else ["best_pcc"]
    for plan in plan_rows:
        cfg = load_yaml(plan["config"])
        result_base = Path(cfg["results_dir"].replace("${root_dir}", cfg["root_dir"])) / cfg["exp_name"]
        model_dir = result_base / cfg["model_name"]
        for checkpoint_name in checkpoint_names:
            metrics_path = model_dir / checkpoint_name / "metrics_ptbxl_fold10_test.yaml"
            complexity_path = model_dir / checkpoint_name / "complexity_summary.yaml"
            if not metrics_path.exists():
                continue
            metrics = load_yaml(metrics_path)
            complexity = load_yaml(complexity_path) if complexity_path.exists() else {}
            row = {
                "ablation": plan["ablation"],
                "checkpoint": checkpoint_name,
                "config": plan["config"],
            }
            for key, value in {**metrics, **complexity}.items():
                row[key] = value
            rows.append(row)

    if not rows:
        print(f"No ablation metrics found yet under {root / 'exp7_ablation' / 'runs'}.")
        return

    fieldnames = sorted({key for row in rows for key in row})
    preferred = ["ablation", "checkpoint", "config"]
    fieldnames = preferred + [key for key in fieldnames if key not in preferred]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved ablation summary -> {summary_path}")


def main():
    args = parse_args()
    if args.command == "exp2-strength":
        run_exp2_strength(args)
    elif args.command == "exp3-frequency":
        run_exp3_frequency(args)
    elif args.command == "exp7-ablation":
        run_exp7_ablation(args)


if __name__ == "__main__":
    main()
