import logging
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import spectrogram, welch

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


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


def setup_logger(args, log_dir) -> logging.Logger:
    for noisy in ["matplotlib", "PIL", "torch"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(args.exp_name)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(asctime)s: %(message)s")
    fh = logging.FileHandler(Path(log_dir) / "log.log", mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def plot_loss_curves(train_losses, val_losses, eval_every, results_dir, val_pccs=None):
    loss_curves_dir = Path(results_dir) / "loss_curves"
    loss_curves_dir.mkdir(parents=True, exist_ok=True)

    def save_curve(values, steps, label, filename, ylabel):
        if not values:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, values, label=label, linewidth=2)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(loss_curves_dir / filename, dpi=200)
        plt.close(fig)

    save_curve(train_losses, np.arange(1, len(train_losses) + 1), "Train Loss", "train_loss.png", "loss")
    val_steps = np.arange(1, len(val_losses) + 1) * eval_every
    save_curve(val_losses, val_steps, "Val Loss", "val_loss.png", "loss")
    if val_pccs:
        val_pcc_steps = np.arange(1, len(val_pccs) + 1) * eval_every
        save_curve(val_pccs, val_pcc_steps, "Val PCC", "val_pcc.png", "PCC")
    return loss_curves_dir


def reconstruction_metrics_from_arrays(noisy, clean, pred, fs=250, low_freq_hz=0.5, eps=1e-10):
    noisy = np.squeeze(np.asarray(noisy), axis=1)
    clean = np.squeeze(np.asarray(clean), axis=1)
    pred = np.squeeze(np.asarray(pred), axis=1)
    return {
        "SSD": float(np.mean(ssd(clean, pred))),
        "MAD": float(np.mean(maximum_absolute_distance(clean, pred))),
        "PRD": float(np.mean(prd(clean, pred, eps=eps))),
        "CosSim": float(np.mean(cosine_similarity(clean, pred, eps=eps))),
        "Output_SNR_dB": float(np.mean(snr_db(clean, pred, eps=eps))),
        "SNR_Improvement_dB": float(np.mean(snr_improvement_db(clean, noisy, pred, eps=eps))),
        "LF_Reduction_dB": float(
            np.mean(low_frequency_power_reduction(noisy, pred, fs=fs, high_hz=low_freq_hz, eps=eps))
        ),
        "R_Peak_Timing_Error_ms": nanmean_or_nan(r_peak_timing_error_ms(clean, pred, fs=fs)),
        "RR_Interval_MAE_ms": nanmean_or_nan(rr_interval_mae_ms(clean, pred, fs=fs)),
        "RMSE": float(np.sqrt(np.mean((clean - pred) ** 2))),
        "Centered_CosSim": float(np.mean(centered_cosine_similarity(clean, pred, eps=eps))),
        "QRS_Amplitude_Error": nanmean_or_nan(qrs_amplitude_error(clean, pred, fs=fs)),
    }


def plot_prediction_results(model, dataset, device, results_dir, num_samples=3, seed=42, eps=1e-10, fs=250):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        raise ValueError("Cannot plot predictions from an empty dataset.")

    rng = random.Random(seed)
    sample_indices = rng.sample(range(len(dataset)), min(num_samples, len(dataset)))
    output_paths = []

    model.eval()
    with torch.no_grad():
        for plot_idx, sample_idx in enumerate(sample_indices):
            noisy, clean = dataset[sample_idx]
            pred = model(noisy.unsqueeze(0).to(device)).squeeze(0).detach().cpu().numpy()
            noisy_np = noisy.detach().cpu().numpy()
            clean_np = clean.detach().cpu().numpy()

            noisy_1d = np.squeeze(noisy_np)
            clean_1d = np.squeeze(clean_np)
            pred_1d = np.squeeze(pred)
            time_axis = np.arange(clean_1d.shape[-1])

            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            axes[0].plot(time_axis, noisy_1d, color="tab:gray", linewidth=1.0)
            axes[0].set_ylabel("Input")
            axes[1].plot(time_axis, clean_1d, color="tab:blue", linewidth=1.0)
            axes[1].plot(time_axis, pred_1d, color="tab:red", linewidth=1.0, alpha=0.85)
            axes[1].set_ylabel("ECG")
            axes[2].plot(time_axis, clean_1d - pred_1d, color="tab:purple", linewidth=1.0)
            axes[2].set_ylabel("Residual")
            axes[2].set_xlabel("Sample")
            for ax in axes:
                ax.grid(True, alpha=0.2)
            axes[0].set_title(f"Baseline Wander Removal - Sample {sample_idx}")
            fig.tight_layout()

            output_path = results_dir / f"prediction_sample_{plot_idx + 1}.png"
            fig.savefig(output_path, dpi=200)
            plt.close(fig)
            output_paths.append(output_path)

            spectral_path = results_dir / f"spectrum_sample_{plot_idx + 1}.png"
            _plot_spectral_diagnostics(
                noisy_1d,
                clean_1d,
                pred_1d,
                fs=fs,
                output_path=spectral_path,
            )
            output_paths.append(spectral_path)
    return output_paths


def _plot_spectral_diagnostics(noisy, clean, restored, fs, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for signal, label, color in [
        (noisy, "Input", "tab:gray"),
        (clean, "Clean reference", "tab:blue"),
        (restored, "Restored", "tab:red"),
    ]:
        nperseg = min(len(signal), max(64, int(fs * 2)))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        axes[0, 0].semilogy(freqs, psd + 1e-12, label=label, color=color, linewidth=1.2)

    axes[0, 0].set_title("PSD")
    axes[0, 0].set_xlabel("Hz")
    axes[0, 0].set_ylabel("Power")
    axes[0, 0].set_xlim(0, min(40, fs / 2))
    axes[0, 0].grid(True, alpha=0.2)
    axes[0, 0].legend(loc="best")

    residual = clean - restored
    axes[0, 1].plot(np.arange(len(residual)) / fs, residual, color="tab:purple", linewidth=1.0)
    axes[0, 1].set_title("Residual error")
    axes[0, 1].set_xlabel("Second")
    axes[0, 1].grid(True, alpha=0.2)

    for ax, signal, title in [
        (axes[1, 0], noisy, "Input spectrogram"),
        (axes[1, 1], restored, "Restored spectrogram"),
    ]:
        nperseg = min(len(signal), max(64, int(fs * 2)))
        noverlap = min(nperseg - 1, nperseg // 2)
        freqs, times, spec = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        mask = freqs <= min(40, fs / 2)
        mesh = ax.pcolormesh(
            times,
            freqs[mask],
            10 * np.log10(spec[mask] + 1e-12),
            shading="auto",
            cmap="magma",
        )
        ax.set_title(title)
        ax.set_xlabel("Second")
        ax.set_ylabel("Hz")
        fig.colorbar(mesh, ax=ax, label="dB")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def get_reconstruction_metrics(model, test_loader, device, fs=250, low_freq_hz=0.5, eps=1e-10):
    noisy_all, clean_all, pred_all = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            noisy, clean = batch[0].to(device), batch[1].to(device)
            pred = model(noisy)
            noisy_all.append(noisy.detach().cpu().numpy())
            clean_all.append(clean.detach().cpu().numpy())
            pred_all.append(pred.detach().cpu().numpy())

    return reconstruction_metrics_from_arrays(
        np.concatenate(noisy_all, axis=0),
        np.concatenate(clean_all, axis=0),
        np.concatenate(pred_all, axis=0),
        fs=fs,
        low_freq_hz=low_freq_hz,
        eps=eps,
    )


def profile_model_complexity(model, device, input_length, batch_size=1, warmup=5, repeats=20):
    model.eval()
    dummy = torch.zeros(batch_size, 1, input_length, device=device)
    params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    flops = float("nan")
    try:
        from thop import profile

        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = float(flops)
    except Exception:
        pass

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else float("nan")
    )
    return {
        "Parameters": float(params),
        "Trainable_Parameters": float(trainable_params),
        "FLOPs": flops,
        "Inference_Time_ms": float(elapsed / repeats * 1000.0),
        "Peak_Memory_MB": float(peak_memory_mb),
    }
