import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, coherence
from tslearn.metrics import dtw


def apply_bandpass_filter(data, fs, lowcut, highcut):
    """Applies a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(4, [low, high], btype='band', output='sos') # second-order sections
    # Filter along the last axis (timesteps)
    return sosfiltfilt(sos, data, axis=-1)


def _batched_pearsonr(x, y, eps=1e-10):
    """
    Vectorized Pearson correlation along the last axis.
    x, y: (..., Timesteps). Returns (correlation matrix, valid mask) with
    shape (...), where valid marks entries where neither signal is flat.
    """
    x_c = x - x.mean(axis=-1, keepdims=True)
    y_c = y - y.mean(axis=-1, keepdims=True)
    num = np.sum(x_c * y_c, axis=-1)
    denom = np.linalg.norm(x_c, axis=-1) * np.linalg.norm(y_c, axis=-1) + eps
    corr = num / denom
    valid = (x.std(axis=-1) > eps) & (y.std(axis=-1) > eps)
    return corr, valid


def calculate_base_metrics(target, pred, fs=200, eps=1e-10, nperseg=None, nperseg_coh=None):
    """
    Calculates metrics across a batch.
    Expected shape for target and pred: (Batch, Channels, Timesteps)

    nperseg/nperseg_coh: Welch/coherence window lengths. Default to None, which falls back to
    the old behavior of using the full Timesteps length (chunk_len) -- appropriate when the
    caller already concatenated many trials into one long series. Callers with only a handful of
    short trials (e.g. one 50-sample masked patch) should pass explicit, larger values so
    Welch/coherence average over multiple, better-resolved windows instead of degenerating to a
    single-segment periodogram.
    """
    metrics = {}
    metrics['SSD'] = np.mean(np.sum((target - pred) ** 2, axis=-1))
    metrics['MAD'] = np.mean(np.max(np.abs(target - pred), axis=-1))
    prd_num = np.sum((target - pred) ** 2, axis=-1)
    prd_den = np.sum((target - target.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    metrics['PRD'] = np.mean(np.sqrt(prd_num / (prd_den + eps)) * 100.0)

    num_cos = np.sum(target * pred, axis=-1)
    denom_cos = np.linalg.norm(target, axis=-1) * np.linalg.norm(pred, axis=-1) + eps
    cos_matrix = num_cos / denom_cos
    metrics['CosSim'] = float(np.mean(cos_matrix))

    metrics['MSE'] = np.mean((target - pred)**2)

    # PCC calculated per channel, then average (vectorized over batch/channel)
    pcc_matrix, pcc_valid = _batched_pearsonr(target, pred, eps=eps)
    metrics['PCC'] = float(pcc_matrix[pcc_valid].mean()) if pcc_valid.any() else 0.0

    # Envelope Correlation (via Hilbert). Keep the analytic signal around so
    # IPC below can reuse it instead of recomputing the Hilbert transform.
    analytic_target = hilbert(target, axis=-1)
    analytic_pred = hilbert(pred, axis=-1)
    env_target = np.abs(analytic_target)
    env_pred = np.abs(analytic_pred)
    env_pcc_matrix, env_valid = _batched_pearsonr(env_target, env_pred, eps=eps)
    metrics['Env_Corr'] = float(env_pcc_matrix[env_valid].mean()) if env_valid.any() else 0.0

    # Frequency-Domain Error (Log Spectral Distance)
    # ensures the model generates the correct oscillatory bands
    # MAE (mean absolute error) in PSD 
    chunk_len = target.shape[-1]
    welch_nperseg = min(nperseg or chunk_len, chunk_len)
    f, Pxx_target = welch(target, fs=fs, nperseg=welch_nperseg, axis=-1)
    f, Pxx_pred = welch(pred, fs=fs, nperseg=welch_nperseg, axis=-1)
    # Log scale difference in dB (10 * log10)
    metrics['PSD_Error'] = np.mean(np.abs(10 * np.log10(Pxx_target + eps) - 10 * np.log10(Pxx_pred + eps)))

    # Magnitude-Squared Coherence
    # Measuring phase stability across frequencies (frequency-domain version of correlation)
    # need multiple segments for coherence to be meaningful
    coh_nperseg = min(nperseg_coh or (chunk_len // 4), fs, chunk_len)
    f_coh, Cxy = coherence(target, pred, fs=fs, nperseg=coh_nperseg, axis=-1)
    metrics['Coherence'] = np.mean(Cxy)

    # Signal-to-Noise Ratio (SNR)
    var_target = np.var(target, axis=-1)
    var_error = np.var(target - pred, axis=-1)
    snr = 10 * np.log10(var_target / (var_error + eps))
    metrics['SNR_dB'] = np.mean(snr)

    # Instantaneous Phase Cosine (IPC) — penalizes temporal phase offsets.
    # Reuses the analytic signal computed above instead of re-running Hilbert.
    phi_t = np.angle(analytic_target)
    phi_p = np.angle(analytic_pred)
    ipc_matrix = np.mean(np.cos(phi_t - phi_p), axis=-1)
    metrics['IPC'] = float(np.mean(ipc_matrix))

    # # Dynamic Time Warping (DTW)
    # # give credit for slight temporal misalignments
    # # Note: DTW is computationally expensive. If this is too slow on large batches,
    # # consider taking a random subset of channels/batches for this specific metric.
    # dtw_dists = []
    # for b in range(target.shape[0]):
    #     for c in range(target.shape[1]):
    #         dist = dtw(target[b, c], pred[b, c])
    #         dtw_dists.append(dist)
    # metrics['DTW'] = np.mean(dtw_dists)

    return metrics
