import numpy as np


def ssd(clean, restored):
    clean = np.asarray(clean)
    restored = np.asarray(restored)
    return np.sum((restored - clean) ** 2, axis=-1)


def maximum_absolute_distance(clean, restored):
    clean = np.asarray(clean)
    restored = np.asarray(restored)
    return np.max(np.abs(restored - clean), axis=-1)


def prd(clean, restored, eps=1e-10):
    clean = np.asarray(clean)
    restored = np.asarray(restored)
    numerator = np.sum((restored - clean) ** 2, axis=-1)
    denominator = np.sum((clean - clean.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
    return np.sqrt(numerator / (denominator + eps)) * 100.0


def snr_db(clean, restored, eps=1e-10):
    clean = np.asarray(clean)
    restored = np.asarray(restored)
    signal = np.sum(clean**2, axis=-1)
    noise = np.sum((clean - restored) ** 2, axis=-1)
    return 10.0 * np.log10((signal + eps) / (noise + eps))


def snr_improvement_db(clean, noisy, restored, eps=1e-10):
    return snr_db(clean, restored, eps=eps) - snr_db(clean, noisy, eps=eps)


def centered_cosine_similarity(clean, restored, eps=1e-10):
    clean = np.asarray(clean) - np.asarray(clean).mean(axis=-1, keepdims=True)
    restored = np.asarray(restored) - np.asarray(restored).mean(axis=-1, keepdims=True)
    numerator = np.sum(clean * restored, axis=-1)
    denominator = np.linalg.norm(clean, axis=-1) * np.linalg.norm(restored, axis=-1)
    return numerator / (denominator + eps)


def cosine_similarity(clean, restored, eps=1e-10):
    clean = np.asarray(clean)
    restored = np.asarray(restored)
    numerator = np.sum(clean * restored, axis=-1)
    denominator = np.linalg.norm(clean, axis=-1) * np.linalg.norm(restored, axis=-1)
    return numerator / (denominator + eps)


def low_frequency_power(signal, fs, high_hz=0.5, eps=1e-10):
    signal = np.asarray(signal)
    freqs = np.fft.rfftfreq(signal.shape[-1], d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    mask = freqs <= high_hz
    return spectrum[..., mask].sum(axis=-1) + eps


def low_frequency_power_reduction(input_ecg, restored, fs, high_hz=0.5, eps=1e-10):
    before = low_frequency_power(input_ecg, fs=fs, high_hz=high_hz, eps=eps)
    after = low_frequency_power(restored, fs=fs, high_hz=high_hz, eps=eps)
    return 10.0 * np.log10(before / after)


def detect_r_peaks(signal, fs, refractory_ms=250, threshold_scale=0.5):
    """
    Fixed lightweight R-peak detector for metric consistency.

    It emphasizes steep QRS slopes with a first derivative, smooths the energy with
    an 80 ms moving average, then applies a global threshold and refractory window.
    This is intentionally simple and deterministic; all models are evaluated with
    the same detector.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < 3:
        return np.array([], dtype=np.int64)

    centered = signal - np.median(signal)
    slope_energy = np.diff(centered, prepend=centered[0]) ** 2
    smooth_window = max(1, int(round(0.08 * fs)))
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    score = np.convolve(slope_energy, kernel, mode="same")
    threshold = np.median(score) + threshold_scale * np.std(score)
    candidate_indices = np.flatnonzero(score > threshold)
    if candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    refractory = max(1, int(round(refractory_ms * fs / 1000.0)))
    peaks = []
    start = int(candidate_indices[0])
    prev = start
    for idx in candidate_indices[1:]:
        idx = int(idx)
        if idx - prev > refractory:
            peaks.append(_strongest_peak(centered, score, start, prev, fs))
            start = idx
        prev = idx
    peaks.append(_strongest_peak(centered, score, start, prev, fs))
    return np.asarray(peaks, dtype=np.int64)


def _strongest_peak(signal, score, start, end, fs):
    margin = max(1, int(round(0.05 * fs)))
    lo = max(0, start - margin)
    hi = min(signal.size, end + margin + 1)
    local = np.arange(lo, hi)
    if local.size == 0:
        return int(start)
    # Prefer the ECG amplitude extremum inside the high-slope QRS candidate.
    return int(local[np.argmax(np.abs(signal[local]))])


def _match_peaks(reference_peaks, comparison_peaks, fs, tolerance_ms=150):
    reference_peaks = np.asarray(reference_peaks, dtype=np.int64)
    comparison_peaks = np.asarray(comparison_peaks, dtype=np.int64)
    if reference_peaks.size == 0 or comparison_peaks.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    tolerance = int(round(tolerance_ms * fs / 1000.0))
    pairs = []
    used = set()
    for ref in reference_peaks:
        distances = np.abs(comparison_peaks - ref)
        order = np.argsort(distances)
        for idx in order:
            if int(idx) in used:
                continue
            if distances[idx] <= tolerance:
                pairs.append((int(ref), int(comparison_peaks[idx])))
                used.add(int(idx))
            break
    return np.asarray(pairs, dtype=np.int64)


def r_peak_timing_error_ms(clean, restored, fs, tolerance_ms=150):
    values = []
    for clean_signal, restored_signal in zip(np.asarray(clean), np.asarray(restored)):
        clean_peaks = detect_r_peaks(clean_signal, fs)
        restored_peaks = detect_r_peaks(restored_signal, fs)
        pairs = _match_peaks(clean_peaks, restored_peaks, fs, tolerance_ms=tolerance_ms)
        if pairs.size:
            values.append(np.mean(np.abs(pairs[:, 1] - pairs[:, 0])) / fs * 1000.0)
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def rr_interval_mae_ms(clean, restored, fs, tolerance_ms=150):
    values = []
    for clean_signal, restored_signal in zip(np.asarray(clean), np.asarray(restored)):
        clean_peaks = detect_r_peaks(clean_signal, fs)
        restored_peaks = detect_r_peaks(restored_signal, fs)
        pairs = _match_peaks(clean_peaks, restored_peaks, fs, tolerance_ms=tolerance_ms)
        if pairs.shape[0] >= 2:
            clean_rr = np.diff(pairs[:, 0]) / fs * 1000.0
            restored_rr = np.diff(pairs[:, 1]) / fs * 1000.0
            values.append(np.mean(np.abs(clean_rr - restored_rr)))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def qrs_amplitude_error(clean, restored, fs, tolerance_ms=150):
    values = []
    for clean_signal, restored_signal in zip(np.asarray(clean), np.asarray(restored)):
        clean_peaks = detect_r_peaks(clean_signal, fs)
        restored_peaks = detect_r_peaks(restored_signal, fs)
        pairs = _match_peaks(clean_peaks, restored_peaks, fs, tolerance_ms=tolerance_ms)
        if pairs.size:
            clean_amp = np.asarray(clean_signal)[pairs[:, 0]]
            restored_amp = np.asarray(restored_signal)[pairs[:, 1]]
            values.append(np.mean(np.abs(clean_amp - restored_amp)))
        else:
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")
