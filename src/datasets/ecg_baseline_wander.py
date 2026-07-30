from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .factory import register_dataset
except ImportError:
    from factory import register_dataset


def _as_single_lead_tensor(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3 or array.shape[1] != 1:
        raise ValueError(f"Expected ECG array shaped [N, T] or [N, 1, T], got {array.shape}.")
    return torch.from_numpy(array)


@register_dataset("ecg_baseline_wander")
class ECGBaselineWanderDataset(Dataset):
    """
    Dataset adapter for the ECG baseline-wander-removal benchmark.

    It expects preprocessed NPZ files under `processed_data_dir` or `data_dir/processed`:
      train.npz, val.npz, test.npz

    External test datasets are optional and use the same NPZ schema:
      mit_bih.npz, chapman.npz, cpsc.npz

    Each NPZ must contain:
      noisy_ecg or input: model input with synthetic/real baseline wander
      clean_reference or target: ECG clean reference

    Arrays may be [N, T] or [N, 1, T]. The loader returns (noisy_ecg, clean_reference).
    """

    split_file_map = {
        "train": "train.npz",
        "val": "val.npz",
        "test": "test.npz",
        "mit_bih": "mit_bih.npz",
        "mit-bih": "mit_bih.npz",
        "chapman": "chapman.npz",
        "cpsc": "cpsc.npz",
    }

    def __init__(self, data_mode="train", **kwargs):
        if data_mode not in self.split_file_map:
            raise ValueError(
                f"Unsupported data_mode={data_mode!r}. Expected one of {sorted(self.split_file_map)}."
            )

        data_dir = Path(kwargs["data_dir"])
        processed_dir = Path(kwargs.get("processed_data_dir", data_dir / "processed"))
        split_path = processed_dir / self.split_file_map[data_mode]

        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing preprocessed ECG split: {split_path}. Create NPZ files with "
                "`noisy_ecg`/`clean_reference` arrays before training."
            )

        loaded = np.load(split_path)
        noisy_key = "noisy_ecg" if "noisy_ecg" in loaded else "input"
        clean_key = "clean_reference" if "clean_reference" in loaded else "target"
        if noisy_key not in loaded or clean_key not in loaded:
            raise KeyError(
                f"{split_path} must contain noisy_ecg/input and clean_reference/target arrays."
            )

        self.noisy = _as_single_lead_tensor(loaded[noisy_key])
        self.clean = _as_single_lead_tensor(loaded[clean_key])
        if len(self.noisy) != len(self.clean):
            raise ValueError(
                f"noisy and clean arrays must have the same length, got {len(self.noisy)} and {len(self.clean)}."
            )

        self._validate_ptbxl_fold_metadata(loaded, data_mode, kwargs.get("split"))

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], self.clean[index]

    def _validate_ptbxl_fold_metadata(self, loaded, data_mode, split_config):
        if data_mode not in {"train", "val", "test"}:
            return
        if not isinstance(split_config, dict) or split_config.get("strategy") != "ptbxl_official_folds":
            return

        fold_key = None
        for candidate in ("ptbxl_fold", "strat_fold", "fold"):
            if candidate in loaded:
                fold_key = candidate
                break
        if fold_key is None:
            raise ValueError(
                f"{data_mode}.npz is used with ptbxl_official_folds but does not contain "
                "ptbxl_fold/strat_fold/fold metadata."
            )

        folds = np.asarray(loaded[fold_key], dtype=np.int64)
        if folds.shape[0] != len(self.noisy):
            raise ValueError(
                f"{fold_key} metadata length {folds.shape[0]} does not match ECG length {len(self.noisy)}."
            )

        if data_mode == "train":
            expected = set(split_config.get("train_folds", []))
        elif data_mode == "val":
            expected = {int(split_config.get("validation_fold", 9))}
        else:
            expected = {int(split_config.get("test_fold", 10))}

        observed = set(np.unique(folds).tolist())
        if observed - expected:
            raise ValueError(
                f"{data_mode}.npz contains PTB-XL folds {sorted(observed)}, expected only {sorted(expected)}."
            )
