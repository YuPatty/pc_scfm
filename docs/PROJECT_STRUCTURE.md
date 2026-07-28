# Project Structure

## Maintained Code Path

Use `src/` for new experiments and paper-facing runs.

```text
src/train_supervised.py       Train a configured model and evaluate checkpoints
src/inference.py              Run inference on an NPZ and write restored ECG + metrics
src/experiment_suite.py       Controlled sweeps and PC-SCFM ablations
src/preprocess_ecg.py         Build processed NPZ files
src/result_analysis.py        Aggregate metrics and statistics
src/models/                   Registered model implementations
src/datasets/                 Dataset adapters
src/configs/                  Reproducible experiment configs
notes/                         Design notes and experiment menus
references/pc_scfm_original/   Original/reference PC-SCFM workspace
scripts/                       Common workflow wrappers
```

## Model Configs

```text
src/configs/ecg_baseline_wander_mecg_e.yaml
src/configs/ecg_baseline_wander_mambattention.yaml
src/configs/ecg_baseline_wander_pc_scfm.yaml
```

All three can run the standard training, inference, external testing, strength sweep, and frequency sweep. Only `pc_scfm` should run `exp7-ablation`.

## Reference Workspace

`references/pc_scfm_original/` is the original/reference PC-SCFM workspace. It is ignored by the root repository so the main experiment environment stays clean.

Do not add new paper-facing experiment logic there unless you intentionally want to preserve a reference implementation. Port maintained behavior into `src/`.

## Generated Output

Generated outputs should stay out of git:

```text
runs/
outputs/
results/
checkpoint/
logs/
*.pt
*.pth
*.pkl
*.npz
```

The normal result root is:

```text
/mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander
```

## Recommended Workflow

1. Put processed data under `/mnt/c/Users/中研院/data/ecg_baseline_wander/processed`.
2. Train `mecg_e`, `mambattention_ecg`, and `pc_scfm` from `src/configs`.
3. Run common robustness experiments for all major models.
4. Run ablation experiments only for PC-SCFM.
5. Aggregate metrics into paper tables.
