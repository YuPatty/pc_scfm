# Data Layout

Keep source ECG records, processed NPZ files, generated experiment outputs, and reference implementations separate.

This project's maintained training environment is `src/`. The original PC-SCFM workspace is preserved under `references/pc_scfm_original/` for reference only.

## Project Workspace

```text
rl_exp/
  src/
    configs/
      ecg_baseline_wander_mecg_e.yaml
      ecg_baseline_wander_mambattention.yaml
      ecg_baseline_wander_pc_scfm.yaml
    datasets/
    models/
    train_supervised.py
    inference.py
    preprocess_ecg.py
    experiment_suite.py
    result_analysis.py

  docs/
  notes/
  references/
    pc_scfm_original/
  scripts/
  runs/                  # generated, not tracked
```

## Data Root

Default data root used by the configs:

```text
/mnt/c/Users/中研院/data/ecg_baseline_wander
```

Windows path:

```text
C:\Users\中研院\data\ecg_baseline_wander
```

Recommended data layout:

```text
ecg_baseline_wander/
  raw/
    PTBXL/
      ptbxl_database.csv
      records100/
      records500/
    NSTDB/
    MITBIH/
    Chapman/
    CPSC/
    other/

  processed/
    train.npz
    train_val.npz
    val.npz
    test.npz
    mit_bih.npz
    chapman.npz
    cpsc.npz

  controlled_tests/
    exp2_strength/
    exp3_frequency/
```

The folders above are already created at:

```text
/mnt/c/Users/中研院/data/ecg_baseline_wander
```

Use `raw/` for downloaded or extracted source datasets. Do not train directly from `raw/`.

Use `processed/` for model-ready NPZ files. The dataset loader reads files from `processed/` by split name.

Use `controlled_tests/` for generated robustness test sets, such as baseline strength and frequency sweeps.

## Required Processed Files

Minimum files for training and in-domain testing:

```text
processed/train.npz
processed/val.npz
processed/test.npz
```

Recommended files:

```text
processed/train.npz
processed/train_val.npz
processed/val.npz
processed/test.npz
processed/mit_bih.npz
processed/chapman.npz
processed/cpsc.npz
```

If `train_val.npz` is missing, the loader falls back to `val.npz` for train-validation evaluation.

## NPZ Schema

Each processed NPZ must contain either:

```text
noisy_ecg
clean_reference
```

or the alternate names:

```text
input
target
```

Expected array shape:

```text
[N, T]
[N, 1, T]
```

Do not store 12-lead arrays as `[N, 12, T]` for the current models. This experiment environment is configured for single-lead ECG baseline-wander removal.

For PTB-XL official fold validation, processed NPZ files may include one of these metadata keys:

```text
ptbxl_fold
strat_fold
fold
```

Expected folds:

```text
train.npz: folds 1-8
val.npz: fold 9
test.npz: fold 10
```

## Config Data Path

All maintained configs use this top-level setting:

```yaml
data_dir: "/mnt/c/Users/中研院/data/ecg_baseline_wander"
```

Dataset loading resolves processed files as:

```text
${data_dir}/processed/train.npz
${data_dir}/processed/val.npz
${data_dir}/processed/test.npz
```

To use a different data location, update `data_dir` in the config, or pass an OmegaConf override:

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 train_supervised.py \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  data_dir=/path/to/ecg_baseline_wander
```

## Output Layout

Default output root:

```text
/mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander
```

Generated outputs:

```text
runs/ecg_baseline_wander/
  checkpoint/
    <exp_name>/<model_name>/
      best_pcc_model.pt
      best_model.pt
      model_last.pt
      training_state.pt

  results/
    <exp_name>/<model_name>/
      best_pcc/
        metrics_ptbxl_fold10_test.yaml
        metrics_mit_bih.yaml
        metrics_chapman.yaml
        metrics_cpsc.yaml
        complexity_summary.yaml
      best_loss/

  log/
    <exp_name>/<model_name>/log.log

  exp7_ablation/
    ablation_plan.csv
    configs/
    summary.csv
```

Generated outputs should not be committed.

## Common Commands

Check whether required data exists:

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/check_project.sh
```

Train PC-SCFM:

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/train_model.sh pc_scfm
```

Train comparison models:

```bash
bash scripts/train_model.sh mecg_e
bash scripts/train_model.sh mambattention
```

Generate PC-SCFM ablation configs:

```bash
bash scripts/run_pcscfm_ablation.sh --summarize-only
```

Train all PC-SCFM ablations:

```bash
bash scripts/run_pcscfm_ablation.sh --run-train
```

## Notes

`references/pc_scfm_original/` may still contain older `pkl`-based scripts and folders such as `data/`, `raw_data/`, and `external_data/`. Those belong to the original reference workspace.

For the current `rl_exp/src` experiments, use the NPZ layout described above.
