# PC-SCFM RL-No-Flow 實驗流程

這份文件只針對 `pc_scfm_rl_no_flow`，也就是：

```text
MambAttention backbone + policy/controller
without flow matching
without flow proposal
```

不包含消融實驗流程，也不包含 real-world no-reference data 實驗。

## 1. 模型設定

使用 config：

```text
src/configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml
```

此版本保留：

```text
MambAttention
multi-step restoration
policy/controller
classical baseline proposal
reject mechanism
morphology safety
```

此版本移除：

```text
FlowBaselineHead
flow matching loss
flow_mu proposal
flow uncertainty reject
policy action space 中的 flow branch
```

主要設定：

```yaml
model_name: pc_scfm
model:
  pcscfm_enabled: true
  use_flow_proposal: false
  loss_fn: "time+com+con+lf+morph+bc+value+risk"
  lambda_flow: 0.0
```

## 2. 檢查專案與資料

從專案根目錄執行：

```bash
cd /mnt/c/users/中研院/rl_exp
bash scripts/check_project.sh
```

預期主要資料位置：

```text
data/ecg_baseline_wander/raw/PTBXL/records100
data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv
data/ecg_baseline_wander/raw/NSTDB
data/ecg_baseline_wander/processed/train.npz
data/ecg_baseline_wander/processed/val.npz
data/ecg_baseline_wander/processed/test.npz
```

外部測試資料若已建立，會在：

```text
data/ecg_baseline_wander/processed/mit_bih.npz
data/ecg_baseline_wander/processed/chapman.npz
data/ecg_baseline_wander/processed/cpsc.npz
```

## 3. 建立 PTB-XL Train/Val/Test

如果還沒有 `processed/train.npz`、`processed/val.npz`、`processed/test.npz`，先跑：

```bash
cd /mnt/c/users/中研院/rl_exp
CONFIG=configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  bash scripts/preprocess_ptbxl_records100.sh
```

這會使用：

```text
PTB-XL Lead II
official folds 1-8 作 train
official fold 9 作 validation
official fold 10 作 in-domain test
NSTDB baseline，若 NSTDB 不存在則改用 random low-frequency drift
```

## 4. 建立 External Test NPZ

MIT-BIH：

```bash
cd /mnt/c/users/中研院/rl_exp/src
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/MITBIH \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name mit_bih
```

Chapman：

```bash
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/Chapman \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name chapman
```

CPSC：

```bash
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/CPSC \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name cpsc
```

如果沒有 NSTDB，可把 `--noise-dir ...` 拿掉，並加上：

```bash
--baseline-kind random_low_frequency_drift
```

## 5. Experiment 1：In-Domain Performance

訓練 `pc_scfm_rl_no_flow`：

```bash
cd /mnt/c/users/中研院/rl_exp/src
python3 train_supervised.py \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml
```

訓練結束後會自動使用 best validation PCC checkpoint 評估：

```text
PTB-XL fold 10 test
```

主要輸出：

```text
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/metrics_ptbxl_fold10_test.yaml
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/complexity_summary.yaml
```

## 6. Experiment 4-6：External Generalization

如果以下檔案存在：

```text
data/ecg_baseline_wander/processed/mit_bih.npz
data/ecg_baseline_wander/processed/chapman.npz
data/ecg_baseline_wander/processed/cpsc.npz
```

第 5 步訓練結束後會自動額外評估：

```text
MIT-BIH
Chapman
CPSC
```

輸出位置：

```text
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/metrics_mit_bih.yaml
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/metrics_chapman.yaml
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/metrics_cpsc.yaml
```

External datasets 不參與 training、validation、early stopping 或 checkpoint selection。

## 7. Experiment 2：Baseline Strength Robustness

先確認 checkpoint 存在：

```text
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt
```

只跑 `pc_scfm_rl_no_flow` 的 strength sweep：

```bash
cd /mnt/c/users/中研院/rl_exp/src
python3 experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv ../data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --checkpoint ../runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow \
  --alpha-values 0.05,0.1,0.2,0.3,0.5
```

輸出：

```text
runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow/exp2_strength/summary.csv
```

## 8. Experiment 3：Baseline Frequency Robustness

只跑 `pc_scfm_rl_no_flow` 的 frequency sweep：

```bash
cd /mnt/c/users/中研院/rl_exp/src
python3 experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv ../data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --checkpoint ../runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```

輸出：

```text
runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow/exp3_frequency/summary.csv
```

## 9. Experiment 8：Complexity Analysis

Complexity 會在訓練結束後自動輸出：

```text
runs/ecg_baseline_wander/results/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc/complexity_summary.yaml
```

包含：

```text
Parameters
Trainable_Parameters
FLOPs
Inference_Time_ms
Peak_Memory_MB
```

注意：`FLOPs` 需要安裝 `thop`；若環境沒有 `thop`，會是 `NaN`。

## 10. 結果彙整

若要彙整 inference 或 robustness sweep 產生的 `metrics_summary.csv` / `summary.csv`：

```bash
cd /mnt/c/users/中研院/rl_exp/src
python3 result_analysis.py aggregate \
  --results-root ../runs/ecg_baseline_wander \
  --output ../runs/ecg_baseline_wander/paper_tables/pc_scfm_rl_no_flow_metrics.csv
```

選 best、median、worst 案例圖：

```bash
python3 result_analysis.py select-cases \
  --inference-dir ../runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow/exp2_strength/alpha_0p2/inference \
  --output-dir ../runs/ecg_baseline_wander/case_figures/pc_scfm_rl_no_flow_alpha_0p2 \
  --metric PRD \
  --direction lower
```

## 11. 最短完整流程

已經有 PTB-XL raw data 時：

```bash
cd /mnt/c/users/中研院/rl_exp

CONFIG=configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  bash scripts/preprocess_ptbxl_records100.sh

cd src
python3 train_supervised.py \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml

python3 experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv ../data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --checkpoint ../runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow \
  --alpha-values 0.05,0.1,0.2,0.3,0.5

python3 experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_pc_scfm_rl_no_flow.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv ../data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --checkpoint ../runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm_rl_no_flow/pc_scfm/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests/pc_scfm_rl_no_flow \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```
