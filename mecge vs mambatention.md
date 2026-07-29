# MECG-E vs MambAttention 實驗流程

位置：

```text
C:\Users\中研院\rl_exp
/mnt/c/Users/中研院/rl_exp
```

目的：按照 `notes/experiment_實驗設計.txt`，比較 `mecg_e` 與 `mambattention_ecg` 在 Single-lead ECG Baseline Wander Removal 任務上的表現。

固定公平比較條件：

```text
Lead: Lead II
Train dataset: PTB-XL
Train split: PTB-XL official folds 1-8
Validation split: PTB-XL official fold 9
In-domain test split: PTB-XL official fold 10
External test: MIT-BIH, Chapman, CPSC
Clean reference: 經一致前處理後、尚未加入 Baseline Wander 的 ECG
Baseline Wander train source: NSTDB
Window size: 512
Sampling rate: 250 Hz
Normalization: z-score
```

比較模型：

```text
MECG-E config:
src/configs/ecg_baseline_wander_mecg_e.yaml

MambAttention config:
src/configs/ecg_baseline_wander_mambattention.yaml
```

## 0. 檢查專案

從專案根目錄執行：

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/check_project.sh
```

目前至少需要這三個 processed files 才能訓練：

```text
data/ecg_baseline_wander/processed/train.npz
data/ecg_baseline_wander/processed/val.npz
data/ecg_baseline_wander/processed/test.npz
```

若這三個檔案顯示 `MISSING`，先做第 1 步資料準備。

## 1. 準備資料

建議資料擺放位置：

```text
data/ecg_baseline_wander/
  raw/
    PTBXL/
      ptbxl_database.csv
      records100/
      records500/
    NSTDB/
    MITBIH/
    Chapman/
    CPSC/
  processed/
```

### 1.1 產生 PTB-XL train / val / test

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name ptbxl
```

完成後應產生：

```text
data/ecg_baseline_wander/processed/train.npz
data/ecg_baseline_wander/processed/val.npz
data/ecg_baseline_wander/processed/test.npz
```

PTB-XL fold 規則：

```text
train.npz: folds 1-8
val.npz: fold 9
test.npz: fold 10
```

### 1.2 產生外部測試資料

MIT-BIH：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/MITBIH \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name mit_bih
```

Chapman：

```bash
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/Chapman \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name chapman
```

CPSC：

```bash
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/CPSC \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name cpsc
```

完成後建議有：

```text
data/ecg_baseline_wander/processed/mit_bih.npz
data/ecg_baseline_wander/processed/chapman.npz
data/ecg_baseline_wander/processed/cpsc.npz
```

外部資料只做 testing，不可用於 training、validation、early stopping 或 checkpoint selection。

## 2. 訓練模型

兩個模型必須使用相同 train / validation / test split。

### 2.1 訓練 MECG-E

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/train_model.sh mecg_e
```

主要 checkpoint 輸出：

```text
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mecg_e/mecg_e/best_pcc_model.pt
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mecg_e/mecg_e/best_model.pt
```

### 2.2 訓練 MambAttention

```bash
cd /mnt/c/Users/中研院/rl_exp
bash scripts/train_model.sh mambattention
```

主要 checkpoint 輸出：

```text
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mambattention/mambattention_ecg/best_pcc_model.pt
runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mambattention/mambattention_ecg/best_model.pt
```

### 2.3 小型 smoke test

正式長訓練前可先確認資料與模型能跑：

```bash
cd /mnt/c/Users/中研院/rl_exp

bash scripts/train_model.sh mecg_e \
  training.train_iterations=2 \
  training.eval_every=1 \
  training.batch_size=2

bash scripts/train_model.sh mambattention \
  training.train_iterations=2 \
  training.eval_every=1 \
  training.batch_size=2
```

注意：`training.resume=False` 時，程式會清掉目前 run 的同名 checkpoint / results / log 資料夾。正式訓練中斷後要繼續，請加：

```bash
training.resume=True
```

## 3. 訓練後自動測試

`train_supervised.py` 訓練完成後會自動評估：

```text
Experiment 1: PTB-XL fold 10 in-domain test
Experiment 4: MIT-BIH external test，若 mit_bih.npz 存在
Experiment 5: Chapman external test，若 chapman.npz 存在
Experiment 6: CPSC external test，若 cpsc.npz 存在
Experiment 8: Complexity analysis
```

常見結果位置：

```text
runs/ecg_baseline_wander/results/<exp_name>/<model_name>/best_pcc/
runs/ecg_baseline_wander/results/<exp_name>/<model_name>/best_loss/
```

主要檔案：

```text
metrics_ptbxl_fold10_test.yaml
metrics_mit_bih.yaml
metrics_chapman.yaml
metrics_cpsc.yaml
complexity_summary.yaml
prediction_sample_*.png
spectrum_sample_*.png
```

正式比較建議優先使用 `best_pcc_model.pt` / `best_pcc` 結果；若要改用 `best_model.pt` / `best_loss`，兩個模型要一致。

## 4. 明確執行測試與 inference

若要指定 checkpoint 對某個 NPZ 重新測試：

MECG-E on PTB-XL fold 10：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 inference.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mecg_e/mecg_e/best_pcc_model.pt \
  --input /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/processed/test.npz \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mecg_e_ptbxl_fold10
```

MambAttention on PTB-XL fold 10：

```bash
python3 inference.py \
  --config configs/ecg_baseline_wander_mambattention.yaml \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mambattention/mambattention_ecg/best_pcc_model.pt \
  --input /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/processed/test.npz \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mambattention_ptbxl_fold10
```

把 `--input` 換成下面檔案即可測外部資料：

```text
data/ecg_baseline_wander/processed/mit_bih.npz
data/ecg_baseline_wander/processed/chapman.npz
data/ecg_baseline_wander/processed/cpsc.npz
```

Inference 輸出：

```text
restored_ecg.npz
metrics_summary.csv
metrics_per_window.csv
metrics_summary.json
```

## 5. Experiment 2：Baseline Strength Robustness

目的：比較模型在不同 baseline 強度下是否穩定。

強度：

```text
5%, 10%, 20%, 30%, 50%
alpha = 0.05, 0.1, 0.2, 0.3, 0.5
```

MECG-E：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mecg_e/mecg_e/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests/mecg_e \
  --alpha-values 0.05,0.1,0.2,0.3,0.5
```

MambAttention：

```bash
python3 experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_mambattention.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/NSTDB \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mambattention/mambattention_ecg/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests/mambattention \
  --alpha-values 0.05,0.1,0.2,0.3,0.5
```

輸出：

```text
runs/ecg_baseline_wander/controlled_tests/mecg_e/exp2_strength/summary.csv
runs/ecg_baseline_wander/controlled_tests/mambattention/exp2_strength/summary.csv
```

## 6. Experiment 3：Baseline Frequency Robustness

目的：比較模型在不同低頻漂移頻率下是否穩定。

頻率：

```text
0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0 Hz
```

MECG-E：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mecg_e/mecg_e/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests/mecg_e \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```

MambAttention：

```bash
python3 experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_mambattention.yaml \
  --input-dir /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --checkpoint /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_mambattention/mambattention_ecg/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests/mambattention \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```

輸出：

```text
runs/ecg_baseline_wander/controlled_tests/mecg_e/exp3_frequency/summary.csv
runs/ecg_baseline_wander/controlled_tests/mambattention/exp3_frequency/summary.csv
```

## 7. 彙整結果

產生總表：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 result_analysis.py aggregate \
  --results-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander \
  --output /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/paper_tables/mecge_vs_mambattention_aggregate.csv
```

## 8. Paired Statistical Test

目的：同一批 ECG windows 上，比較 MECG-E 與 MambAttention 是否有統計顯著差異。

範例使用 PTB-XL fold 10 inference 輸出的 per-window metrics：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 result_analysis.py paired-stats \
  --baseline /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mecg_e_ptbxl_fold10/metrics_per_window.csv \
  --candidate /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mambattention_ptbxl_fold10/metrics_per_window.csv \
  --output /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/statistics/mecge_vs_mambattention_ptbxl_fold10.csv \
  --metrics SSD,MAD,PRD,CosSim,SNR_Improvement_dB,LF_Reduction_dB,R_Peak_Timing_Error_ms,RR_Interval_MAE_ms,QRS_Amplitude_Error \
  --correction holm
```

可對 MIT-BIH、Chapman、CPSC 各自重複一次 paired test。

## 9. 挑選視覺化案例

每個模型至少挑 best / median / worst cases。

MECG-E：

```bash
cd /mnt/c/Users/中研院/rl_exp/src

python3 result_analysis.py select-cases \
  --inference-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mecg_e_ptbxl_fold10 \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/case_figures/mecg_e_ptbxl_fold10 \
  --metric PRD \
  --direction lower
```

MambAttention：

```bash
python3 result_analysis.py select-cases \
  --inference-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mambattention_ptbxl_fold10 \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/case_figures/mambattention_ptbxl_fold10 \
  --metric PRD \
  --direction lower
```

## 10. 最後報告應包含

主要表格：

```text
PTB-XL fold 10
MIT-BIH
Chapman
CPSC
```

每個 dataset 回報：

```text
SSD
MAD
PRD
CosSim
Output_SNR_dB
SNR_Improvement_dB
LF_Reduction_dB
R_Peak_Timing_Error_ms
RR_Interval_MAE_ms
QRS_Amplitude_Error
```

Robustness 圖：

```text
Baseline strength: alpha 0.05, 0.1, 0.2, 0.3, 0.5
Baseline frequency: 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0 Hz
```

Complexity 表：

```text
Parameters
Trainable_Parameters
FLOPs
Inference_Time_ms
Peak_Memory_MB
```

視覺化圖：

```text
Input ECG with Baseline Wander
Restored ECG
Clean reference
Residual error
PSD / spectrogram
```

結論要回答：

```text
1. MambAttention 是否比 MECG-E 有更低 PRD / SSD / MAD。
2. MambAttention 是否有更高 CosSim / SNR improvement / LF reduction。
3. MambAttention 是否更能保留 R-peak / RR interval / QRS amplitude。
4. MambAttention 在 external datasets 是否仍穩定。
5. MambAttention 的效能提升是否值得額外參數量、FLOPs、inference time。
```

