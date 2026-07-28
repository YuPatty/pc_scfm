# ECG 基線漂移移除

本專案用於訓練與評估單導程 ECG 基線漂移移除模型。

此 benchmark 使用 Lead II ECG。PTB-XL 用於訓練、驗證與同域測試；MIT-BIH、Chapman 與 CPSC 則作為外部測試資料集，僅用於測試。

## 1. 專案結構

請從此目錄執行命令：

```bash
cd /mnt/c/Users/中研院/rl_exp/src
```

主要檔案：

```text
configs/
  ecg_baseline_wander_mecg_e.yaml
  ecg_baseline_wander_mambattention.yaml
  ecg_baseline_wander_pc_scfm.yaml

datasets/
  ecg_baseline_wander.py
  factory.py

models/
  mecg_e.py
  mambattention.py
  pc_scfm.py
  pc_scfm_components.py
  factory.py

preprocess_ecg.py      # 從 ECG records 建立 NPZ 檔案
train_supervised.py    # 訓練與評估模型
inference.py           # 從 NPZ 還原 ECG 並輸出指標
experiment_suite.py    # 實驗 2、3、7 的流程
result_analysis.py     # 彙整表格、案例選取、成對統計
utils.py
utils_ecg.py
analysis.py
```

可用的模型名稱：

```text
mecg_e
mambattention_ecg
pc_scfm
```

## 2. 環境

訓練與評估需要：

```text
torch
numpy
scipy
pyyaml
omegaconf
matplotlib
tensorboard
mamba_ssm
```

選用套件：

```text
wfdb   # 直接讀取 PhysioNet WFDB records
thop   # 在複雜度報告中計算 FLOPs
```

如果未安裝 `thop`，FLOPs 會回報為 `NaN`。訓練與推論仍可正常執行。

## 3. 資料切分

PTB-XL 遵循 DISR-ECG 論文使用的官方 fold 切分：

```text
train.npz  -> PTB-XL folds 1-8，約 80%
val.npz    -> PTB-XL fold 9，約 10%
test.npz   -> PTB-XL fold 10，約 10%
```

外部資料集僅用於測試：

```text
mit_bih.npz   -> 100% 僅測試
chapman.npz   -> 100% 僅測試
cpsc.npz      -> 100% 僅測試
```

外部資料不得用於訓練、驗證、early stopping、checkpoint 選擇或超參數調整。

## 4. 資料目錄

YAML 預設值為：

```yaml
data_dir: "/mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander"
```

資料載入器預期處理後的 NPZ 檔案放在：

```text
/mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/processed
```

Windows 路徑：

```text
C:\Users\中研院\rl_exp\data\ecg_baseline_wander\processed
```

必要檔案：

```text
train.npz
val.npz
test.npz
```

選用檔案：

```text
train_val.npz
mit_bih.npz
chapman.npz
cpsc.npz
```

如果缺少 `train_val.npz`，會使用 `val.npz` 進行 train-validation 監控。

## 5. NPZ 格式

每個 NPZ 必須包含：

```text
noisy_ecg
clean_reference
```

也接受替代 key：

```text
input
target
```

建議 shape：

```text
[N, 1, T]
```

也接受：

```text
[N, T]
```

目前預設：

```text
T = 512
sampling rate = 250 Hz
dtype = float32
```

請不要將 `[N, 12, T]` 這類多導程 tensor 傳入資料載入器。請先使用 `preprocess_ecg.py` 擷取 Lead II。

如果 NPZ 包含 `fold`、`strat_fold` 或 `ptbxl_fold`，載入器會檢查：

```text
train -> 只允許 folds 1-8
val   -> 只允許 fold 9
test  -> 只允許 fold 10
```

## 6. 前處理

`preprocess_ecg.py` 會將 ECG records 轉換成模型可直接使用的 NPZ 檔案。

處理流程：

```text
讀取 ECG
選擇 Lead II
濾波產生 clean reference
重新取樣
切割 window
正規化
合成基線漂移
寫入 NPZ
```

支援的輸入格式：

```text
NPZ
NPY
CSV/TXT 數值陣列
WFDB records，需安裝 wfdb
```

建立 PTB-XL train/val/test：

```bash
python preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --noise-dir /path/to/nstdb_records \
  --dataset-name ptbxl
```

對 PTB-XL 而言，metadata CSV 必須包含下列其中之一：

```text
strat_fold
fold
ptbxl_fold
```

建立外部測試檔案：

```bash
python preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/chapman_records \
  --noise-dir /path/to/nstdb_records \
  --dataset-name chapman
```

若要處理其他外部資料集，請將 `--dataset-name` 設為 `mit_bih` 或 `cpsc`。

外部資料集會輸出成：

```text
mit_bih.npz
chapman.npz
cpsc.npz
```

這些檔案只會被訓練後評估或推論讀取，不會參與 training、validation、early stopping 或 checkpoint 選擇。

只建立 test split：

```bash
python preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --noise-dir /path/to/nstdb_records \
  --dataset-name ptbxl \
  --splits test
```

覆寫 baseline 類型、強度或頻率：

```bash
python preprocess_ecg.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --baseline-kind sinusoidal \
  --alpha-values 0.05,0.1,0.2,0.3,0.5 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5
```

支援的 baseline 類型：

```text
nstdb
sinusoidal
multi_sine
random_low_frequency_drift
```

## 7. 訓練

訓練 MECG-E：

```bash
python train_supervised.py --config configs/ecg_baseline_wander_mecg_e.yaml
```

因為 MECG-E 是預設 config，所以下列命令等價：

```bash
python train_supervised.py
```

訓練 MambAttention：

```bash
python train_supervised.py --config configs/ecg_baseline_wander_mambattention.yaml
```

訓練 PC-SCFM：

```bash
python train_supervised.py --config configs/ecg_baseline_wander_pc_scfm.yaml
```

先執行小型 smoke test：

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  training.train_iterations=2 \
  training.eval_every=1 \
  training.batch_size=2
```

繼續訓練：

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  training.resume=True
```

從指定狀態繼續訓練：

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  training.resume=True \
  training.resume_checkpoint=/path/to/training_state.pt
```

重要行為：

```text
training.resume=False -> 清除目前 run 的 checkpoint/results/log 資料夾
training.resume=True  -> 保留既有資料夾，並從 training_state.pt 繼續
```

## 8. 測試與推論

訓練流程會自動評估：

```text
PTB-XL fold 10 test set
若存在外部測試 NPZ 檔案，也會評估：mit_bih.npz、chapman.npz、cpsc.npz
```

明確執行推論：

```bash
python inference.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --checkpoint /path/to/best_pcc_model.pt \
  --input /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/processed/test.npz \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mecg_e_test
```

可接受的 checkpoint：

```text
best_pcc_model.pt
best_model.pt
model_last.pt
training_state.pt
```

如果使用 `training_state.pt`，`inference.py` 會自動讀取 `model_state_dict`。

推論輸出：

```text
restored_ecg.npz
metrics_summary.csv
metrics_per_window.csv
metrics_summary.json
```

如果輸入 NPZ 沒有 `clean_reference` 或 `target`，只能計算無參考指標，目前主要是 `LF_Reduction_dB`。

## 9. 結果輸出路徑

預設根目錄：

```text
/mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander
```

Checkpoint 根目錄：

```text
runs/ecg_baseline_wander/checkpoint/<exp_name>/<model_name>/
```

常見 checkpoint 檔案：

```text
best_pcc_model.pt
best_model.pt
model_last.pt
training_state.pt
tensorboard_logs/
```

結果根目錄：

```text
runs/ecg_baseline_wander/results/<exp_name>/<model_name>/<best_pcc_or_best_loss>/
```

常見結果檔案：

```text
loss_curves/
prediction_sample_*.png
spectrum_sample_*.png
metrics_ptbxl_fold10_test.yaml
metrics_mit_bih.yaml
metrics_chapman.yaml
metrics_cpsc.yaml
complexity_summary.yaml
```

Log 根目錄：

```text
runs/ecg_baseline_wander/log/<exp_name>/<model_name>/log.log
```

## 10. 指標

與 MECG-E 對齊的指標：

```text
SSD     越低越好
MAD     越低越好
PRD     越低越好
CosSim  越高越好
```

`MAD` 表示 maximum absolute distance，不是 mean absolute deviation。

基線漂移指標：

```text
Output_SNR_dB        越高越好
SNR_Improvement_dB   越高越好
LF_Reduction_dB      越高越好
```

形態學指標：

```text
R_Peak_Timing_Error_ms  越低越好
RR_Interval_MAE_ms      越低越好
QRS_Amplitude_Error     越低越好
```

複雜度指標：

```text
Parameters
Trainable_Parameters
FLOPs
Inference_Time_ms
Peak_Memory_MB
```

低頻頻帶由下列設定控制：

```yaml
evaluation:
  low_frequency_high_hz: 0.5
```

## 11. 實驗 2：Baseline 強度魯棒性

此流程會建立不同 baseline 強度的受控 PTB-XL fold 10 測試檔案、執行推論，並輸出 summary。

```bash
python experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --noise-dir /path/to/nstdb_records \
  --checkpoint /path/to/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests \
  --alpha-values 0.05,0.1,0.2,0.3,0.5
```

輸出：

```text
controlled_tests/exp2_strength/summary.csv
```

## 12. 實驗 3：Baseline 頻率魯棒性

此流程會建立不同 baseline 頻率的受控 PTB-XL fold 10 測試檔案、執行推論，並輸出 summary。

```bash
python experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --checkpoint /path/to/best_pcc_model.pt \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/controlled_tests \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```

輸出：

```text
controlled_tests/exp3_frequency/summary.csv
```

## 13. 實驗 7：消融實驗

消融實驗限定用於 PC-SCFM。其他模型仍可執行實驗 2、實驗 3、外部資料集測試與一般推論，但不套用 PC-SCFM 消融。

產生消融實驗 config：

```bash
python experiment_suite.py exp7-ablation \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander
```

產生 config 並訓練所有變體：

```bash
python experiment_suite.py exp7-ablation \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --output-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander \
  --run-train
```

目前的消融變體：

```text
full
one_shot
fixed_multistep
no_flow
no_reject
no_safety
phase_sincos
```

輸出：

```text
exp7_ablation/ablation_plan.csv
exp7_ablation/configs/*.yaml
exp7_ablation/summary.csv
```

## 14. 結果分析

彙整 inference 或 sweep 產生的 `metrics_summary.csv`：

```bash
python result_analysis.py aggregate \
  --results-root /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander \
  --output /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/paper_tables/aggregate_metrics.csv
```

注意：訓練結束自動評估會輸出 `metrics_*.yaml`；`aggregate` 目前不會直接彙整這些 YAML。若要進入 `aggregate` 流程，請先用 `inference.py` 對指定 checkpoint 產生 `metrics_summary.csv`，或使用 `experiment_suite.py` 產生 sweep summary。

選取最佳、中位數與最差案例：

```bash
python result_analysis.py select-cases \
  --inference-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/mecg_e_test \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/case_figures/mecg_e_test \
  --metric PRD \
  --direction lower
```

執行成對統計：

```bash
python result_analysis.py paired-stats \
  --baseline /path/to/baseline/metrics_per_window.csv \
  --candidate /path/to/candidate/metrics_per_window.csv \
  --output /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/statistics/mecg_vs_baseline.csv \
  --metrics SSD,MAD,PRD,CosSim,SNR_Improvement_dB,LF_Reduction_dB \
  --correction holm
```

`paired-stats` 會輸出 paired t-tests、Wilcoxon signed-rank tests、校正後 p-values 與顯著性標記。

## 15. 調整參數

你可以編輯 YAML 檔案，或從命令列覆寫設定值。

常用訓練參數：

```yaml
training:
  batch_size: 32
  train_iterations: 20000
  eval_every: 100
  save_every: 5000
  lr: 1.0e-4
  early_stopping_patience: 50
```

命令列範例：

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  training.batch_size=16 \
  training.lr=5e-5 \
  training.train_iterations=50000
```

常用資料集參數：

```yaml
dataset:
  resample_hz: 250
  window_size: 512
  overlap_ratio: 0.5
  normalization: z_score
  clean_reference:
    bandpass_hz: [0.05, 40.0]
```

常用模型參數：

```yaml
model:
  fea: "pha"
  dense_channel: 32
  compress_factor: 0.3
  num_tscblocks: 4
  d_state: 16
  d_conv: 4
  expand: 4
  n_fft: 64
  hop_size: 8
  win_size: 64
  loss_fn: "time+com+con"
```

範例：

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  model.dense_channel=64 \
  model.num_tscblocks=6
```

```bash
python train_supervised.py \
  --config configs/ecg_baseline_wander_mecg_e.yaml \
  model.loss_fn=time+com
```

MambAttention 專用參數：

```yaml
model:
  attention_heads: 8
  attention_dropout: 0.0
```

PC-SCFM 專用 config：

```bash
python train_supervised.py --config configs/ecg_baseline_wander_pc_scfm.yaml
```

PC-SCFM 使用 `model_name: pc_scfm`，並在 `model.pcscfm_enabled: true` 時啟用 multi-step restoration、flow proposal、risk policy、reject 與 morphology safety。

## 16. 新增模型

在 `src/models/` 下新增模型 class、註冊模型、匯入模型，然後建立 config。

### 16.1 建立模型檔案

範例：`src/models/my_model.py`

```python
import torch
import torch.nn as nn

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


@register_model("my_ecg_model")
class MyECGModel(nn.Module):
    def __init__(self, hidden_channels=64, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3),
        )

    def forward(self, noisy):
        if noisy.ndim == 2:
            noisy = noisy.unsqueeze(1)
        return self.net(noisy)

    def compute_loss(self, batch, device, **kwargs):
        noisy, clean = batch[0].to(device), batch[1].to(device)
        pred = self.forward(noisy)
        return torch.mean(torch.abs(pred - clean))
```

必要介面：

```text
forward(noisy) -> restored ECG，shape 為 [B, 1, T]
compute_loss(batch, device, **kwargs) -> scalar loss
```

訓練 loop 預期每個 dataset item 回傳：

```text
(noisy_ecg, clean_reference)
```

### 16.2 匯入模型

編輯 `src/models/__init__.py`：

```python
from .factory import *
from .mecg_e import *
from .my_model import *
```

### 16.3 建立 Config

複製既有 config：

```bash
cp configs/ecg_baseline_wander_mecg_e.yaml configs/ecg_baseline_wander_my_model.yaml
```

編輯：

```yaml
exp_name: ptbxl_lead2_baseline_wander_my_model
model_name: my_ecg_model

model:
  hidden_channels: 64
```

除非實驗明確需要不同設定，否則請保留相同的 `dataset` split 與前處理設定。

### 16.4 訓練新模型

```bash
python train_supervised.py --config configs/ecg_baseline_wander_my_model.yaml
```

### 16.5 測試新模型

```bash
python inference.py \
  --config configs/ecg_baseline_wander_my_model.yaml \
  --checkpoint /path/to/best_pcc_model.pt \
  --input /mnt/c/Users/中研院/rl_exp/data/ecg_baseline_wander/processed/test.npz \
  --output-dir /mnt/c/Users/中研院/rl_exp/runs/ecg_baseline_wander/inference/my_model_test
```

## 17. 建議完整流程

1. 準備 PTB-XL records、PTB-XL metadata CSV、NSTDB records，以及選用的外部資料集。
2. 執行 `preprocess_ecg.py`，建立 `train.npz`、`val.npz`、`test.npz` 與外部 NPZ 檔案。
3. 執行 2-iteration smoke test。
4. 訓練 `mecg_e`。
5. 訓練 `mambattention_ecg`。
6. 訓練 `pc_scfm`。
7. 視需要在 test 與外部 NPZ 檔案上執行明確推論。
8. 對所有主要模型執行實驗 2 強度 sweep。
9. 對所有主要模型執行實驗 3 頻率 sweep。
10. 只對 PC-SCFM 執行實驗 7 消融實驗。
11. 執行 `result_analysis.py aggregate`。
12. 執行 `result_analysis.py paired-stats`。
13. 執行 `result_analysis.py select-cases`。

## 18. 疑難排解

缺少資料：

```text
檢查 data_dir。
檢查 processed/train.npz。
檢查 processed/val.npz。
檢查 processed/test.npz。
```

NPZ key 錯誤：

```text
使用 noisy_ecg + clean_reference，或 input + target。
```

Shape 錯誤：

```text
使用 [N, T] 或 [N, 1, T]。
不要使用 [N, 12, T]。
```

Mamba import error：

```text
在訓練環境中安裝 mamba_ssm。
```

FLOPs 是 NaN：

```text
如果需要 FLOPs，請安裝 thop。
```

沒有 reference metrics：

```text
如果輸入 NPZ 沒有 clean_reference/target，則無法計算 PRD、SNR improvement、RMSE、cosine similarity 與 morphology metrics。
```
