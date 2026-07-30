# rl_exp：ECG Baseline Wander Removal 實驗環境

這個專案是 ECG 基線漂移移除的實驗環境。主要維護程式碼放在 `src/`，原始 PC-SCFM 程式只保留在 `references/pc_scfm_original/` 當參考，不再當作主程式入口。

目前模型關係：

```text
mecg_e
  MECG-E backbone

mambattention_ecg
  MECG-E + Multi-Head Attention

pc_scfm
  MambAttention + flow proposal + policy/controller + reject + morphology safety
```

PC-SCFM 消融實驗只針對 `pc_scfm`。一般訓練、推論、robustness sweep 則三個模型都可以跑。

## 專案結構

```text
rl_exp/
  src/           主程式、模型、資料載入、config
  scripts/       常用指令包裝
  docs/          專案結構說明
  notes/         中文實驗筆記
  references/    原始或參考實作
  runs/          訓練與實驗輸出，不進版控
```

重要檔案：

```text
DATA_LAYOUT.md          這個專案的資料放置規則
requirements.txt        Python 依賴
src/README.md           更細的實驗文件
docs/PROJECT_STRUCTURE.md
```

模型檔案：

```text
src/models/mecg_e.py
src/models/mambattention.py
src/models/pc_scfm.py
src/models/pc_scfm_components.py
src/models/factory.py
```

Config 檔案：

```text
src/configs/ecg_baseline_wander_mecg_e.yaml
src/configs/ecg_baseline_wander_mambattention.yaml
src/configs/ecg_baseline_wander_pc_scfm.yaml
```

## 環境安裝

本專案已改成可攜式路徑。YAML 內的 `data_dir` 與 `root_dir` 使用相對路徑，預設從 `src/` 解析到：

```text
../data/ecg_baseline_wander
../runs/ecg_baseline_wander
```

遠端主機只需要保持 repo 結構相同，從專案根目錄使用 `scripts/*.sh` 執行即可，不需要修改成 `/mnt/c/...`。

從專案根目錄執行：

```bash
cd <PROJECT_ROOT>
pip install -r requirements.txt
```

檢查專案結構、資料位置、Python 語法：

```bash
bash scripts/check_project.sh
```

如果看到 `train.npz`、`val.npz`、`test.npz` 是 `MISSING`，代表資料還沒放到預設位置；語法檢查仍可能是 OK。

## 資料要放哪裡

預設資料根目錄：

```text
<PROJECT_ROOT>/data/ecg_baseline_wander
```

必要檔案：

```text
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/train.npz
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/val.npz
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/test.npz
```

可選外部測試檔：

```text
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/mit_bih.npz
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/chapman.npz
<PROJECT_ROOT>/data/ecg_baseline_wander/processed/cpsc.npz
```

每個 NPZ 建議包含：

```text
noisy_ecg
clean_reference
```

也接受：

```text
input
target
```

Shape 建議：

```text
[N, 1, T]
```

也接受：

```text
[N, T]
```

更完整規則請看 `DATA_LAYOUT.md`。

## 前處理

訓練不會自動把原始 ECG records 轉成 NPZ。你要先執行 `src/preprocess_ecg.py`，建立 `processed/train.npz`、`processed/val.npz`、`processed/test.npz`。

範例：建立 PTB-XL train/val/test。

```bash
cd <PROJECT_ROOT>/src

python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/PTBXL/records100 \
  --metadata-csv ../data/ecg_baseline_wander/raw/PTBXL/ptbxl_database.csv \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name ptbxl
```

如果你目前只先下載完整 PTB-XL `records100` 和 `ptbxl_database.csv`，可以直接從專案根目錄執行：

```bash
cd <PROJECT_ROOT>
bash scripts/preprocess_ptbxl_records100.sh
```

這個腳本會固定使用 `records100`、官方 `ptbxl_database.csv`、Lead II、PTB-XL folds 1-8/9/10，以及 config 內的 clean reference、window、normalization、baseline strength/frequency 設定。如果 `raw/NSTDB` 有資料，會使用 NSTDB baseline；如果還沒有 NSTDB，會先用合成 random low-frequency drift，讓 records100-only 實驗可以先跑。

範例：建立外部測試 NPZ。

```bash
python3 preprocess_ecg.py \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --input-dir ../data/ecg_baseline_wander/raw/MITBIH \
  --noise-dir ../data/ecg_baseline_wander/raw/NSTDB \
  --dataset-name mit_bih
```

外部資料集請把 `--dataset-name` 換成：

```text
mit_bih
chapman
cpsc
```

## 訓練模型

建議從專案根目錄使用 `scripts/train_model.sh`。

訓練 PC-SCFM：

```bash
cd <PROJECT_ROOT>
bash scripts/train_model.sh pc_scfm
```

訓練 MECG-E：

```bash
cd <PROJECT_ROOT>
bash scripts/train_model.sh mecg_e
```

訓練 MambAttention：

```bash
cd <PROJECT_ROOT>
bash scripts/train_model.sh mambattention
```

覆蓋 config 參數：

```bash
bash scripts/train_model.sh pc_scfm training.train_iterations=1000 training.batch_size=16
```

直接從 `src/` 執行也可以：

```bash
cd <PROJECT_ROOT>/src
python3 train_supervised.py --config configs/ecg_baseline_wander_pc_scfm.yaml
```

## 輸出在哪裡

預設輸出根目錄：

```text
<PROJECT_ROOT>/runs/ecg_baseline_wander
```

Checkpoint：

```text
runs/ecg_baseline_wander/checkpoint/<exp_name>/<model_name>/
  best_pcc_model.pt
  best_model.pt
  model_last.pt
  training_state.pt
```

Metric、圖、分析結果：

```text
runs/ecg_baseline_wander/results/<exp_name>/<model_name>/
```

Log：

```text
runs/ecg_baseline_wander/log/<exp_name>/<model_name>/log.log
```

## 推論

範例：用訓練好的 PC-SCFM checkpoint 對 `test.npz` 推論。

```bash
cd <PROJECT_ROOT>/src

python3 inference.py \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --checkpoint ../runs/ecg_baseline_wander/checkpoint/ptbxl_lead2_baseline_wander_pc_scfm/pc_scfm/best_pcc_model.pt \
  --input ../data/ecg_baseline_wander/processed/test.npz \
  --output-dir ../runs/ecg_baseline_wander/inference/pc_scfm_test
```

推論結果會寫到 `--output-dir`。

## PC-SCFM 消融實驗

消融實驗入口：

```bash
cd <PROJECT_ROOT>
```

只產生消融 config，不訓練：

```bash
bash scripts/run_pcscfm_ablation.sh --summarize-only
```

正式跑所有 PC-SCFM 消融：

```bash
bash scripts/run_pcscfm_ablation.sh --run-train
```

先做短測試，確認流程能跑：

```bash
bash scripts/run_pcscfm_ablation.sh \
  --train-iterations 10 \
  --eval-every 5 \
  --run-train
```

目前消融 variant：

```text
full
one_shot
fixed_multistep
no_flow
no_reject
no_safety
phase_sincos
```

彙整已完成的消融：

```bash
bash scripts/summarize_pcscfm_ablation.sh
```

摘要輸出：

```text
runs/ecg_baseline_wander/exp7_ablation/summary.csv
```

## Robustness Sweep

這類實驗不是 PC-SCFM 專用。只要換 `--config` 和 `--checkpoint`，其他模型也可以跑。

Baseline strength sweep：

```bash
cd <PROJECT_ROOT>/src

python3 experiment_suite.py exp2-strength \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --noise-dir /path/to/nstdb_records \
  --checkpoint /path/to/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests \
  --alpha-values 0.05,0.1,0.2,0.3,0.5
```

Baseline frequency sweep：

```bash
cd <PROJECT_ROOT>/src

python3 experiment_suite.py exp3-frequency \
  --config configs/ecg_baseline_wander_pc_scfm.yaml \
  --input-dir /path/to/ptbxl_records \
  --metadata-csv /path/to/ptbxl_database.csv \
  --checkpoint /path/to/best_pcc_model.pt \
  --output-root ../runs/ecg_baseline_wander/controlled_tests \
  --baseline-kind sinusoidal \
  --alpha-value 0.2 \
  --frequencies-hz 0.05,0.1,0.2,0.3,0.5,0.8,1.0
```

## 新模型要放哪裡

新模型放在：

```text
src/models/<your_model>.py
```

基本要求：

```text
forward(noisy) -> restored ECG，shape [B, 1, T]
compute_loss(batch, device, **kwargs) -> scalar loss
```

最小範例：

```python
import torch
import torch.nn as nn

try:
    from .factory import register_model
except ImportError:
    from factory import register_model


@register_model("my_new_model")
class MyNewModel(nn.Module):
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

然後在這裡 import：

```text
src/models/__init__.py
```

加入：

```python
from .my_new_model import *
```

新增 config：

```bash
cd <PROJECT_ROOT>/src
cp configs/ecg_baseline_wander_mecg_e.yaml configs/ecg_baseline_wander_my_new_model.yaml
```

把 config 裡面改成：

```yaml
exp_name: ptbxl_lead2_baseline_wander_my_new_model
model_name: my_new_model

model:
  hidden_channels: 64
```

訓練：

```bash
cd <PROJECT_ROOT>/src
python3 train_supervised.py --config configs/ecg_baseline_wander_my_new_model.yaml
```

如果你希望根目錄指令也支援新模型，就更新：

```text
scripts/train_model.sh
```

在裡面替新模型加一個 case。

## 之後可以怎麼下 Prompt

請 Codex 幫你加新模型時，可以直接這樣下：

```text
請在 <PROJECT_ROOT>/src/models 新增一個模型叫 my_new_model。
它要支援 forward(noisy)->[B,1,T] 和 compute_loss(batch, device)。
請幫我新增 src/models/my_new_model.py、更新 src/models/__init__.py、
新增 src/configs/ecg_baseline_wander_my_new_model.yaml，
並跑 python 語法檢查。
不要改資料路徑，不要動 references/。
```

如果你已經有架構想法：

```text
請幫我加入一個 ECG baseline wander removal 模型，模型名稱是 temporal_unet。
架構用 1D U-Net，輸入 noisy ECG [B,1,T]，輸出 restored ECG [B,1,T]。
loss 用 L1 + low-frequency residual loss。
請放在 src/models/temporal_unet.py，註冊 model_name: temporal_unet，
新增對應 config，並更新 README 的模型列表。
```

如果是改 PC-SCFM：

```text
請只修改 PC-SCFM，不要改 mecg_e 或 mambattention baseline。
我要測試新的 reject rule：flow_uncertainty 超過 X 或 proposal disagreement 超過 Y 就 reject。
請修改 src/models/pc_scfm_components.py 或相關 PC-SCFM 流程，
並更新 ablation config 或 README。
```

如果是新增可比較 baseline：

```text
請幫我新增一個 baseline 模型，名稱是 <model_name>。
它要能使用和 mecg_e、mambattention、pc_scfm 相同的資料、訓練、推論與 robustness sweep。
請新增模型檔、config、註冊、README 使用方式，並跑 scripts/check_project.sh。
```

## 維護規則

主程式碼放 `src/`。

`references/pc_scfm_original/` 只當參考，不把它當主訓練入口。

`runs/`、資料、checkpoint、圖、實驗輸出不要進版控。

改完結構後跑：

```bash
bash scripts/check_project.sh
```
