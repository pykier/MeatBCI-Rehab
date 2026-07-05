<img width="865" height="529" alt="image" src="https://github.com/user-attachments/assets/d61d9a47-b62d-4d79-88d4-980e54b2aac2" /># MetaBCI Demos And Rehabilitation MI Competition Workflow

本目录包含 MetaBCI 官方示例和本项目新增的康复运动想象系统。

本项目是：

> 基于 MetaBCI 的 VR/MR 康复场景运动想象机械手控制系统

其中 `demos/rehab_mi` 是比赛演示入口，核心能力在
`metabci.brainda`、`metabci.brainstim` 和 `metabci.brainflow`。MetaBCI 平台的独立系统。

声明：本项目是依托于MetaBCI官方框架建立的项目，官方地址为：https://github.com/TBC-TJU/MetaBCI
该平台为天津大学建设的脑机接口实验平台，详细介绍见官网链接

## 1. Directory Overview

```text
demos/
  rehab_mi/                  # 比赛主案例：康复 MI + VR + 机械手闭环
  brainstim_demos/           # MetaBCI brainstim 原始刺激范式示例
  brainflow_demos/           # MetaBCI brainflow 原始在线处理示例
  dynamic_stopping_demos/    # 动态停止相关示例
  BNCI2014004/               # 公开 MI 数据集文件目录
  *.py                       # brainda 算法与传统 demo 示例
```

比赛评审可以主要看 `demos/rehab_mi` 和 `metabci` 下新增的可复用模块：

```text
metabci/brainda/datasets/rehab_mi.py
metabci/brainda/algorithms/rehab.py
metabci/brainda/algorithms/deep_learning/fbmsnet.py
metabci/brainda/algorithms/decomposition/fbcspsvmrm.py
metabci/brainstim/rehab_mi.py
metabci/brainstim/vr.py
metabci/brainflow/neuracle.py
metabci/brainflow/rehab.py
metabci/brainflow/feedback.py
```
<img width="865" height="529" alt="image" src="https://github.com/user-attachments/assets/5574d7e0-3ed6-4075-9ee3-ee2f5d60b185" />


----------------------------------------重点快速理解-----------------------------------------------------------------------------------------------------
由于比赛需要测试，直接基于我们已经离线采集好的数据sub17，直接进行第六步 离线训练就行

本实验的主要流程为离线采集session1的数据、再次离线采集session2的数据、切分epoch、挑选某个模型离线训练、在线VR控制机械手

首先离线采集session1（分为两个终端分别执行）
终端1
.venv\Scripts\python.exe demos\rehab_mi\collect_dataset.py `
  --subject sub17 `
  --session formal01 `
  --srate 250 `
  --num-chans 17


终端2
.venv\Scripts\python.exe demos\rehab_mi\rehab_stim_demo.py `
  --direct `
  --nrep 15`
  --lsl-markers `
  --lsl-source-id rehab_mi_marker_stream `
  --feedback-mode target `
  --robot-feedback `
  --left-com COM4 `
  --right-com COM3
-----------------------------------

再次离线采集session2
终端1
.venv\Scripts\python.exe demos\rehab_mi\collect_dataset.py `
  --subject sub17 `
  --session formal02 `
  --srate 250 `
  --num-chans 17

终端2
.venv\Scripts\python.exe demos\rehab_mi\rehab_stim_demo.py `
  --direct `
  --nrep 15`
  --lsl-markers `
  --lsl-source-id rehab_mi_marker_stream `
  --feedback-mode target `
  --robot-feedback `
  --left-com COM4 `
  --right-com COM3
---------------------------------
切分epoch
python demos\rehab_mi\epoch_subject_sessions.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --overwrite
-----------------------------------



六、  挑选某个模型离线训练（举例三个模型）

python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm eegnet `
  --epochs 30 `
  --early-stopping-patience 100 `
  --batch-size 32 `
  --learning-rate 0.01 `
  --dropout 0.3 `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_eegnet.pkl

python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm svc `
  --out my_data\rehab_mi_models\sub017\sub17_all_sessions_svc.pkl


.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm fbcspsvm `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_fbcspsvm.pkl
-----------------------------------------



七、 在线VR控制机械手

python demos\rehab_mi\run_online_demo.py `
  --vr `
  --model my_data\rehab_mi_models\sub17\sub17_all_sessions_eegnet.pkl `
  --control-source prediction `
  --robot-mode serial `
  --robot-side both `
  --left-com COM4 `
  --right-com COM3 `
  --num-chans 17 `
  --eeg-chans 16 `
  --max-trials 10 `
  --nrep 5


其中换被试就更改sub17   换算法就改算法名字，svc之类的 以及对应的模型地址加载也要跟着换  formal01和formal02指的是两个session

----------------------------------------------------------------------------------------------------------------------------------------









## 2. Score Mapping

| Scoring item | Full-score evidence in this project |
| --- | --- |
| `brainda` | `RehabMIDataset`, `MotorImagery` epoching, EEGNet, FBMSNet, FBCSPSVM, FBCSPSVMRM, model bundle |
| `brainstim` | RehabMI state machine, PsychoPy stimulus, LSL marker, browser VR/MR Renderer |
| `brainflow` | Neuracle DataService, software marker alignment, online worker, rolling buffer, robot/FES feedback |
| New large-scale functions | VR/MR rehabilitation scene; EEG online prediction to robot-hand closed loop |
| New dataset | Needs at least 17 anonymized subjects for full dataset score |
| New paradigm/algorithm/device | RehabMI paradigm; FBMSNet/FBCSPSVM/FBCSPSVMRM; robot hand and FES interfaces |
| Improved functions | TCP reading robustness, marker deduplication, automatic crop/stop, fault-tolerant robot feedback |
| Usage degree | Demos are launchers; data, stimulus, online processing and devices are implemented under `metabci` |

代码已经支持统一数据结构，


## 3. Hardware Assumptions

默认硬件配置：

- EEG device: Neuracle EEG cap and Recorder DataService
- DataService endpoint: `127.0.0.1:8712`
- Sample rate: `250 Hz`
- Channels: `16 EEG + 1 marker/trigger = 17 channels`
- Left robot hand: `COM4`
- Right robot hand: `COM3`
- VR/MR headset: browser on the same LAN as the PC

运行任何机械手相关命令前，确认手和设备周围没有障碍物。

## 4. Environment Setup

在 PowerShell 中进入项目根目录：

```powershell
cd <MetaBCI-Rehab project root>
```

推荐使用仓库内虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果你仍使用 conda 环境，也可以：

```powershell
conda activate metabci39
```

后续命令推荐统一使用：

```powershell
.\.venv\Scripts\python.exe
```

这样可以避免 VS Code 选择错 Python 解释器。

## 5. Preflight Checks

### 5.1 Offline Preflight

离线采集前检查 EEG 数据流和机械手串口：

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode offline --num-chans 17 --left-com COM4 --right-com COM3 --move-hands
```

Expected result:

- 终端显示 EEG samples received。
- 显示 `channels=17`。
- 左右机械手各运动一次。
- COM 端口测试后自动释放。

### 5.2 Online + VR Preflight

在线演示前检查 EEG、机械手和 VR ：

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode online --num-chans 17 --left-com COM4 --right-com COM3 --move-hands --vr
```

Expected result:

- 终端显示 EEG 数据正常。
- 左右机械手各运动一次。
- 终端打印 VR URL，例如 `http://192.168.1.xxx:8766`。
- 在 VR 中打开该 URL，可以看到 MetaBCI 康复场景页面。

## 6. Configuration

默认配置文件：

```text
demos/rehab_mi/experiment_config.json
```

关键字段：

```json
{
  "subject": "sub01",
  "training_sessions": ["formal01", "formal02"],
  "data_root": "my_data/rehab_mi_software_marker",
  "model_root": "my_data/rehab_mi_models",
  "srate": 250,
  "num_chans": 17,
  "eeg_chans": 16,
  "tmin": 0.5,
  "tmax": 4.5
}
```

含义：

- `subject`: 当前默认被试编号。
- `training_sessions`: 默认参与训练的 session。
- `tmin/tmax`: 从 MOTOR_IMAGERY marker 开始截取的 EEG 时间窗。
- `num_chans`: Neuracle 输出总通道数，当前为 17。
- `eeg_chans`: 用于模型训练和预测的 EEG 通道数，当前为 16。

## 7. One-command Offline Workflow



1. 启动 `formal01` EEG 录制。
2. 启动 Brainstim 刺激界面。
3. 等待 `formal01` 刺激结束和数据自动保存。
4. 给被试休息。
5. 启动 `formal02` EEG 录制。
6. 启动 Brainstim 刺激界面。
7. 等待 `formal02` 保存完成。
8. 切分两个 session 的 epochs。
9. 训练某个模型并保存到 `my_data/rehab_mi_models/<subject>/`。

参数说明：

- `--subject sub04`: 被试编号，可以改成 `sub01`、`sub02` 等。
- `--sessions formal01 formal02`: 两个训练 session。
- `--nrep 15`: 每个 session 有 15 个 left/right repetition，即 30 个 trial。
- `--algorithm eegnet`: 训练算法。可选 `eegnet`、`fbmsnet`、`secnet`、`eegconformer`、`ifnet`、`mfanet`、`fbcspsvm`、`fbcspsvmrm`、`svc`、`centroid`。
- `--epochs 350`: 深度学习最大训练 epoch。
- `--early-stopping-patience 100`: 早停耐心值。
- `--dropout 0.5`: EEGNet/FBMSNet dropout。

如果不希望离线采集时机械手跟随 target 运动：

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_offline_pipeline.py --subject sub04 --sessions formal01 formal02 --nrep 15 --algorithm eegnet --no-robot-feedback
```

如果已经采过该 session，默认不会覆盖原始录制。确认要覆盖时加：

```powershell
--overwrite-recordings
```

## 8. Manual Offline Workflow

如果一键脚本出问题，可以用手动分步骤排错。

### 8.1 Session 1 Collection

Terminal 1: start EEG collection.

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\collect_dataset.py --subject sub04 --session formal01 --srate 250 --num-chans 17
```

Terminal 2: start Brainstim stimulus.

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\rehab_stim_demo.py --direct --nrep 15 --lsl-markers --lsl-source-id rehab_mi_marker_stream --feedback-mode target --robot-feedback --left-com COM4 --right-com COM3
```

### 8.2 Session 2 Collection

Terminal 1:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\collect_dataset.py --subject sub04 --session formal02 --srate 250 --num-chans 17
```

Terminal 2:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\rehab_stim_demo.py --direct --nrep 15 --lsl-markers --lsl-source-id rehab_mi_marker_stream --feedback-mode target --robot-feedback --left-com COM4 --right-com COM3
```

### 8.3 Epoching

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\epoch_subject_sessions.py --subject sub04 --sessions formal01 formal02 --overwrite
```

Expected result:

- 每个 session 生成 `epochs.npz`。
- 显示 `X shape`、`y shape` 和左右手类别数量。
- `Skipped markers` 应该为 0 或非常少。

### 8.4 Train EEGNet

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm eegnet --epochs 350 --early-stopping-patience 100 --batch-size 32 --learning-rate 0.005 --dropout 0.5 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_eegnet.pkl
```

### 8.5 Train FBMSNet

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm fbmsnet --epochs 350 --early-stopping-patience 100 --batch-size 32 --learning-rate 0.001 --dropout 0.5 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_fbmsnet.pkl
```

### 8.6 Train SECNet

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm secnet --epochs 350 --early-stopping-patience 100 --batch-size 16 --learning-rate 0.001 --dropout 0.2 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_secnet.pkl
```

### 8.7 Train EEGConformer

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm eegconformer --epochs 350 --early-stopping-patience 100 --batch-size 16 --learning-rate 0.001 --dropout 0.3 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_eegconformer.pkl
```

### 8.8 Train IFNet

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm ifnet --epochs 350 --early-stopping-patience 100 --batch-size 16 --learning-rate 0.001 --dropout 0.3 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_ifnet.pkl
```

### 8.9 Train MFANet

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm mfanet --epochs 350 --early-stopping-patience 100 --batch-size 16 --learning-rate 0.001 --dropout 0.1 --out my_data\rehab_mi_models\sub04\sub04_all_sessions_mfanet.pkl
```

### 8.10 Train FBCSP + SVM

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm fbcspsvm --out my_data\rehab_mi_models\sub04\sub04_all_sessions_fbcspsvm.pkl
```

### 8.11 Train FBCSP + SVM + Riemann

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\train_model.py --subject sub04 --sessions formal01 formal02 --algorithm fbcspsvmrm --out my_data\rehab_mi_models\sub04\sub04_all_sessions_fbcspsvmrm.pkl
```

## 9. Online VR Closed-loop Workflow

在线流程使用已经训练好的模型，对实时 EEG 做预测，并控制机械手。

### 9.1 EEGNet Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_eegnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.2 FBMSNet Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_fbmsnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.3 SECNet Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_secnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.4 EEGConformer Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_eegconformer.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.5 IFNet Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_ifnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.6 MFANet Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_mfanet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.7 FBCSP + SVM Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_fbcspsvm.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

### 9.8 FBCSP + SVM + Riemann Online Demo

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_fbcspsvmrm.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

Online command notes:

- `--vr`: 启动 VR/MR 场景服务。
- `--model`: 指定要加载的离线模型。
- `--control-source prediction`: 用模型预测结果控制机械手。
- `--control-source target`: 调试模式，用真实提示标签控制机械手。
- `--max-trials 10`: 在线最多收集并预测 10 个 trial。
- `--nrep 5`: Brainstim 生成 5 个 left/right repetition，即最多 10 个 trial。

## 10. Output Files

Offline data:

```text
my_data/rehab_mi_software_marker/
  sub04/
    formal01/
      recording.npz
      markers.csv
      meta.json
      epochs.npz
    formal02/
      recording.npz
      markers.csv
      meta.json
      epochs.npz
```

Model files:

```text
my_data/rehab_mi_models/
  sub04/
    sub04_all_sessions_eegnet.pkl
    sub04_all_sessions_eegnet.json
    sub04_all_sessions_eegnet.params.pt
```

Online logs are saved by the online demo under the configured output path.

## 11. Dataset Requirement For Full Score

To claim full dataset score, the final dataset should contain:

- at least 15 subjects,
- 2 sessions per subject,
- at least 30 valid left-hand and 30 valid right-hand trials per session,
- anonymous subject IDs, for example `sub01`, `sub02`, ...,
- metadata for sample rate, channel order, event definitions and timing,
- data loadable through `metabci.brainda.datasets.RehabMIDataset`.

Do not commit private raw EEG data directly to GitHub unless anonymization and
release permission are complete. Prefer GitHub Release, Git LFS, institutional
storage, or a documented download link.

## 12. GitHub Submission Checklist

Include:

```text
metabci/
demos/rehab_mi/
demos/README.md
docs/scoring_mapping.md
docs/project_test_report_template.md
docs/dataset_card.md
docs/hardware_setup.md
docs/github_submission_checklist.md
tests/
README.md
requirements.txt
setup.py
pyproject.toml
LICENSE
```

Do not include:

```text
.venv/
__pycache__/
my_data/ private raw data
checkpoints/
log.txt
*.npz
*.pkl
*.params.pt
WeChat temporary screenshots
large local runtime logs
```

Before upload:

```powershell
git status --short
```

Only commit intended source code, documentation, tests and safe demo assets.

## 13. Troubleshooting

### VR page cannot open on headset

1. Confirm PC and VR headset are on the same LAN.
2. Use the LAN URL printed by `run_online_demo.py`, not `127.0.0.1`.
3. Check Windows Firewall for Python.
4. Ensure ports `8765` and `8766` are not occupied by an old process.

### Neuracle DataService connection refused

1. Confirm Recorder is open and EEG waveform is visible.
2. Confirm DataService is enabled on port `8712`.
3. Do not run two acquisition clients at the same time.
4. Restart Recorder if the stream breaks repeatedly.

### Robot hand does not move

1. Run `preflight_check.py --move-hands`.
2. Confirm COM port numbers in Device Manager.
3. Close old Python terminals that may still hold COM3/COM4.
4. Keep robot power on and cables stable.

### Marker count mismatch

1. Use `--lsl-source-id rehab_mi_marker_stream` consistently.
2. Start collection before stimulus.
3. Ensure every trial sends exactly one MOTOR_IMAGERY marker.
4. Inspect `markers.csv` and `meta.json` in the session directory.

## 14. Why This Meets MetaBCI Usage Requirements

The competition application is built on MetaBCI because:

1. Dataset loading and epoching are owned by `brainda`.
2. MI algorithms and model bundles are owned by `brainda`.
3. Stimulus phases, LSL marker and VR/MR renderer are owned by `brainstim`.
4. Neuracle streaming, online prediction worker and robot feedback are owned by
   `brainflow`.
5. `demos/rehab_mi` only assembles these platform APIs into a reproducible
   competition demo.

For strict review, cite `docs/scoring_mapping.md` together with this file.
