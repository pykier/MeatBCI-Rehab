# 念力搬砖BCI赛队作品测试说明

项目名称：基于虚拟现实的运动康复机械手控制系统  
项目仓库：[https://github.com/pykier/MeatBCI-Rehab](https://github.com/pykier/MeatBCI-Rehab)

本说明面向 MetaBCI 创新应用开发赛项代码测试。仓库中的 `demos/rehab_mi` 是完整演示入口，`metabci/` 下包含本项目对 MetaBCI 三个子平台的新增、修复和封装能力。
声明： MetaBCI官方平台由天津大学设计，官方网址为：https://github.com/TBC-TJU/MetaBCI
评测人员如果没有 VR 眼镜、Neuracle 脑电帽和机械手，也可以使用仓库中提供的 `sub17` 验证数据和 `sim` 模式完成离线训练、模型指标验证和无硬件在线流程演示。
如果完整测试，请联系赛队负责人
<img width="865" height="529" alt="image" src="https://github.com/user-attachments/assets/4d4fc3a2-dba2-44f6-a207-dd6724369dc3" />

<img width="865" height="445" alt="image" src="https://github.com/user-attachments/assets/aed57c6a-6760-42ee-a41e-fb92c49757d3" />
<img width="542" height="359" alt="image" src="https://github.com/user-attachments/assets/c21e1d26-e205-4990-9be6-6f42c32f10c5" />


## 1. 项目功能概述

本项目基于 MetaBCI 平台实现运动想象（Motor Imagery, MI）二分类闭环康复系统。系统支持左右手运动想象刺激、Neuracle 脑电采集、1-4 s 运动想象 EEG 时间窗切片、多算法离线训练、在线实时预测、VR/MR 场景同步显示和左右康复机械手反馈。

整体链路如下：

```text
brainstim 康复 MI 范式 / VR 场景
        ↓  LSL marker / phase event
brainflow Neuracle 数据接入 / marker 同步 / 在线切片
        ↓  1-4 s MI epoch
brainda EEGNet / SECNet / FBMSNet / FBCSP 等算法预测
        ↓  left_hand / right_hand
brainflow 机械手反馈 / VR 反馈 / sim 模式日志输出
```

核心特点：

- 支持左右手 2 类 MI 任务，在线预测使用 1-4 s 共 3 秒 EEG 数据，匹配赛事基础性能指标中的 3 秒分类指标。
- 支持真实硬件闭环：Neuracle 脑电帽、VR/MR 场景、COM3/COM4 康复机械手。
- 支持无硬件测试：使用 `sub17` 验证数据训练模型，并使用 `--robot-mode sim` 验证在线流程。
- 支持多算法离线训练：`eegnet`、`secnet`、`fbmsnet`、`ifnet`、`mfanet`、`fbcspsvm`、`fbcspsvmrm`。

## 2. MetaBCI 平台使用与评分映射

本项目是在 MetaBCI 三个子平台基础上扩展康复 MI 场景。`demos/rehab_mi` 仅作为案例入口，核心复用和扩展功能位于 `metabci/`。

| 评分项 | MetaBCI 子平台 | 本项目功能 | 主要代码路径 |
|---|---|---|---|
| 使用 brainda | Brainda | RehabMI 数据集、MotorImagery 范式切片、多算法训练和模型保存 | `metabci/brainda/datasets/rehab_mi.py`、`metabci/brainda/paradigms/imagery.py`、`metabci/brainda/algorithms/` |
| 使用 brainstim | Brainstim | 康复 MI 状态机、PsychoPy 刺激、LSL marker、VR/MR 场景事件同步 | `metabci/brainstim/rehab_mi.py`、`metabci/brainstim/vr.py`、`demos/rehab_mi/rehab_stim_demo.py` |
| 使用 brainflow | Brainflow | Neuracle DataService 接入、软件 marker 同步、在线切片、机械手反馈、sim 模式 | `metabci/brainflow/neuracle.py`、`metabci/brainflow/feedback.py`、`demos/rehab_mi/online_neuracle_closed_loop.py` |
| 新增大规模功能 | 三个平台协同 | VR/MR 康复场景 + EEG 在线预测 + 机械手闭环控制 | `demos/rehab_mi/run_online_demo.py` |
| 新增数据集 | Brainda | RehabMI 数据集，支持 subject/session 结构和 MetaBCI 加载 | `my_data/rehab_mi_software_marker/`、`metabci/brainda/datasets/rehab_mi.py` |
| 新增范式/算法/设备 | Brainstim/Brainda/Brainflow | RehabMI 范式、SECNet/FBMSNet/IFNet/MFANet/FBCSPSVMRM、机械手和 VR 反馈接口 | `metabci/brainstim/rehab_mi.py`、`metabci/brainda/algorithms/`、`metabci/brainflow/feedback.py` |
| 修复/优化 | 三个平台协同 | marker 去重、Neuracle 断连容错、串口反馈容错、深度模型在线 float32 输入统一 | `metabci/brainflow/neuracle.py`、`metabci/brainflow/feedback.py`、`metabci/brainda/algorithms/rehab.py` |

## 3. 代码目录结构

```text
metabci/
  brainda/
    datasets/rehab_mi.py                 # RehabMI 数据集接口
    algorithms/deep_learning/            # EEGNet、SECNet、FBMSNet、IFNet、MFANet 等
    algorithms/decomposition/            # FBCSP、FBCSPSVM、FBCSPSVMRM 等
  brainstim/
    rehab_mi.py                          # 康复 MI 范式、阶段状态机、PsychoPy 渲染、marker 发送
    vr.py                                # VR/MR 场景事件发送与同步
  brainflow/
    neuracle.py                          # Neuracle DataService、软件 marker 桥接、离线记录
    feedback.py                          # 机械手串口反馈、模拟反馈、闭环反馈封装

demos/
  README.md                              # 本测试说明
  rehab_mi/
    README.md                            # 康复 MI 案例补充说明
    experiment_config.json               # 默认实验参数，含 1-4 s 时间窗
    collect_dataset.py                   # 真实硬件离线采集入口
    run_offline_pipeline.py              # 一键离线采集、切 epoch、训练
    epoch_subject_sessions.py            # 多 session epoch 切分
    train_model.py                       # 多算法离线训练入口
    run_online_demo.py                   # 在线 VR/机械手闭环总入口
    preflight_check.py                   # EEG、机械手、VR 预检
    vr_scene_server.py                   # 本地 VR/MR 网页服务
    assets/                              # 刺激图片、机械手图片等资源

my_data/
  rehab_mi_software_marker/sub17/        # 用于赛事测试的 sub17 数据
  rehab_mi_models/sub17/                 # sub17 对应模型或测试输出

docs/
  rehab_mi_platform.md                   # 平台化设计与评分映射补充说明
```

## 4. 环境安装

推荐系统：Windows 10/11  
推荐 Python：3.9  
推荐执行目录：仓库根目录，即包含 `metabci/`、`demos/`、`requirements.txt` 的目录。

### 4.1 创建环境

```powershell
cd <repo-root>
conda create -n metabci39 python=3.9 -y
conda activate metabci39
python -m pip install --upgrade pip setuptools wheel
```

如果已使用虚拟环境，也可以直接进入虚拟环境后继续安装依赖。

### 4.2 安装依赖

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

如果出现 `pkg_resources` 缺失，可执行：

```powershell
python -m pip install "setuptools==68.2.2" wheel
```

### 4.3 检查基础导入

```powershell
python -c "import metabci, mne, sklearn, torch; print('basic import ok')"
python -c "import psychopy, pylsl, serial; print('runtime import ok')"
```

说明：

- PsychoPy 可能输出 `pkg_resources is deprecated` 警告，一般不影响运行。
- 若评测电脑没有 VR、脑电帽和机械手，请直接使用第 5 节的无硬件测试流程。

## 5. 无硬件测试流程（推荐评测人员优先使用）

本流程不需要 Neuracle 脑电帽、VR 眼镜或机械手，只使用仓库中的 `sub17` 验证数据完成离线指标验证，并通过模拟反馈检查在线流程是否可运行。

### 5.1 使用 sub17 数据切分 epoch

```powershell
cd <repo-root>
conda activate metabci39

python demos\rehab_mi\epoch_subject_sessions.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --overwrite
```

期望结果：

- 程序读取 `my_data\rehab_mi_software_marker\sub17\formal01` 和 `formal02`。
- 输出每个 session 的 `epochs.npz`。
- 运动想象时间窗为 1-4 s，共 3 秒 EEG 数据。
- 输出类别数量，左右手样本数应基本一致。

### 5.2 离线训练并验证分类指标

推荐使用 `secnet` 作为赛事验证算法：

```powershell
python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm secnet `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_secnet.pkl
```

可选算法命令示例：

```powershell
python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm eegnet `
  --epochs 150 `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_eegnet.pkl
```

```powershell
python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm fbcspsvm `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_fbcspsvm.pkl
```

期望结果：

- 终端打印数据形状、类别数量、算法名称、训练/验证日志或交叉验证结果。
- 模型保存到 `my_data\rehab_mi_models\sub17\`。
- 本项目提交测试记录中，`sub17` 两个 session、60 个 trial、16 导联、3 s 数据窗，`secnet` 本地测试分类正确率约为 75%。

### 5.3 无硬件启动 VR 网页和刺激流程

终端 1：启动 VR/MR 本地网页服务。

```powershell
python demos\rehab_mi\vr_scene_server.py --port 8766 --udp-port 8765
```

然后在浏览器中打开：

```text
http://127.0.0.1:8766
```

终端 2：启动无硬件刺激流程，并向网页发送阶段事件。

```powershell
python demos\rehab_mi\rehab_stim_demo.py `
  --direct `
  --nrep 3 `
  --feedback-mode random `
  --vr-events
```

期望结果：

- 电脑端 PsychoPy 刺激界面显示 READY、TRIAL、PROMPT、MOTOR IMAGERY、FEEDBACK、REST。
- 浏览器网页同步显示对应阶段。
- 此流程仅验证范式和 VR/MR 场景同步，不需要脑电帽和机械手。

### 5.4 无硬件在线闭环模拟

如果评测人员没有机械手和脑电帽，可使用 `sim` 模式验证在线流程入口、模型加载、VR 同步和反馈日志：

```powershell
python demos\rehab_mi\run_online_demo.py `
  --vr `
  --model my_data\rehab_mi_models\sub17\sub17_all_sessions_secnet.pkl `
  --control-source target `
  --robot-mode sim `
  --robot-side both `
  --num-chans 17 `
  --eeg-chans 16 `
  --max-trials 6 `
  --nrep 3
```

说明：

- `--robot-mode sim` 不打开串口，只在终端输出模拟反馈。
- `--control-source target` 使用范式目标作为反馈结果，用于无脑电帽环境下验证在线流程。
- 如果要验证真实 EEG 在线预测，需要使用第 6 节硬件流程。

## 6. 真实硬件闭环流程

硬件默认配置：

| 设备 | 默认配置 |
|---|---|
| Neuracle Recorder DataService | `127.0.0.1:8712` |
| EEG 采样率 | `250 Hz` |
| 数据通道 | `--num-chans 17`，其中 16 EEG + 1 marker |
| EEG 通道数 | `--eeg-chans 16` |
| 左手机械手 | `COM4` |
| 右手机械手 | `COM3` |
| VR 网页端口 | `8766` |
| VR 事件端口 | `8765` |

### 6.1 采集前硬件预检

```powershell
python demos\rehab_mi\preflight_check.py `
  --mode offline `
  --num-chans 17 `
  --left-com COM4 `
  --right-com COM3 `
  --move-hands
```

期望结果：

- 终端显示 EEG DataService 能读取到数据。
- 左右机械手各运动一次。
- 如果机械手未连接，可去掉 `--move-hands` 或使用后续 `--no-robot-feedback` 进行采集。

### 6.2 一键完成两次 session 采集、epoch 切分和模型训练

推荐每名被试采集两个 session，每个 session `--nrep 15`，即左右手共 30 个 trial。

```powershell
python demos\rehab_mi\run_offline_pipeline.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --nrep 15 `
  --algorithm secnet
```

如果采集时机械手接触不稳定，为避免机械手断连影响 EEG 数据采集，可使用：

```powershell
python demos\rehab_mi\run_offline_pipeline.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --nrep 15 `
  --algorithm secnet `
  --no-robot-feedback
```

该脚本会依次完成：

1. 启动 session 1 数据记录。
2. 启动 session 1 刺激范式和 marker 发送。
3. 保存 session 1 的 `recording.npz`、`markers.csv`、`meta.json`。
4. 等待后启动 session 2。
5. 自动切分两个 session 的 epoch。
6. 启动指定算法训练并保存模型。

### 6.3 单步采集命令

如果不使用一键脚本，也可以手动执行。

终端 1：启动记录。

```powershell
python demos\rehab_mi\collect_dataset.py `
  --subject sub17 `
  --session formal01 `
  --srate 250 `
  --num-chans 17
```

终端 2：启动刺激和 marker。

```powershell
python demos\rehab_mi\rehab_stim_demo.py `
  --direct `
  --nrep 15 `
  --lsl-markers `
  --lsl-source-id rehab_mi_marker_stream `
  --feedback-mode target `
  --robot-feedback `
  --left-com COM4 `
  --right-com COM3
```

采集第二个 session 时，将 `formal01` 改为 `formal02` 后重复上述步骤。

### 6.4 手动切 epoch 和训练

```powershell
python demos\rehab_mi\epoch_subject_sessions.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --overwrite
```

```powershell
python demos\rehab_mi\train_model.py `
  --subject sub17 `
  --sessions formal01 formal02 `
  --algorithm secnet `
  --out my_data\rehab_mi_models\sub17\sub17_all_sessions_secnet.pkl
```

### 6.5 真实在线 VR + EEG 预测 + 机械手闭环

在线前建议再次预检：

```powershell
python demos\rehab_mi\preflight_check.py `
  --mode online `
  --num-chans 17 `
  --left-com COM4 `
  --right-com COM3 `
  --move-hands
```

启动在线闭环：

```powershell
python demos\rehab_mi\run_online_demo.py `
  --vr `
  --model my_data\rehab_mi_models\sub17\sub17_all_sessions_secnet.pkl `
  --control-source prediction `
  --robot-mode serial `
  --robot-side both `
  --left-com COM4 `
  --right-com COM3 `
  --num-chans 17 `
  --eeg-chans 16 `
  --max-trials 10 `
  --nrep 5
```

期望结果：

- 电脑端刺激界面实时运行。
- VR/MR 网页同步显示 PROMPT、MOTOR IMAGERY、FEEDBACK、REST。
- 在线程序在每个 trial 中截取 1-4 s EEG 数据，调用模型输出左右手预测结果。
- 预测结果发送至对应机械手；终端保存在线预测日志。

## 7. 算法说明

训练入口统一为：

```powershell
python demos\rehab_mi\train_model.py --subject <subxx> --sessions formal01 formal02 --algorithm <算法名>
```

当前支持的算法包括：

| 命令参数 | 类型 | 说明 |
|---|---|---|
| `eegnet` | 深度学习 | MetaBCI 中常用 EEGNet 结构，适合端到端 MI 分类 |
| `secnet` | 深度学习 | 新增 SECNet 算法，用于 MI 数据分类 |
| `fbmsnet` | 深度学习 | 新增 FBMSNet 算法，结合多尺度/滤波思想 |
| `ifnet` | 深度学习 | 新增 IFNet 算法 |
| `mfanet` | 深度学习 | 新增 MFANet 算法 |
| `fbcspsvm` | 机器学习 | FBCSP 特征 + SVM 分类 |
| `fbcspsvmrm` | 机器学习 | FBCSP + SVM + 黎曼空间相关特征 |

所有模型保存为 bundle，用于保证离线训练和在线预测使用一致的采样率、通道数、时间窗、类别映射和预处理参数。

## 8. 数据集说明

仓库中包含用于赛事测试的 `sub17` 数据，路径如下：

```text
my_data/
  rehab_mi_software_marker/
    sub17/
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
  rehab_mi_models/
    sub17/
      sub17_all_sessions_secnet.pkl
      ...
```

数据格式：

- 任务类型：运动想象 MI。
- 类别数：2 类，`left_hand` 与 `right_hand`。
- 通道数：16 EEG。
- 采样率：250 Hz。
- 使用时间窗：1-4 s，共 3 秒。
- 测试数据：sub17 两个 session，约 60 个 trial。
- 本地测试指标：`secnet` 在该验证数据上的分类正确率记录约为 75%。

若评测人员只需验证基础性能指标，可直接运行第 5.1 和 5.2 节命令，无需重新采集数据。



## 10. 常见问题

### 10.1 没有 VR 眼镜怎么办？

使用浏览器打开 `http://127.0.0.1:8766` 即可验证 VR/MR 网页场景。真实 VR 眼镜只是在同一局域网下访问电脑 IP，例如：

```text
http://<电脑局域网IP>:8766
```

### 10.2 没有机械手怎么办？

在线命令使用：

```powershell
--robot-mode sim
```

系统不会打开 COM 口，只会在终端输出模拟反馈。

### 10.3 没有 Neuracle 脑电帽怎么办？

不要运行真实采集和 `--control-source prediction` 的硬件在线命令。使用第 5 节的离线数据训练和 `--control-source target --robot-mode sim` 测试流程。

### 10.4 VR 页面打不开怎么办？

检查：

1. `vr_scene_server.py` 是否正在运行。
2. 电脑和 VR 是否在同一局域网。
3. Windows 防火墙是否允许 Python 访问局域网。
4. 端口 `8766` 是否被其他程序占用。

### 10.5 机械手串口被占用怎么办？

关闭其他占用 COM3/COM4 的程序，重新插拔 USB 转串口设备。若只想采集 EEG，可使用 `--no-robot-feedback`。

### 10.6 Neuracle DataService 断开怎么办？

先确认 Neuracle Recorder 已开启 DataService，并且端口为 `127.0.0.1:8712`。同一时间只允许一个程序占用 DataService 数据流。

## 11. 推荐评测顺序

若评测人员没有硬件，建议按以下顺序：

1. 安装环境并检查 import。
2. 运行 `epoch_subject_sessions.py` 切分 sub17。
3. 运行 `train_model.py --algorithm secnet` 验证 3 s MI 分类指标。
4. 运行 `vr_scene_server.py` 和 `rehab_stim_demo.py` 验证 VR/MR 场景同步。
5. 运行 `run_online_demo.py --robot-mode sim --control-source target` 验证在线无硬件闭环入口。

若评测人员具备硬件，则在上述基础上增加：

1. 运行 `preflight_check.py --move-hands`。
2. 运行 `run_offline_pipeline.py` 重新采集并训练。
3. 运行 `run_online_demo.py --control-source prediction --robot-mode serial` 完成真实在线闭环控制。
