# Project Test Report Template

Use this document as the source text for the official project test report.

## 1. Basic Information

- Project name: 念力搬砖 BCI: 基于 MetaBCI 的 VR 康复场景运动想象机械手控制系统
- Track: MetaBCI 创新应用开发赛项
- Team:
- Organization:
- Repository:
- Demo video:

## 2. Hardware And Environment

- PC:
- OS:
- Python:
- MetaBCI commit/tag:
- EEG device: Neuracle EEG cap, 16 EEG channels + 1 marker channel
- EEG sample rate: 250 Hz
- DataService endpoint: `127.0.0.1:8712`
- Left robot hand: `COM4`
- Right robot hand: `COM3`
- VR/MR headset:
- Network: PC and headset on the same LAN

## 3. Objective Scoring Self-check

| Scoring item | Claimed score | Evidence |
| --- | --- | --- |
| brainda | 10 | `RehabMIDataset`, `MotorImagery`, EEGNet/FBMSNet/FBCSPSVM/FBCSPSVMRM |
| brainstim | 10 | RehabMI state machine, PsychoPy stimulus, LSL marker, VR renderer |
| brainflow | 10 | Neuracle DataService, marker alignment, online worker, robot feedback |
| New large-scale functions | 20 | VR/MR rehabilitation scene and EEG-to-robot-hand closed loop |
| New dataset | 15 only if subjects >= 15 | Attach dataset subject/session/trial count table |
| New paradigm/algorithm/device | 15 | RehabMI paradigm, new algorithms, robot/FES/VR devices |
| Improvement | 5 | DataService robustness, marker deduplication, one-command workflow |
| Usage degree | 15 | Core functions implemented under `metabci`, demos are launchers |

## 4. Test Cases

### 4.1 Hardware Preflight

Command:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode offline --num-chans 17 --left-com COM4 --right-com COM3 --move-hands
```

Expected result:

- EEG samples are received.
- Both robot hands move once.
- Serial ports are released after the test.

Actual result:

- EEG:
- Robot:
- Screenshot/video:

### 4.2 Offline Collection

Command:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_offline_pipeline.py
```

Expected result:

- Two sessions are recorded.
- Each session saves `recording.npz`, `markers.csv`, `meta.json` and `epochs.npz`.
- Marker count equals the number of formal trials.

Actual result:

- Subject:
- Sessions:
- Trial count:
- Marker count:
- Saved path:

### 4.3 Epoching And Training

Expected result:

- Epoch shape:
- Class counts:
- Algorithm:
- Training/validation metric:
- Saved model:

Paste the training log here.

### 4.4 Online VR Closed Loop

Command:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_eegnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

Expected result:

- Brainstim window displays the same phases as the VR browser scene.
- Each MOTOR_IMAGERY trial produces one prediction.
- FEEDBACK displays actual instruction and predicted result.
- The predicted class drives the corresponding robot hand.
- The online log is saved.

Actual result:

- VR URL:
- Online trials:
- Prediction log path:
- Robot result:
- Video evidence:

### 4.5 Fault Tolerance

Test:

- Temporarily disconnect one robot hand or block a serial write.

Expected result:

- EEG acquisition and marker recording continue.
- The failed robot action is logged.
- The next trials continue running.

Actual result:

-

## 5. Dataset Description

- Number of subjects:
- Sessions per subject:
- Trials per session:
- Events: `left_hand=1`, `right_hand=2`
- Sample rate:
- EEG channels:
- Data root:
- Anonymization method:
- License or release condition:

For full dataset score, this section must show at least 15 subjects.

## 6. Known Limitations

- MI decoding accuracy depends on subject state and calibration quality.
- Real electric stimulation is disabled unless a safety-reviewed device protocol
  and stimulation parameters are configured.
- VR/MR is implemented as a browser renderer compatible with `brainstim`; it is
  not a Unity-only independent scene.

