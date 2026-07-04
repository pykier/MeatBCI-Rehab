# MetaBCI Competition Scoring Mapping

This document maps the rehabilitation MI closed-loop project to the objective
scoring table. It is intended to be submitted with the GitHub repository and
the project test report.

## Strict Self Assessment

| Category | Full-score requirement | Current evidence | Risk |
| --- | --- | --- | --- |
| brainda | Use at least 3 valid brainda functions | `RehabMIDataset`, `MotorImagery` epoching, EEGNet/FBMSNet/FBCSPSVM/FBCSPSVMRM, model bundle and validation | Low if commands and code paths are documented |
| brainstim | Use at least 3 valid brainstim functions | RehabMI state machine, PsychoPy stimulus, LSL marker publisher, VR/MR browser renderer | Low if VR is described as brainstim-compatible, not standalone Unity |
| brainflow | Use at least 3 valid brainflow functions | Neuracle DataService, software marker alignment, online worker, rolling buffer, robot/FES feedback | Low if online demo logs are provided |
| New large-scale functions | At least 2 effective new functions | VR/MR rehabilitation scene support; EEG-to-robot-hand closed-loop control | Low after hardware video/log evidence |
| New dataset | At least 15 subjects for full score, at least 5 for partial score | Platform data structure exists; actual subject count must be collected | High until 15 anonymized subjects are available |
| New paradigm/algorithm/device | At least 2 new paradigms, algorithms or devices | RehabMI paradigm, FBMSNet, FBCSPSVM, FBCSPSVMRM, robot-hand interface, FES interface | Low if import paths are documented |
| Existing function improvement | At least 1 validated improvement | Neuracle TCP robustness, marker deduplication, automatic stop/crop, one-command pipeline, robot fault tolerance | Low if tests are included |
| Usage degree | Fully based on MetaBCI for 15 points | Core dataset, algorithm, stimulus, online processing and feedback are in `metabci`; `demos` are launchers | Medium unless README clearly proves module ownership |

## Evidence By MetaBCI Module

### brainda

- `metabci/brainda/datasets/rehab_mi.py`
  - Adds `RehabMIDataset` for subject/session recordings.
  - Supports MetaBCI dataset loading and MNE `Raw` generation.
- `demos/rehab_mi/epoch_neuracle_recording.py`
  - Cuts MI epochs through `RehabMIDataset` and `MotorImagery`.
- `metabci/brainda/algorithms/rehab.py`
  - Owns preprocessing, algorithm selection, model bundles and online restore.
- `metabci/brainda/algorithms/deep_learning/fbmsnet.py`
  - Adds FBMSNet.
- `metabci/brainda/algorithms/decomposition/fbcspsvmrm.py`
  - Adds FBCSP + SVM + Riemann feature classifier.

### brainstim

- `metabci/brainstim/rehab_mi.py`
  - Defines START, READY, TRIAL, PROMPT, MOTOR_IMAGERY, FEEDBACK, REST and STOP.
  - Publishes the single authoritative LSL marker event per trial.
  - Sends the same phase event to PsychoPy and VR.
- `metabci/brainstim/vr.py`
  - Provides the browser-based VR/MR rehabilitation scene backend.
- `demos/rehab_mi/rehab_stim_demo.py`
  - Competition launcher for the MetaBCI stimulus and optional robot feedback.

### brainflow

- `metabci/brainflow/neuracle.py`
  - Adds robust Neuracle DataService reading, timestamps and recording.
  - Aligns LSL software markers with the EEG stream.
- `metabci/brainflow/rehab.py`
  - Adds `RehabMIPredictionWorker` based on MetaBCI online worker logic.
- `metabci/brainflow/feedback.py`
  - Adds serial robot-hand control, simulation mode, VR feedback and FES stub.
- `demos/rehab_mi/run_online_demo.py`
  - Assembles Neuracle stream, online prediction, VR scene and robot feedback.

## Commands To Demonstrate

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\preflight_check.py --mode offline --num-chans 17 --left-com COM4 --right-com COM3 --move-hands
.\.venv\Scripts\python.exe demos\rehab_mi\run_offline_pipeline.py
.\.venv\Scripts\python.exe demos\rehab_mi\run_online_demo.py --vr --model my_data\rehab_mi_models\sub04\sub04_all_sessions_eegnet.pkl --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

## Full-score blockers

1. The dataset score cannot be full unless the submitted dataset has at least
   15 anonymized subjects.
2. The usage score may be reduced if reviewers cannot see that `demos` are thin
   entry points over MetaBCI APIs.
3. The VR/MR feature must be described as a `brainstim` browser renderer, not as
   an independent Unity-only scene.
4. Real electric stimulation should not be claimed unless the safety parameters,
   device protocol and hardware tests are included.

