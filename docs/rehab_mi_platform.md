# Rehabilitation MI Platform Integration

## Architecture

The rehabilitation application follows the same ownership rule as the rest of
MetaBCI:

```text
demos = configuration and complete runnable examples
metabci = reusable datasets, paradigms, algorithms, acquisition, and devices
```

### Brainda

- `metabci/brainda/datasets/rehab_mi.py`
  implements `RehabMIDataset(BaseDataset)`.
- `metabci/brainda/algorithms/decomposition/fbcspsvmrm.py`
  implements an sklearn-compatible FBCSP, SVM, and Riemann feature model.
- `metabci/brainda/algorithms/deep_learning/fbmsnet.py`
  implements FBMSNet.
- `metabci/brainda/algorithms/rehab.py`
  owns preprocessing, estimator selection, model bundles, save/load, and
  online restoration.

`epoch_neuracle_recording.py` now loads recordings through
`RehabMIDataset` and cuts epochs through `MotorImagery`, so the formal dataset
uses MetaBCI hooks and metadata conventions.

### Brainstim

- `metabci/brainstim/rehab_mi.py`
  defines the shared rehabilitation state model, LSL publisher, PsychoPy
  adapter, and VR event renderer.
- `metabci/brainstim/vr.py`
  provides the HTTP, SSE, UDP, and asset-serving VR backend.

The state order is:

```text
START -> READY -> TRIAL -> PROMPT -> MOTOR_IMAGERY
      -> FEEDBACK -> REST -> STOP
```

`rehab_stim_demo.py` keeps the existing polished PsychoPy presentation while
using the platform trial generator, LSL publisher, VR sender, and Experiment.

### Brainflow

- `metabci/brainflow/neuracle.py`
  provides exact TCP packet reads, timestamps, disconnection reporting,
  LSL marker synchronization, marker deduplication, recording, and rolling
  buffers.
- `metabci/brainflow/rehab.py`
  provides `RehabMIPredictionWorker(ProcessWorker)`.
- `metabci/brainflow/feedback.py`
  provides simulation, asynchronous serial robot hands, VR feedback, and a
  safety-gated FES interface.

The formal online path is:

```text
NeuracleDataService(BaseAmplifier)
    -> Marker([1.0, 4.0], events=[1, 2])
    -> RehabMIPredictionWorker(ProcessWorker)
    -> ClosedLoopFeedback
    -> VR + left/right robot hand + optional FES
```

## Compatibility

The old script names remain available. Their reusable implementations now come
from `metabci`:

- `fbcspsvmrm.py`, `fbmsnet.py`, `mi_models.py`, and `device_adapters.py`
  are compatibility wrappers.
- `vr_scene_server.py` delegates to `metabci.brainstim.vr`.
- collection, epoching, training, and online scripts assemble platform APIs.

## Model bundle

Every formal model records:

- algorithm and estimator/parameter file
- fitted preprocessing values
- channel order and count
- sample rate and epoch time window
- class names
- subject and included sessions

Online preprocessing therefore reuses fitted training statistics rather than
estimating them again.

## Scoring evidence

- Brainda: dataset interface, MotorImagery, hooks, algorithms, model selection.
- Brainstim: Experiment, formal RehabMI phases, LSL markers, PsychoPy and VR.
- Brainflow: BaseAmplifier, Marker, ProcessWorker, Neuracle, feedback devices.
- New feature: browser VR rehabilitation and EEG-to-robot closed loop.
- New dataset: multi-subject and multi-session BaseDataset implementation.
- New algorithm/device: FBCSPSVMRM, FBMSNet, robot hand and FES interfaces.
- Improvements: exact TCP reads, marker deduplication, unified event ownership,
  asynchronous robot actions, and one-command startup.

## Remaining hardware acceptance

These checks require the actual equipment and are intentionally not simulated:

1. Ten-minute uninterrupted Neuracle DataService acquisition.
2. Two sessions automatically stopping and saving on the STOP marker.
3. Trial count equal to marker count.
4. EEGNet model loading in the online worker.
5. Ten online trials producing prediction logs.
6. Correct COM4/COM3 robot movement.
7. VR and PsychoPy phase synchronization.
8. Clean release of ports 8712, 8765, 8766, COM3, and COM4.

## Upstream synchronization

This working copy was prepared from the locally available full MetaBCI source
because the official GitHub repository could not be cloned from the current
network. It is versioned as 0.2.0-compatible and has an `upstream` remote
pointing to `TBC-TJU/MetaBCI`.

Before final submission, fetch the official `master`, compare this branch
against it, resolve any upstream API differences, and rerun the automated and
hardware acceptance checks.
