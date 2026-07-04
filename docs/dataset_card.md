# Rehabilitation MI Dataset Card

## Dataset Name

MetaBCI Rehab MI Dataset

## Purpose

The dataset supports left-hand and right-hand motor imagery decoding for a
VR/MR rehabilitation robot-hand closed-loop task.

## Required Full-score Scale

The competition scoring table requires at least 15 subjects for full dataset
credit. The recommended final target is:

- Subjects: at least 15
- Sessions per subject: 2
- Trials per session: at least 30 valid left-hand and 30 valid right-hand trials
- Sample rate: 250 Hz
- Channels: 16 EEG channels + 1 marker channel

## Directory Structure

```text
my_data/rehab_mi_software_marker/
  sub01/
    formal01/
      recording.npz
      markers.csv
      meta.json
      epochs.npz
      raw.fif
    formal02/
      ...
  sub02/
    formal01/
    formal02/
```

## Event Definitions

| Label | Meaning |
| --- | --- |
| 1 | left_hand |
| 2 | right_hand |

The MOTOR_IMAGERY phase is the authoritative marker boundary. Each trial should
produce exactly one marker.

## Files

- `recording.npz`: EEG samples and timestamps.
- `markers.csv`: LSL software markers aligned to the experiment timeline.
- `meta.json`: subject, session, sample rate, channel count and timing metadata.
- `epochs.npz`: model-ready MI epochs.
- `raw.fif`: optional exported MNE file for platform loading.

## Loading Through MetaBCI

The platform loader is `metabci.brainda.datasets.RehabMIDataset`. Epoching
scripts use this dataset interface and the MetaBCI MotorImagery paradigm rather
than reading private arrays directly.

Example:

```powershell
.\.venv\Scripts\python.exe demos\rehab_mi\epoch_subject_sessions.py --subject sub01 --sessions formal01 formal02 --overwrite
```

## Anonymization

- Use anonymous IDs such as `sub01`, `sub02`, ...
- Do not store names, student IDs, phone numbers or medical identifiers.
- Keep consent forms outside the public repository.
- If the full dataset cannot be public, provide a controlled access link and a
  small anonymized sample for code verification.

