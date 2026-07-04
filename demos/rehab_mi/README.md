# MetaBCI Rehabilitation MI Closed Loop

This application is a thin demonstration layer over reusable MetaBCI APIs.

## Platform ownership

- `metabci.brainda.datasets.RehabMIDataset`
  loads subject/session recordings as MNE `Raw` objects.
- `metabci.brainda.algorithms`
  provides EEGNet, FBCSPSVMRM, FBMSNet, preprocessing, and model bundles.
- `metabci.brainstim.rehab_mi`
  defines the START/READY/TRIAL/PROMPT/MOTOR_IMAGERY/FEEDBACK/REST/STOP state model.
- `metabci.brainstim.vr`
  provides the browser VR scene server.
- `metabci.brainflow`
  provides Neuracle DataService, LSL marker alignment, online workers, and feedback devices.

The files in this directory only parse parameters and assemble those APIs.
Legacy file names are retained so existing commands continue to work.

## Dataset structure

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
```

Use anonymous subject identifiers. For the current scoring table, full dataset
credit requires at least 15 subjects. A practical target is two sessions per
subject and at least 30 valid trials per class in each session. If the dataset
has only 5 to 14 subjects, it should be documented as a partial-score dataset.
The formal evaluation window is `1.0-4.0 s` after the motor-imagery marker,
which gives exactly 3 seconds of EEG data at 250 Hz.

## Offline workflow

Edit `experiment_config.json`, then start collection:

Before formal collection, replace the placeholder `EEG01...EEG16` channel
names with the exact Recorder channel order. This order is stored in every
epoch file and model bundle and is validated again online.

```powershell
python demos\rehab_mi\collect_dataset.py
```

Start the Brainstim paradigm in another terminal:

```powershell
python demos\rehab_mi\rehab_stim_demo.py --direct --nrep 30 --lsl-markers --feedback-mode none --vr-events
```

Cut both sessions through `RehabMIDataset` and `MotorImagery`:

```powershell
python demos\rehab_mi\epoch_subject_sessions.py --subject sub01 --sessions formal01 formal02 --overwrite
```

This uses the `1.0-4.0 s` window from `experiment_config.json`. Re-run this
step after changing the window; models trained from older `0.5-4.5 s` epochs
should not be used for the 3-second performance claim.

Train a model:

```powershell
python demos\rehab_mi\train_model.py --subject sub01 --sessions formal01 formal02 --algorithm eegnet --epochs 150 --early-stopping-patience 40
```

Export portable MNE FIF files:

```powershell
python demos\rehab_mi\export_dataset.py --subject sub01 --sessions formal01 formal02 --overwrite
```

## Online workflow

```powershell
python demos\rehab_mi\run_online_demo.py --vr --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 10 --nrep 5
```

The launcher keeps the competition workflow at one command while the actual
data, stimulus, online processing, algorithms, and device feedback remain
inside MetaBCI modules.

## GitHub submission evidence

For competition review, include these files in the repository:

- `docs/scoring_mapping.md`
- `docs/project_test_report_template.md`
- `docs/dataset_card.md`
- `docs/github_submission_checklist.md`

Do not commit `.venv`, `my_data`, runtime logs, raw private EEG data, or large
trained models directly. Publish large anonymized datasets through Release
assets, Git LFS, or an external download link.
