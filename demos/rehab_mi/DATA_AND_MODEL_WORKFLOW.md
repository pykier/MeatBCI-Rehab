# Subject, Session, and Model Workflow

Edit `experiment_config.json` before recording or online use:

```json
{
  "subject": "sub01",
  "session": "session01",
  "data_root": "my_data/rehab_mi_software_marker",
  "model_root": "my_data/rehab_mi_models",
  "selected_model": "sub01_all_sessions_centroid.pkl",
  "srate": 250,
  "num_chans": 17,
  "eeg_chans": 16,
  "tmin": 1.0,
  "tmax": 4.0
}
```

The default formal MI decoding window is `1.0-4.0 s` after the
`MOTOR_IMAGERY` marker, i.e. exactly 3 seconds of EEG data.

The resulting directory structure is:

```text
my_data/
  rehab_mi_software_marker/
    sub01/
      session01/
        recording.npz
        markers.csv
        meta.json
        epochs.npz
      session02/
        ...
  rehab_mi_models/
    sub01/
      sub01_all_sessions_centroid.pkl
```

## Record the configured session

```powershell
python demos\rehab_mi\record_neuracle_lsl_markers.py
```

Start the stimulus in another terminal:

```powershell
python demos\rehab_mi\rehab_stim_demo.py --direct --nrep 40 --lsl-markers --lsl-source-id rehab_mi_marker_stream --feedback-mode none --vr-events
```

## Cut the configured session into epochs

```powershell
python demos\rehab_mi\epoch_neuracle_recording.py
```

To cut every recorded session for the configured subject with the same window:

```powershell
python demos\rehab_mi\epoch_subject_sessions.py --overwrite
```

The generated `epochs.npz` stores `tmin=1.0`, `tmax=4.0`, and 750 samples at
250 Hz. If existing epochs were produced with an older window, rerun the epoch
command with `--overwrite` before training.

## Train one subject from all available sessions

```powershell
python demos\rehab_mi\train_neuracle_epochs_model.py
```

To train only selected sessions:

```powershell
python demos\rehab_mi\train_neuracle_epochs_model.py --sessions session01 session03
```

## Run online with the selected model

```powershell
python demos\rehab_mi\run_online_demo.py --vr --control-source prediction --robot-mode serial --robot-side both --left-com COM4 --right-com COM3 --num-chans 17 --eeg-chans 16 --max-trials 6 --nrep 3
```

`run_online_demo.py` loads `selected_model` from `experiment_config.json`.
Passing `--model PATH` overrides the configured model for one run.
