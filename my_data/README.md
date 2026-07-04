# Local Data Directory

This directory is intentionally kept out of normal Git tracking except for this
README and empty placeholders. It is the expected runtime location for
competition data and trained models.

## Expected Layout

```text
my_data/
  rehab_mi_software_marker/
    sub01/
      formal01/
        recording.npz
        markers.csv
        meta.json
        epochs.npz
      formal02/
        ...
  rehab_mi_models/
    sub01/
      sub01_all_sessions_eegnet.pkl
      sub01_all_sessions_eegnet.json
      sub01_all_sessions_eegnet.params.pt
```

## Submission Policy

Raw EEG recordings and trained subject-specific models are not committed by
default because they can be large and may contain private human-subject data.
For competition submission, publish the anonymized full dataset through one of
the following methods:

- GitHub Release assets,
- Git LFS,
- institutional storage,
- or a documented external download link.

If a small anonymized sample dataset is needed for code verification, add it
intentionally and document the subject count, event labels, channel order and
license in `docs/dataset_card.md`.
