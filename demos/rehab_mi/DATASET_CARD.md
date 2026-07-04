# Rehabilitation MI Dataset Card

## Scope

The dataset contains left- and right-hand motor-imagery trials collected with
the MetaBCI Brainstim rehabilitation paradigm and a Neuracle EEG system.

## Required release fields

- anonymous subject identifier, for example `sub01`
- session identifier, for example `formal01`
- exact EEG channel order
- sample rate
- event map: left hand `1`, right hand `2`
- task window and all phase durations
- hardware and software versions
- number of valid and rejected trials
- consent, ethics, anonymization, and dataset license statement

## Recommended minimum scale

- at least five subjects
- two sessions per subject
- at least 30 valid trials per class in each session
- at least 600 valid trials in total

## Directory layout

```text
rehab_mi_dataset/
  sub01/
    formal01/
      raw.fif
      recording.npz
      markers.csv
      meta.json
    formal02/
  sub02/
```

`RehabMIDataset` can load the NPZ source directly and prefers `raw.fif` after
export. Formal evaluation splits must keep test sessions or test subjects out
of training.

## Release status

The repository contains the loader and conversion workflow. Do not publish
human recordings until consent, anonymization, ethics requirements, and the
data license have been reviewed by the project supervisor.
