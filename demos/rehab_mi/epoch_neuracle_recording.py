import argparse
import json
from pathlib import Path

import numpy as np

from metabci.brainda.datasets import RehabMIDataset
from metabci.brainda.paradigms import MotorImagery

from rehab_config import load_config, session_dir


LABEL_MAP = {1: "left_hand", 2: "right_hand"}
Y_MAP = {1: 0, 2: 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut Neuracle + LSL software-marker recording into MI epochs."
    )
    parser.add_argument(
        "record_dir",
        nargs="?",
        default=None,
        help="Directory containing recording.npz and meta.json. Defaults to the configured subject/session.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--srate", type=float, default=None)
    parser.add_argument("--eeg-chans", type=int, default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument(
        "--allow-short-last",
        action="store_true",
        help="Keep epochs only when exact full window is available by default.",
    )
    return parser.parse_args()


def load_meta(record_dir):
    meta_path = record_dir / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cut_epochs(eeg, timestamps, marker_values, marker_timestamps, srate, eeg_chans, tmin, tmax, allow_short_last):
    expected_samples = int(round((tmax - tmin) * srate))
    X = []
    y = []
    events = []
    kept_markers = []
    skipped = []

    for idx, (label_raw, marker_time) in enumerate(zip(marker_values, marker_timestamps), start=1):
        try:
            label = int(label_raw)
        except Exception:
            skipped.append((idx, label_raw, "non_trial_marker"))
            continue
        if label not in Y_MAP:
            skipped.append((idx, label, "unknown_label"))
            continue

        start_time = float(marker_time) + tmin
        start_idx = int(np.searchsorted(timestamps, start_time, side="left"))
        stop_idx = start_idx + expected_samples
        if stop_idx > len(eeg):
            if not allow_short_last:
                skipped.append((idx, label, "not_enough_data"))
                continue
            epoch = eeg[start_idx:, :eeg_chans]
            if len(epoch) == 0:
                skipped.append((idx, label, "empty_epoch"))
                continue
        else:
            epoch = eeg[start_idx:stop_idx, :eeg_chans]

        if len(epoch) != expected_samples and not allow_short_last:
            skipped.append((idx, label, f"bad_length_{len(epoch)}"))
            continue

        X.append(epoch.T.copy())
        y.append(Y_MAP[label])
        events.append(LABEL_MAP[label])
        kept_markers.append(
            {
                "marker_index": idx,
                "label": label,
                "event": LABEL_MAP[label],
                "marker_time": float(marker_time),
                "start_idx": int(start_idx),
                "stop_idx": int(start_idx + len(epoch)),
            }
        )

    if not X:
        raise RuntimeError("No epochs were cut. Check marker timestamps, channel count and tmin/tmax.")

    return np.stack(X), np.asarray(y, dtype=np.int64), np.asarray(events, dtype=object), kept_markers, skipped


def main():
    args = parse_args()
    config = load_config(args.config)
    record_dir = Path(args.record_dir) if args.record_dir else session_dir(config)
    recording_path = record_dir / "recording.npz"
    if not record_dir.is_dir() or not recording_path.is_file():
        raise FileNotFoundError(
            f"Recorded session does not exist: {record_dir}\n"
            f"Expected file: {recording_path}\n"
            "Collect this subject/session before cutting epochs."
        )
    args.eeg_chans = int(args.eeg_chans or config.get("eeg_chans", 16))
    args.tmin = float(args.tmin if args.tmin is not None else config.get("tmin", 1.0))
    args.tmax = float(args.tmax if args.tmax is not None else config.get("tmax", 4.0))
    print(f"Config: {config['_config_path']}")
    print(f"Record directory: {record_dir}")
    meta = load_meta(record_dir)
    srate = float(args.srate or meta.get("srate") or 250.0)

    subject = record_dir.parent.name
    dataset_root = record_dir.parents[1]
    channels = list(
        config.get("channels")
        or [f"EEG{index + 1:02d}" for index in range(args.eeg_chans)]
    )
    if len(channels) != args.eeg_chans:
        raise ValueError(
            f"Configured channels has {len(channels)} names, "
            f"but eeg_chans={args.eeg_chans}."
        )
    dataset = RehabMIDataset(
        root=dataset_root,
        subjects=[subject],
        channels=channels,
        srate=srate,
        interval=(args.tmin, args.tmax),
    )
    paradigm = MotorImagery(
        channels=channels,
        events=["left_hand", "right_hand"],
        intervals=[(args.tmin, args.tmax)],
    )
    X, y, epoch_meta = paradigm.get_data(
        dataset,
        subjects=[subject],
        return_concat=True,
        n_jobs=1,
    )
    session_mask = epoch_meta["session"].astype(str) == record_dir.name
    X = X[session_mask.to_numpy()]
    y = y[session_mask.to_numpy()].astype(np.int64)
    epoch_meta = epoch_meta.loc[session_mask].reset_index(drop=True)
    events = epoch_meta["event"].to_numpy(dtype=object)
    kept_markers = epoch_meta.to_dict(orient="records")
    skipped = []

    out_path = Path(args.out) if args.out else record_dir / "epochs.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        events=events,
        srate=np.asarray(srate),
        tmin=np.asarray(args.tmin),
        tmax=np.asarray(args.tmax),
        eeg_chans=np.asarray(args.eeg_chans),
        channels=np.asarray(channels, dtype=object),
        kept_markers=np.asarray(kept_markers, dtype=object),
        skipped=np.asarray(skipped, dtype=object),
    )

    counts = {name: int(np.sum(events == name)) for name in sorted(set(events))}
    print(f"Saved epochs: {out_path}")
    print(f"X shape: {X.shape}  y shape: {y.shape}")
    print(f"Class counts: {counts}")
    print(f"Skipped markers: {len(skipped)}")
    if skipped:
        print(f"Skipped detail: {skipped[:5]}")


if __name__ == "__main__":
    main()
