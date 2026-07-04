import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect one Neuracle + LSL software-marker recording directory."
    )
    parser.add_argument(
        "record_dir",
        help="Directory containing recording.npz, markers.csv and meta.json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    record_dir = Path(args.record_dir)
    npz_path = record_dir / "recording.npz"
    csv_path = record_dir / "markers.csv"
    meta_path = record_dir / "meta.json"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    eeg = data["data"]
    timestamps = data["timestamps"]
    marker_values = data["marker_values"]
    marker_timestamps = data["marker_timestamps"]

    print(f"Directory: {record_dir}")
    print(f"EEG shape: {eeg.shape}")
    if len(timestamps) >= 2:
        duration = float(timestamps[-1] - timestamps[0])
        srate = (len(timestamps) - 1) / duration if duration > 0 else 0.0
        print(f"Duration: {duration:.2f} s")
        print(f"Estimated srate: {srate:.2f} Hz")
    print(f"Marker count in npz: {len(marker_values)}")

    if len(marker_values) > 0:
        preview_count = min(10, len(marker_values))
        print("First markers from npz:")
        for idx in range(preview_count):
            print(f"  {idx + 1}: label={marker_values[idx]}, time={marker_timestamps[idx]:.6f}")

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        print(f"Marker count in csv: {len(rows)}")
        for idx, row in enumerate(rows[:10], start=1):
            print(f"  csv {idx}: {row}")

    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"Meta subject/session: {meta.get('subject')} / {meta.get('session')}")
        print(f"Meta srate/channels: {meta.get('srate')} / {meta.get('num_chans')}")


if __name__ == "__main__":
    main()
