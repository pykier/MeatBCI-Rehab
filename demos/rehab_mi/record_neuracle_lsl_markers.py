import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from metabci.brainflow.neuracle import (
    LSLMarkerBridge,
    NeuracleDataService,
    NeuracleRecorder,
)

from rehab_config import data_root, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record Neuracle Recorder TCP data with LSL software markers."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--srate", type=int, default=250)
    parser.add_argument(
        "--num-chans",
        type=int,
        default=17,
        help="Total channels sent by Recorder DataService, usually EEG channels plus trigger channel.",
    )
    parser.add_argument("--subject", default=None)
    parser.add_argument("--session", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing subject/session directory.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="0 means record until Ctrl+C.")
    parser.add_argument("--lsl-source-id", default="rehab_mi_marker_stream")
    parser.add_argument("--marker-timeout", type=float, default=8.0)
    return parser.parse_args()


MarkerRecorder = LSLMarkerBridge


def label_to_event(label):
    if label == 1:
        return "left_hand"
    if label == 2:
        return "right_hand"
    if str(label).lower() == "start":
        return "experiment_start"
    if str(label).lower() == "stop":
        return "experiment_stop"
    return "unknown"


def make_session_dir(args):
    session = args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / args.subject / session
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def record_data(args, marker_bridge):
    service = NeuracleDataService(
        device_address=(args.host, args.port),
        srate=args.srate,
        num_chans=args.num_chans,
        eeg_chans=min(args.num_chans - 1, 16),
    )
    recorder = NeuracleRecorder(service, marker_bridge=marker_bridge)
    print(f"Connecting to Neuracle Recorder DataService {args.host}:{args.port}")
    return recorder.record(duration=args.duration)


def get_crop_window(marker_rows):
    start_times = [row["lsl_time"] for row in marker_rows if row["event"] == "experiment_start"]
    if not start_times:
        return None, None
    start_time = float(start_times[0])
    stop_times = [
        row["lsl_time"]
        for row in marker_rows
        if row["event"] == "experiment_stop" and float(row["lsl_time"]) > start_time
    ]
    stop_time = float(stop_times[0]) if stop_times else None
    return start_time, stop_time


def crop_to_experiment(data, timestamps, marker_rows):
    start_time, stop_time = get_crop_window(marker_rows)
    if start_time is None:
        return data, timestamps, marker_rows, None, None

    keep = timestamps >= start_time
    if stop_time is not None:
        keep &= timestamps <= stop_time
    if not np.any(keep):
        print("Experiment crop window found, but no EEG samples were inside it. Keeping full recording.")
        return data, timestamps, marker_rows, start_time, stop_time

    cropped_data = data[keep]
    cropped_timestamps = timestamps[keep]
    cropped_markers = [
        row
        for row in marker_rows
        if float(row["lsl_time"]) >= start_time and (stop_time is None or float(row["lsl_time"]) <= stop_time)
    ]
    print(
        "Cropped recording to experiment window: "
        f"start={start_time:.6f}, stop={stop_time if stop_time is not None else 'None'}, "
        f"samples={len(cropped_timestamps)}"
    )
    return cropped_data, cropped_timestamps, cropped_markers, start_time, stop_time


def is_trial_marker(row):
    return row["event"] in ("left_hand", "right_hand")


def save_outputs(out_dir, args, data, timestamps, marker_rows):
    data, timestamps, marker_rows, crop_start, crop_stop = crop_to_experiment(data, timestamps, marker_rows)
    trial_marker_rows = [row for row in marker_rows if is_trial_marker(row)]
    marker_values = np.array([row["label"] for row in trial_marker_rows], dtype=object)
    marker_times = np.array([row["lsl_time"] for row in trial_marker_rows], dtype=float)

    np.savez_compressed(
        out_dir / "recording.npz",
        data=data,
        timestamps=timestamps,
        marker_values=marker_values,
        marker_timestamps=marker_times,
    )

    with (out_dir / "markers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lsl_time", "label", "event", "trial_id"],
        )
        writer.writeheader()
        writer.writerows(marker_rows)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject": args.subject,
        "session": args.session,
        "srate": args.srate,
        "num_chans": args.num_chans,
        "channels": getattr(args, "channel_names", None),
        "host": args.host,
        "port": args.port,
        "lsl_source_id": args.lsl_source_id,
        "data_shape": list(data.shape),
        "experiment_crop_start": crop_start,
        "experiment_crop_stop": crop_stop,
        "saved_files": ["recording.npz", "markers.csv", "meta.json"],
        "label_map": {"1": "left_hand", "2": "right_hand"},
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved data: {out_dir / 'recording.npz'}")
    print(f"Saved markers: {out_dir / 'markers.csv'}")
    print(f"Saved meta: {out_dir / 'meta.json'}")


def main():
    args = parse_args()
    config = load_config(args.config)
    args.subject = args.subject or config["subject"]
    args.session = args.session or config["session"]
    args.out_root = args.out_root or str(data_root(config))
    args.channel_names = config.get("channels")
    out_dir = make_session_dir(args)
    existing = [out_dir / name for name in ("recording.npz", "markers.csv", "meta.json")]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        existing_text = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            f"Session already contains recorded files:\n{existing_text}\n"
            "Change session in experiment_config.json or pass --overwrite intentionally."
        )
    print(f"Config: {config['_config_path']}")
    print(f"Subject/session: {args.subject}/{args.session}")
    print(f"Output directory: {out_dir}")

    markers = MarkerRecorder(
        args.lsl_source_id,
        resolve_timeout=args.marker_timeout,
        verbose=True,
    )
    data, timestamps = record_data(args, markers)
    save_outputs(out_dir, args, data, timestamps, markers.rows)


if __name__ == "__main__":
    main()
