import argparse
import subprocess
import sys
from pathlib import Path

from rehab_config import data_root, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut every recorded session for one subject using the shared configuration."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    subject = args.subject or config["subject"]
    subject_dir = data_root(config) / subject
    sessions = args.sessions if args.sessions is not None else config.get("training_sessions")

    if sessions:
        session_dirs = [subject_dir / session for session in sessions]
    else:
        session_dirs = sorted(
            path for path in subject_dir.iterdir() if path.is_dir() and (path / "recording.npz").exists()
        )

    if not session_dirs:
        raise FileNotFoundError(f"No recorded sessions found under {subject_dir}")

    missing_sessions = [
        session_dir
        for session_dir in session_dirs
        if not session_dir.is_dir()
        or not (session_dir / "recording.npz").is_file()
    ]
    if missing_sessions:
        missing_text = "\n".join(
            f"  {session_dir} (missing recording.npz)"
            for session_dir in missing_sessions
        )
        raise FileNotFoundError(
            "The following requested sessions have not been recorded:\n"
            f"{missing_text}\n"
            "Run collect_dataset.py and rehab_stim_demo.py for each session "
            "before cutting epochs."
        )

    script = Path(__file__).resolve().parent / "epoch_neuracle_recording.py"
    for session_dir in session_dirs:
        output = session_dir / "epochs.npz"
        if output.exists() and not args.overwrite:
            print(f"Skip existing epochs: {output}")
            continue
        command = [
            sys.executable,
            str(script),
            str(session_dir),
            "--config",
            config["_config_path"],
        ]
        print(f"Cutting session: {session_dir.name}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
