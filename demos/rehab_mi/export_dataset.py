"""Export collected NPZ sessions as MetaBCI/MNE FIF dataset runs."""

import argparse

from metabci.brainda.datasets import RehabMIDataset

from rehab_config import data_root, load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    subject = args.subject or config["subject"]
    root = data_root(config)
    dataset = RehabMIDataset(
        root=root,
        subjects=[subject],
        channels=config.get("channels"),
        srate=config.get("srate", 250),
        interval=(config.get("tmin", 1.0), config.get("tmax", 4.0)),
    )
    sessions = args.sessions or [
        run[0].parent.name
        for run in dataset.data_path(subject)
    ]
    for session in sessions:
        destination = dataset.export_session_to_fif(
            subject,
            session,
            overwrite=args.overwrite,
        )
        print(f"Exported: {destination}")


if __name__ == "__main__":
    main()
