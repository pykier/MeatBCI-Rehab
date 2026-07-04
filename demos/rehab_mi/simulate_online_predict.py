import argparse
from pathlib import Path

import joblib
import numpy as np

from bnci_gdf_dataset import load_bnci_gdf_trials, select_balanced_indices
from device_adapters import LABEL_NAMES


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Simulate online single-trial MI prediction.")
    parser.add_argument("--data-dir", default=str(root / "BNCI2014004"))
    parser.add_argument("--model", default=str(Path(__file__).resolve().parent / "outputs" / "model.pkl"))
    parser.add_argument("--subjects", nargs="*", type=int, default=[1])
    parser.add_argument("--limit", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    estimator = bundle["estimator"]

    _, X, y, meta = load_bnci_gdf_trials(
        args.data_dir,
        subjects=args.subjects,
        window_s=bundle["window_s"],
    )
    trial_indices = select_balanced_indices(y, args.limit)
    pred = estimator.predict(X[trial_indices])
    true = y[trial_indices]
    correct = pred == true

    for i, (trial_index, true_label, pred_label, ok) in enumerate(
        zip(trial_indices, true, pred, correct), 1
    ):
        subject = meta.iloc[trial_index]["subject"]
        event = meta.iloc[trial_index]["event"]
        print(
            f"trial={i:02d} subject={subject} event={event} "
            f"true={LABEL_NAMES[int(true_label)]} pred={LABEL_NAMES[int(pred_label)]} "
            f"correct={bool(ok)}"
        )

    print(f"Simulated online accuracy: {np.mean(correct):.4f}")


if __name__ == "__main__":
    main()
