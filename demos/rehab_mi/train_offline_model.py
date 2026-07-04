import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from metabci.brainda.algorithms.decomposition import FBCSP
from metabci.brainda.algorithms.utils.model_selection import (
    generate_kfold_indices,
    match_kfold_indices,
    set_random_seeds,
)

from bnci_gdf_dataset import load_bnci_gdf_trials, make_filterbank
from mi_models import LogVariance, NearestCentroidMI


def build_estimator(classifier_name, filterbank, feature_name):
    if classifier_name == "lda":
        classifier = LinearDiscriminantAnalysis()
    elif classifier_name == "svc":
        classifier = SVC(probability=True)
    elif classifier_name == "centroid":
        classifier = NearestCentroidMI()
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    if feature_name == "logvar":
        return make_pipeline(LogVariance(), classifier)
    if feature_name == "fbcsp":
        return make_pipeline(
            FBCSP(
                n_components=2,
                n_mutualinfo_components=6,
                filterbank=filterbank,
            ),
            classifier,
        )
    raise ValueError(f"Unsupported feature extractor: {feature_name}")


def cross_validate(estimator, X, y, meta, kfold):
    indices = generate_kfold_indices(meta, kfold=kfold, random_state=38)
    accs = []

    for k in range(kfold):
        train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
        train_ind = np.concatenate((train_ind, validate_ind))
        pred = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
        acc = np.mean(pred == y[test_ind])
        accs.append(acc)
        print(f"Fold {k + 1}: {acc:.4f}")

    print(f"Mean accuracy: {np.mean(accs):.4f}")
    return accs


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train BNCI2014004 MI model.")
    parser.add_argument("--data-dir", default=str(root / "BNCI2014004"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs" / "model.pkl"))
    parser.add_argument("--subjects", nargs="*", type=int, default=None)
    parser.add_argument("--classifier", choices=["centroid", "lda", "svc"], default="centroid")
    parser.add_argument("--feature", choices=["logvar", "fbcsp"], default="logvar")
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--kfold", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seeds(38)

    dataset, X, y, meta = load_bnci_gdf_trials(
        args.data_dir,
        subjects=args.subjects,
        window_s=args.window_s,
    )
    subjects = args.subjects or dataset.subjects
    print(f"Loaded X={X.shape}, y={y.shape}, subjects={subjects}")

    filterbank = make_filterbank(dataset.srate)
    estimator = build_estimator(args.classifier, filterbank, args.feature)
    accs = cross_validate(estimator, X, y, meta, args.kfold)

    estimator.fit(X, y)
    model_bundle = {
        "estimator": estimator,
        "classifier": args.classifier,
        "feature": args.feature,
        "subjects": subjects,
        "channels": dataset.channels,
        "events": {"left_hand": 0, "right_hand": 1},
        "event_codes": {"left_hand": 769, "right_hand": 770},
        "srate": dataset.srate,
        "window_s": args.window_s,
        "cv_accs": accs,
        "mean_acc": float(np.mean(accs)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, out_path)
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()
