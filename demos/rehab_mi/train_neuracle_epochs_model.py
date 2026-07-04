import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from metabci.brainda.algorithms.deep_learning.fbmsnet import (
    fit_fbmsnet_preprocessor,
)
from metabci.brainda.algorithms.rehab import (
    STANDARD_DEEP_ALGORITHMS,
    build_rehab_mi_estimator,
    create_model_bundle,
    fit_eegnet_preprocessor,
    save_model_bundle,
)
from rehab_config import data_root, load_config, model_root, selected_model_path

DEEP_ALGORITHMS = (*STANDARD_DEEP_ALGORITHMS, "fbmsnet")
STANDARD_PREPROCESS_DEEP_ALGORITHMS = STANDARD_DEEP_ALGORITHMS
FEATURE_NAMES = {
    "eegnet": "eegnet",
    "fbmsnet": "filterbank_multiscale_network",
    "secnet": "second_order_secnet",
    "eegconformer": "eeg_conformer_transformer",
    "ifnet": "inter_frequency_network",
    "mfanet": "multi_frequency_attention_network",
    "fbcspsvm": "filterbank_csp_svm",
    "fbcspsvmrm": "filterbank_csp_riemann_tangent_space",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple MI model from Neuracle epochs.npz.")
    parser.add_argument(
        "epoch_files",
        nargs="*",
        help="Optional epochs.npz paths. Without paths, all configured subject sessions are merged.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="Optional session names to include. Default: every session containing epochs.npz.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--algorithm",
        choices=[
            "eegnet",
            "fbmsnet",
            "secnet",
            "eegconformer",
            "ifnet",
            "mfanet",
            "fbcspsvm",
            "fbcspsvmrm",
            "centroid",
            "svc",
        ],
        default=None,
        help="Training algorithm. Default: eegnet.",
    )
    parser.add_argument(
        "--classifier",
        choices=["centroid", "svc"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--epochs",
        dest="training_epochs",
        type=int,
        default=150,
        help="Maximum EEGNet training epochs. Use 1 for an interface smoke test.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability for deep learning algorithms. Default: 0.5.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio for deep learning models. Default 0.2 means 80/20 train/validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the stratified train/validation split.",
    )
    parser.add_argument("--band-low", type=float, default=8.0)
    parser.add_argument("--band-high", type=float, default=30.0)
    return parser.parse_args()


def leave_one_out_accuracy(estimator, X, y):
    if len(y) < 4 or min(np.bincount(y)) < 2:
        return None, []

    preds = []
    for test_idx in range(len(y)):
        train_idx = np.setdiff1d(np.arange(len(y)), [test_idx])
        if len(np.unique(y[train_idx])) < 2:
            continue
        estimator.fit(X[train_idx], y[train_idx])
        preds.append((test_idx, int(estimator.predict(X[test_idx : test_idx + 1])[0])))

    if not preds:
        return None, []
    correct = [pred == int(y[idx]) for idx, pred in preds]
    return float(np.mean(correct)), preds


def stratified_holdout_accuracy(estimator, X, y, val_ratio=0.2, random_state=42):
    labels, counts = np.unique(y, return_counts=True)
    if len(y) < 4 or len(labels) < 2 or np.min(counts) < 2:
        return None, None

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(val_ratio),
        random_state=int(random_state),
    )
    train_idx, val_idx = next(splitter.split(X, y))
    eval_estimator = clone(estimator)
    eval_estimator.fit(X[train_idx], y[train_idx])
    pred = eval_estimator.predict(X[val_idx])
    return float(np.mean(pred == y[val_idx])), {
        "train_trials": int(len(train_idx)),
        "val_trials": int(len(val_idx)),
        "val_ratio": float(val_ratio),
        "random_state": int(random_state),
    }


def build_estimator(
    algorithm,
    X,
    y,
    band_low,
    band_high,
    srate,
    epochs=100,
    early_stopping_patience=80,
    batch_size=16,
    learning_rate=1e-3,
    val_ratio=0.2,
    random_state=42,
    dropout=0.5,
):
    return build_rehab_mi_estimator(
        algorithm=algorithm,
        X=X,
        y=y,
        srate=srate,
        band_low=band_low,
        band_high=band_high,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        random_state=random_state,
        dropout=dropout,
    )


def discover_epoch_paths(config, subject, sessions=None):
    subject = str(subject).strip()
    sessions = [str(session).strip() for session in sessions if str(session).strip()] if sessions else None
    subject_dir = data_root(config) / subject
    if sessions:
        paths = [subject_dir / session / "epochs.npz" for session in sessions]
    else:
        paths = sorted(subject_dir.glob("*/epochs.npz"))
    missing = [path for path in paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing epochs files:\n{missing_text}")
    if not paths:
        raise FileNotFoundError(f"No epochs.npz files found under {subject_dir}")
    return paths


def load_epoch_sets(paths):
    sets = []
    reference = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            eeg_chans = int(data["eeg_chans"])
            current = {
                "path": Path(path).resolve(),
                "X": data["X"].copy(),
                "y": data["y"].astype(np.int64).copy(),
                "srate": float(data["srate"]),
                "tmin": float(data["tmin"]),
                "tmax": float(data["tmax"]),
                "eeg_chans": eeg_chans,
                "channels": (
                    data["channels"].astype(str).tolist()
                    if "channels" in data
                    else [
                        f"EEG{index + 1:02d}"
                        for index in range(eeg_chans)
                    ]
                ),
            }
        signature = (
            current["srate"],
            current["tmin"],
            current["tmax"],
            current["eeg_chans"],
            tuple(current["channels"]),
            current["X"].shape[1:],
        )
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(
                f"Incompatible epoch settings in {path}. "
                "All sessions must use the same srate, tmin/tmax, channel count and sample length."
            )
        sets.append(current)
    return sets


def main():
    args = parse_args()
    config = load_config(args.config)
    subject = str(args.subject or config["subject"]).strip()
    sessions = args.sessions if args.sessions is not None else config.get("training_sessions")
    sessions = [str(session).strip() for session in sessions if str(session).strip()] if sessions else sessions
    epoch_paths = (
        [Path(path) for path in args.epoch_files]
        if args.epoch_files
        else discover_epoch_paths(config, subject, sessions)
    )
    epoch_sets = load_epoch_sets(epoch_paths)
    X = np.concatenate([item["X"] for item in epoch_sets], axis=0)
    y = np.concatenate([item["y"] for item in epoch_sets], axis=0)
    srate = epoch_sets[0]["srate"]
    tmin = epoch_sets[0]["tmin"]
    tmax = epoch_sets[0]["tmax"]
    eeg_chans = epoch_sets[0]["eeg_chans"]
    channels = epoch_sets[0]["channels"]
    algorithm = args.algorithm or args.classifier or "eegnet"
    preprocessing = None

    if algorithm in STANDARD_PREPROCESS_DEEP_ALGORITHMS:
        X, preprocessing = fit_eegnet_preprocessor(
            X,
            srate=srate,
            low=args.band_low,
            high=args.band_high,
        )
    elif algorithm == "fbmsnet":
        X, preprocessing = fit_fbmsnet_preprocessor(X, srate=srate)

    estimator = build_estimator(
        algorithm=algorithm,
        X=X,
        y=y,
        band_low=args.band_low,
        band_high=args.band_high,
        srate=srate,
        epochs=args.training_epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_ratio=args.val_ratio,
        random_state=args.random_state,
        dropout=args.dropout,
    )

    counts = {int(label): int(np.sum(y == label)) for label in sorted(set(y))}
    print(f"Config: {config['_config_path']}")
    print(f"Subject: {subject}")
    print("Included sessions:")
    for path in epoch_paths:
        print(f"  {Path(path).parent.name}: {path}")
    print(f"Loaded X={X.shape}, y={y.shape}, class_counts={counts}")
    print(f"Window: {tmin:.2f}-{tmax:.2f}s, srate={srate:.1f}, eeg_chans={eeg_chans}")
    print(f"Algorithm: {algorithm}")
    if algorithm in DEEP_ALGORITHMS:
        print(
            f"{algorithm.upper()} training: epochs={args.training_epochs}, "
            f"patience={args.early_stopping_patience}, "
            f"batch_size={min(args.batch_size, len(X))}, lr={args.learning_rate}, "
            f"dropout={args.dropout}, "
            f"train/val={1.0 - args.val_ratio:.1f}/{args.val_ratio:.1f}, "
            f"random_state={args.random_state}"
        )
        if algorithm in STANDARD_PREPROCESS_DEEP_ALGORITHMS:
            print(
                f"{algorithm.upper()} preprocessing: {args.band_low:.1f}-{args.band_high:.1f} Hz "
                "+ channel standardization"
            )
        else:
            print("FBMSNet preprocessing: 5-band MI filter bank + channel standardization")

    holdout_acc = None
    holdout_detail = None

    if algorithm in DEEP_ALGORITHMS:
        loo_acc, loo_preds = None, []
        print(
            f"Leave-one-out accuracy skipped for {algorithm.upper()}; "
            "use the validation metrics printed each epoch."
        )
    elif algorithm == "fbcspsvm":
        loo_acc, loo_preds = None, []
        holdout_acc, holdout_detail = stratified_holdout_accuracy(
            estimator,
            X,
            y,
            val_ratio=args.val_ratio,
            random_state=args.random_state,
        )
        if holdout_acc is None:
            print("Stratified holdout accuracy skipped: need at least 2 trials per class.")
        else:
            print(
                "Stratified holdout accuracy: "
                f"{holdout_acc:.4f} "
                f"({holdout_detail['train_trials']} train / "
                f"{holdout_detail['val_trials']} val)"
            )
    elif algorithm == "fbcspsvmrm":
        loo_acc, loo_preds = None, []
        print(
            f"Leave-one-out accuracy skipped for {algorithm.upper()}; "
            "fitting the final model once."
        )
    else:
        loo_acc, loo_preds = leave_one_out_accuracy(estimator, X, y)
        if loo_acc is None:
            print("Leave-one-out accuracy skipped: need at least 2 trials per class.")
        else:
            print(f"Leave-one-out accuracy: {loo_acc:.4f}")

    estimator.fit(X, y)
    bundle = create_model_bundle(
        algorithm=algorithm,
        estimator=None if algorithm in DEEP_ALGORITHMS else estimator,
        preprocessing=preprocessing,
        channels=channels,
        srate=srate,
        tmin=tmin,
        tmax=tmax,
        subject=subject,
        sessions=[item["path"].parent.name for item in epoch_sets],
        source_epochs=[str(item["path"]) for item in epoch_sets],
        classifier=algorithm,
        feature=FEATURE_NAMES.get(algorithm, "bandpass_logvar"),
        training_epochs=(
            args.training_epochs if algorithm in DEEP_ALGORITHMS else None
        ),
        batch_size=args.batch_size if algorithm in DEEP_ALGORITHMS else None,
        learning_rate=(
            args.learning_rate if algorithm in DEEP_ALGORITHMS else None
        ),
        dropout=args.dropout if algorithm in DEEP_ALGORITHMS else None,
        early_stopping_patience=(
            args.early_stopping_patience
            if algorithm in DEEP_ALGORITHMS
            else None
        ),
        val_ratio=args.val_ratio if algorithm in DEEP_ALGORITHMS else None,
        random_state=(
            args.random_state if algorithm in DEEP_ALGORITHMS else None
        ),
        labels={"1": "left_hand", "2": "right_hand"},
        band_low=args.band_low,
        band_high=args.band_high,
        loo_acc=loo_acc,
        loo_predictions=loo_preds,
        holdout_acc=holdout_acc,
        holdout_detail=holdout_detail,
        class_counts=counts,
    )

    if args.out:
        out_path = Path(args.out)
    elif config.get("selected_model") and algorithm in Path(config["selected_model"]).stem:
        out_path = selected_model_path(config, subject)
    else:
        out_path = model_root(config) / subject / f"{subject}_all_sessions_{algorithm}.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if algorithm in DEEP_ALGORITHMS:
        params_path = out_path.with_suffix(".params.pt")
        estimator.save_params(f_params=str(params_path))
        bundle["params_file"] = params_path.name
        print(f"Saved {algorithm.upper()} weights: {params_path}")

    save_model_bundle(bundle, out_path)
    print(f"Saved model: {out_path}")

    metadata = {
        key: value
        for key, value in bundle.items()
        if key not in ("estimator", "loo_predictions")
    }
    metadata_path = out_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"Saved model metadata: {metadata_path}")


if __name__ == "__main__":
    main()
