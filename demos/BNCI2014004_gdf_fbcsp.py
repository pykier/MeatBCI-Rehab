from pathlib import Path

import mne
import numpy as np
from mne.channels import make_standard_montage
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from metabci.brainda.algorithms.decomposition import FBCSP
from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.algorithms.utils.model_selection import (
    generate_kfold_indices,
    match_kfold_indices,
    set_random_seeds,
)
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.paradigms import MotorImagery


class LocalBNCI2014004GDF(BaseDataset):
    """Read local BNCI 2014-004 GDF training files.

    The upstream MetaBCI BNCI2014004 class reads downloaded .mat files.
    This local wrapper is for the raw .gdf files placed under demos/BNCI2014004.
    """

    _EVENTS = {
        "left_hand": (769, (0, 3)),
        "right_hand": (770, (0, 3)),
    }
    _CHANNELS = ["C3", "CZ", "C4"]

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        subjects = sorted(
            {
                int(path.name[1:3])
                for path in self.data_dir.glob("B????T.gdf")
            }
        )
        super().__init__(
            dataset_code="local_bnci2014004_gdf",
            subjects=subjects,
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=250,
            paradigm="imagery",
        )

    def data_path(
        self,
        subject,
        path=None,
        force_update=False,
        update_path=None,
        proxies=None,
        verbose=None,
    ):
        files = sorted(self.data_dir.glob(f"B{int(subject):02d}??T.gdf"))
        if not files:
            raise FileNotFoundError(
                f"No training GDF files found for subject {subject} in {self.data_dir}"
            )
        return [[file] for file in files]

    def _get_single_subject_data(self, subject, verbose=None):
        montage = make_standard_montage("standard_1005")
        runs = {}

        for run_id, run_file_group in enumerate(self.data_path(subject)):
            run_file = run_file_group[0]
            raw = mne.io.read_raw_gdf(run_file, preload=True, verbose=False)
            raw.rename_channels(
                {
                    "EEG:C3": "C3",
                    "EEG:Cz": "CZ",
                    "EEG:C4": "C4",
                    "EOG:ch01": "EOG1",
                    "EOG:ch02": "EOG2",
                    "EOG:ch03": "EOG3",
                }
            )
            raw.set_montage(montage, on_missing="ignore")
            runs[f"run_{run_id}"] = raw

        return {f"session_{subject}": runs}


def raw_hook(raw, caches):
    raw.filter(
        6,
        30,
        l_trans_bandwidth=2,
        h_trans_bandwidth=5,
        phase="zero-double",
        verbose=False,
    )
    return raw, caches


def main():
    data_dir = Path(__file__).resolve().parent / "BNCI2014004"
    dataset = LocalBNCI2014004GDF(data_dir)
    print(dataset)

    wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 32)]
    filterbank = generate_filterbank(wp, ws, srate=dataset.srate, order=4, rp=0.5)

    paradigm = MotorImagery(
        channels=["C3", "CZ", "C4"],
        events=["left_hand", "right_hand"],
        intervals=[(0, 3)],
        srate=250,
    )
    paradigm.register_raw_hook(raw_hook)

    subjects = dataset.subjects
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=None,
        verbose=False,
    )
    print(f"Loaded X={X.shape}, y={y.shape}, subjects={subjects}")

    set_random_seeds(38)
    kfold = 5
    indices = generate_kfold_indices(meta, kfold=kfold)
    estimator = make_pipeline(
        FBCSP(
            n_components=2,
            n_mutualinfo_components=6,
            filterbank=filterbank,
        ),
        SVC(),
    )

    accs = []
    for k in range(kfold):
        train_ind, validate_ind, test_ind = match_kfold_indices(k, meta, indices)
        train_ind = np.concatenate((train_ind, validate_ind))
        pred = estimator.fit(X[train_ind], y[train_ind]).predict(X[test_ind])
        acc = np.mean(pred == y[test_ind])
        accs.append(acc)
        print(f"Fold {k + 1}: {acc:.4f}")

    print(f"Mean accuracy: {np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
