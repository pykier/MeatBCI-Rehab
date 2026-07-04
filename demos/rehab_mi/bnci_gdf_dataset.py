from pathlib import Path

import mne
import numpy as np

from metabci.brainda.algorithms.decomposition.base import generate_filterbank
from metabci.brainda.datasets.base import BaseDataset
from metabci.brainda.paradigms import MotorImagery


class LocalBNCI2014004GDF(BaseDataset):
    """Local BNCI 2014-004 GDF wrapper for MetaBCI MotorImagery."""

    events_def = {
        "left_hand": (769, (0, 3)),
        "right_hand": (770, (0, 3)),
    }
    channels_def = ["C3", "CZ", "C4"]

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        subjects = sorted(
            {
                int(path.name[1:3])
                for path in self.data_dir.glob("B????T.gdf")
            }
        )
        if not subjects:
            raise FileNotFoundError(f"No BNCI2014004 training GDF files in {data_dir}")

        super().__init__(
            dataset_code="local_bnci2014004_gdf",
            subjects=subjects,
            events=self.events_def,
            channels=self.channels_def,
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
        runs = {}

        for run_id, run_file_group in enumerate(self.data_path(subject)):
            run_file = run_file_group[0]
            raw = read_raw_gdf_compat(run_file)
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
            runs[f"run_{run_id}"] = raw

        return {f"session_{subject}": runs}


def read_raw_gdf_compat(run_file):
    """Read GDF files across the MNE 1.3.x / NumPy 1.26 combination."""
    original_clip = np.clip

    def clip_compat(a, a_min, a_max, out=None, **kwargs):
        if (
            out is not None
            and np.issubdtype(out.dtype, np.integer)
            and np.isscalar(a_max)
            and np.isinf(a_max)
        ):
            clipped = original_clip(a, a_min, np.iinfo(out.dtype).max, **kwargs)
            out[...] = clipped.astype(out.dtype, copy=False)
            return out
        return original_clip(a, a_min, a_max, out=out, **kwargs)

    np.clip = clip_compat
    try:
        return mne.io.read_raw_gdf(run_file, preload=True, verbose=False)
    finally:
        np.clip = original_clip


def make_filterbank(srate):
    wp = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    ws = [(2, 10), (6, 14), (10, 18), (14, 22), (18, 26), (22, 32)]
    return generate_filterbank(wp, ws, srate=srate, order=4, rp=0.5)


def raw_bandpass_hook(raw, caches):
    raw.filter(
        6,
        30,
        l_trans_bandwidth=2,
        h_trans_bandwidth=5,
        phase="zero-double",
        verbose=False,
    )
    return raw, caches


def make_motor_imagery_paradigm(window_s=3.0, srate=250):
    paradigm = MotorImagery(
        channels=LocalBNCI2014004GDF.channels_def,
        events=["left_hand", "right_hand"],
        intervals=[(0, window_s)],
        srate=srate,
    )
    paradigm.register_raw_hook(raw_bandpass_hook)
    return paradigm


def load_bnci_gdf_trials(data_dir, subjects=None, window_s=3.0):
    dataset = LocalBNCI2014004GDF(data_dir)
    if subjects is None:
        subjects = dataset.subjects

    paradigm = make_motor_imagery_paradigm(window_s=window_s, srate=dataset.srate)
    X, y, meta = paradigm.get_data(
        dataset,
        subjects=subjects,
        return_concat=True,
        n_jobs=None,
        verbose=False,
    )
    return dataset, X, y, meta


def select_balanced_indices(y, limit):
    labels = sorted(np.unique(y))
    per_label = {label: np.flatnonzero(y == label) for label in labels}
    cursors = {label: 0 for label in labels}
    selected = []

    while len(selected) < limit:
        progressed = False
        for label in labels:
            cursor = cursors[label]
            label_indices = per_label[label]
            if cursor < len(label_indices):
                selected.append(label_indices[cursor])
                cursors[label] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break

    return np.array(selected, dtype=int)
