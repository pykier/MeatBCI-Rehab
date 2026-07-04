# -*- coding: utf-8 -*-
"""MetaBCI dataset adapter for collected rehabilitation MI recordings."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from mne.io import Raw

from .base import BaseDataset


DEFAULT_CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T3", "T4", "P7", "P8",
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _deduplicate_markers(
    values: np.ndarray,
    timestamps: np.ndarray,
    min_interval: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop repeated software markers produced within one trial transition."""
    kept_values = []
    kept_times = []
    last_time = None
    last_value = None
    for value, timestamp in zip(values, timestamps):
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        timestamp = float(timestamp)
        if value not in (1, 2):
            continue
        if (
            last_time is not None
            and value == last_value
            and timestamp - last_time < min_interval
        ):
            continue
        kept_values.append(value)
        kept_times.append(timestamp)
        last_value = value
        last_time = timestamp
    return np.asarray(kept_values, dtype=int), np.asarray(kept_times, dtype=float)


class RehabMIDataset(BaseDataset):
    """Load subject/session Neuracle recordings through MetaBCI ``BaseDataset``.

    The supported session format is the existing project format:
    ``recording.npz`` plus optional ``meta.json``. A converted ``raw.fif`` is
    preferred when it exists.
    """

    def __init__(
        self,
        root: Union[str, Path],
        subjects: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        srate: float = 250.0,
        interval: Tuple[float, float] = (1.0, 4.0),
        data_scale: float = 1e-6,
    ):
        self.root = Path(root).expanduser().resolve()
        self.data_scale = float(data_scale)
        if subjects is None:
            subjects = sorted(
                path.name
                for path in self.root.glob("*")
                if path.is_dir()
            )
        channels = list(channels or DEFAULT_CHANNELS)
        super().__init__(
            dataset_code="rehab_mi",
            subjects=list(subjects),
            events={
                "left_hand": (1, interval),
                "right_hand": (2, interval),
            },
            channels=channels,
            srate=float(srate),
            paradigm="imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose=None,
    ) -> List[List[Path]]:
        del force_update, update_path, proxies, verbose
        root = Path(path).expanduser().resolve() if path else self.root
        subject_dir = root / str(subject)
        if not subject_dir.exists():
            raise FileNotFoundError(f"Rehab MI subject directory not found: {subject_dir}")

        sessions = []
        for session_dir in sorted(item for item in subject_dir.iterdir() if item.is_dir()):
            fif_path = session_dir / "raw.fif"
            npz_path = session_dir / "recording.npz"
            if fif_path.exists():
                sessions.append([fif_path])
            elif npz_path.exists():
                sessions.append([npz_path])
        if not sessions:
            raise FileNotFoundError(f"No raw.fif or recording.npz found under {subject_dir}")
        return sessions

    def _get_single_subject_data(
        self,
        subject: Union[str, int],
        verbose=None,
    ) -> Dict[str, Dict[str, Raw]]:
        del verbose
        sessions = {}
        for run_paths in self.data_path(subject):
            source = Path(run_paths[0])
            session_name = source.parent.name
            if source.suffix.lower() == ".fif":
                raw = mne.io.read_raw_fif(source, preload=True, verbose=False)
            else:
                raw = self._raw_from_npz(source)
            sessions[session_name] = {"run_0": raw}
        return sessions

    def _raw_from_npz(self, npz_path: Path) -> Raw:
        with np.load(npz_path, allow_pickle=True) as content:
            data = np.asarray(content["data"], dtype=np.float64).copy()
            timestamps = np.asarray(
                content["timestamps"],
                dtype=np.float64,
            ).copy()
            marker_values = np.asarray(
                content.get("marker_values", []),
                dtype=object,
            ).copy()
            marker_timestamps = np.asarray(
                content.get("marker_timestamps", []),
                dtype=np.float64,
            ).copy()
        meta = _load_json(npz_path.parent / "meta.json")
        srate = float(meta.get("srate", self.srate))

        eeg_count = min(len(self.channels), data.shape[1])
        eeg = data[:, :eeg_count].T * self.data_scale
        stim = np.zeros(data.shape[0], dtype=np.float64)
        marker_values, marker_timestamps = _deduplicate_markers(
            marker_values,
            marker_timestamps,
        )
        for value, marker_time in zip(marker_values, marker_timestamps):
            index = int(np.searchsorted(timestamps, marker_time, side="left"))
            if 0 <= index < len(stim):
                stim[index] = value

        channel_names = self.channels[:eeg_count] + ["STI 014"]
        info = mne.create_info(
            channel_names,
            sfreq=srate,
            ch_types=["eeg"] * eeg_count + ["stim"],
        )
        raw = mne.io.RawArray(
            np.vstack((eeg, stim[None, :])),
            info,
            verbose=False,
        )
        raw.info["description"] = (
            f"MetaBCI rehabilitation MI recording: {npz_path.parent}"
        )
        return raw

    def export_session_to_fif(
        self,
        subject: Union[str, int],
        session: str,
        overwrite: bool = False,
    ) -> Path:
        """Convert one collected session to a portable MNE FIF file."""
        session_dir = self.root / str(subject) / session
        source = session_dir / "recording.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        destination = session_dir / "raw.fif"
        raw = self._raw_from_npz(source)
        raw.save(destination, overwrite=overwrite)
        return destination
