# -*- coding: utf-8 -*-
"""MetaBCI ProcessWorker implementation for online rehabilitation MI."""

import multiprocessing
import time
from pathlib import Path

import numpy as np

from metabci.brainda.algorithms.rehab import (
    STANDARD_DEEP_ALGORITHMS,
    load_model_bundle,
    preprocess_for_bundle,
    restore_bundle_estimator,
)

from .feedback import ClosedLoopFeedback
from .workers import ProcessWorker


class RehabMIPredictionWorker(ProcessWorker):
    """Load a model bundle once and consume epochs from ``Marker``."""

    def __init__(
        self,
        model_path,
        eeg_chans=16,
        feedback=None,
        timeout=0.01,
        name="rehab_mi_prediction",
    ):
        super().__init__(timeout=timeout, name=name)
        self.model_path = str(Path(model_path))
        self.eeg_chans = int(eeg_chans)
        self.feedback = feedback
        self.result_queue = multiprocessing.Queue()

    def pre(self):
        self.bundle = load_model_bundle(self.model_path)
        self.estimator = restore_bundle_estimator(
            self.bundle,
            self.model_path,
        )
        if hasattr(self.estimator, "module_"):
            self.estimator.module_.float()
        if self.feedback:
            self.feedback.open()

    def consume(self, data):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Expected online epoch [samples, channels], got {data.shape}")
        epoch = data[:, :self.eeg_chans].T[None, ...]
        inference_start = time.perf_counter()
        prepared = preprocess_for_bundle(epoch, self.bundle)
        algorithm = str(self.bundle["algorithm"]).strip().lower()
        if algorithm in (*STANDARD_DEEP_ALGORITHMS, "fbmsnet") or hasattr(
            self.estimator, "module_"
        ):
            prepared = np.asarray(prepared, dtype=np.float32)
        label = int(self.estimator.predict(prepared)[0])
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
        payload = {
            "label": label,
            "inference_ms": inference_ms,
        }
        if self.feedback:
            result = self.feedback.send(label)
            payload.update(
                robot_command=result.robot_command,
                fes_command=result.fes_command,
                vr_command=result.vr_command,
            )
        self.result_queue.put(payload)

    def post(self):
        if self.feedback:
            self.feedback.close()
