import json
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metabci.brainda.algorithms.decomposition import FBCSPSVMRM
from metabci.brainda.algorithms.rehab import (
    NearestCentroidMI,
    create_model_bundle,
    load_model_bundle,
    save_model_bundle,
)
from metabci.brainda.datasets import RehabMIDataset
from metabci.brainda.paradigms import MotorImagery
from metabci.brainflow.feedback import ClosedLoopFeedback
from metabci.brainflow.neuracle import LSLMarkerBridge
from metabci.brainstim.rehab_mi import (
    OnlineFeedbackReceiver,
    RehabMIParadigm,
    RehabMIPhase,
    VREventSender,
)
from metabci.brainstim.vr import EventHub


class RehabMIPlatformTests(unittest.TestCase):
    def make_recording(self, root):
        session = root / "sub01" / "formal01"
        session.mkdir(parents=True)
        srate = 100.0
        samples = 600
        timestamps = np.arange(samples) / srate + 1000.0
        data = np.random.RandomState(7).normal(
            size=(samples, 4)
        ).astype(np.float32)
        np.savez_compressed(
            session / "recording.npz",
            data=data,
            timestamps=timestamps,
            marker_values=np.asarray([1, 2]),
            marker_timestamps=np.asarray([1001.0, 1003.5]),
        )
        with (session / "meta.json").open("w", encoding="utf-8") as file:
            json.dump({"srate": srate, "num_chans": 4}, file)

    def test_dataset_and_motor_imagery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_recording(root)
            dataset = RehabMIDataset(
                root,
                channels=["C3", "CZ", "C4"],
                srate=100,
                interval=(0.0, 0.5),
            )
            paradigm = MotorImagery(
                channels=["C3", "CZ", "C4"],
                events=["left_hand", "right_hand"],
                intervals=[(0.0, 0.5)],
            )
            X, y, meta = paradigm.get_data(
                dataset,
                subjects=["sub01"],
                return_concat=True,
                n_jobs=1,
            )
            self.assertEqual(X.shape, (2, 3, 50))
            self.assertEqual(sorted(y.tolist()), [0, 1])
            self.assertEqual(set(meta["session"]), {"formal01"})

    def test_model_bundle_roundtrip(self):
        X = np.asarray([[0.0], [0.2], [2.0], [2.2]])
        y = np.asarray([0, 0, 1, 1])
        estimator = NearestCentroidMI().fit(X, y)
        bundle = create_model_bundle(
            algorithm="centroid",
            estimator=estimator,
            preprocessing=None,
            channels=["C3"],
            srate=250,
            tmin=0.5,
            tmax=4.5,
            subject="sub01",
            sessions=["formal01", "formal02"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_model_bundle(bundle, Path(directory) / "model.pkl")
            loaded = load_model_bundle(path)
            self.assertEqual(loaded["subject"], "sub01")
            self.assertEqual(
                loaded["estimator"].predict([[2.1]]).tolist(),
                [1],
            )

    def test_state_machine_has_one_marker_per_trial(self):
        events = list(RehabMIParadigm(2, random_state=1).events())
        imagery = [
            event
            for event in events
            if event.phase == RehabMIPhase.MOTOR_IMAGERY
        ]
        self.assertEqual(len(imagery), 4)
        self.assertEqual(len({event.trial_id for event in imagery}), 4)
        self.assertEqual({event.marker for event in imagery}, {1, 2})
        self.assertEqual(events[0].phase, RehabMIPhase.START)
        self.assertEqual(events[-1].phase, RehabMIPhase.STOP)

    def test_lsl_marker_deduplication(self):
        bridge = LSLMarkerBridge(duplicate_window=0.1)
        first = bridge._parse_sample("1", 10.0)
        duplicate = bridge._parse_sample("1", 10.02)
        next_trial = bridge._parse_sample("1", 11.0)
        self.assertFalse(bridge._is_duplicate(first))
        self.assertTrue(bridge._is_duplicate(duplicate))
        self.assertFalse(bridge._is_duplicate(next_trial))

    def test_lsl_feedback_boundary_marker(self):
        bridge = LSLMarkerBridge()
        marker = bridge._parse_sample("feedback_start:7", 12.5)
        self.assertEqual(marker.event, "feedback_start")
        self.assertEqual(marker.trial_id, 7)

    def test_feedback_simulation(self):
        feedback = ClosedLoopFeedback().open()
        result = feedback.send(0)
        feedback.close()
        self.assertIn("left_hand", result.robot_command)
        self.assertTrue(result.fes_command.endswith("disabled"))

    def test_online_feedback_udp_roundtrip(self):
        receiver = OnlineFeedbackReceiver(port=0).start()
        sender = VREventSender(
            enabled=True,
            port=receiver.port,
            source="test_online_decoder",
        )
        try:
            sender.send(
                "feedback_sent",
                trial=2,
                target="right_hand",
                prediction="left_hand",
            )
            deadline = time.time() + 1.0
            result = None
            while result is None and time.time() < deadline:
                result = receiver.get(2)
                time.sleep(0.01)
            self.assertIsNotNone(result)
            self.assertEqual(result["target"], "right_hand")
            self.assertEqual(result["prediction"], "left_hand")
        finally:
            sender.close()
            receiver.close()

    def test_vr_feedback_waits_for_feedback_boundary(self):
        hub = EventHub()
        hub.publish(
            {
                "event": "feedback_sent",
                "phase": "FEEDBACK",
                "trial": 3,
                "target": "right_hand",
                "prediction": "left_hand",
            }
        )
        hub.publish(
            {
                "event": "feedback_start",
                "phase": "FEEDBACK",
                "trial": 3,
                "target": "right_hand",
            }
        )
        self.assertEqual(hub.latest["event"], "feedback_start")
        self.assertEqual(hub.latest["prediction"], "left_hand")

    def test_fbcspsvmrm_fit_predict(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(12, 3, 128))
        X[6:, 0] *= 2.5
        y = np.asarray([0] * 6 + [1] * 6)
        model = FBCSPSVMRM(
            srate=128,
            bands=((8, 12), (12, 20)),
            n_csp_components=2,
        )
        model.fit(X, y)
        self.assertEqual(model.predict(X).shape, (12,))


if __name__ == "__main__":
    unittest.main()
