"""Compatibility wrapper for MetaBCI brainflow feedback devices."""

from metabci.brainflow.feedback import (
    ClosedLoopFeedback,
    FESController,
    FESStimulatorSimulator,
    FeedbackResult,
    LABEL_NAMES,
    MRFeedbackSimulator,
    RobotHandSimulator,
    SerialRobotHandController,
    VRFeedbackController,
)

__all__ = [
    "ClosedLoopFeedback",
    "FESController",
    "FESStimulatorSimulator",
    "FeedbackResult",
    "LABEL_NAMES",
    "MRFeedbackSimulator",
    "RobotHandSimulator",
    "SerialRobotHandController",
    "VRFeedbackController",
]
