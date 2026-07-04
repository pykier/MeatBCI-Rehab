"""Backward-compatible robot-control imports."""

from .feedback import (
    ClosedLoopFeedback,
    RobotHandSimulator,
    SerialRobotHandController,
)

__all__ = [
    "ClosedLoopFeedback",
    "RobotHandSimulator",
    "SerialRobotHandController",
]
