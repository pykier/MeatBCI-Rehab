"""Brainstim public API with optional PsychoPy components loaded lazily."""

from .rehab_mi import (
    CompositeRenderer,
    LSLMarkerPublisher,
    PsychoPyRenderer,
    RehabMIEvent,
    RehabMIParadigm,
    RehabMIPhase,
    RehabMITiming,
    VREventSender,
    VRSceneRenderer,
)

__all__ = [
    "AVEP",
    "Experiment",
    "MI",
    "P300",
    "SSVEP",
    "CompositeRenderer",
    "LSLMarkerPublisher",
    "PsychoPyRenderer",
    "RehabMIEvent",
    "RehabMIParadigm",
    "RehabMIPhase",
    "RehabMITiming",
    "VREventSender",
    "VRSceneRenderer",
]


def __getattr__(name):
    if name == "Experiment":
        from .framework import Experiment

        return Experiment
    if name in {"SSVEP", "P300", "AVEP", "MI"}:
        from .paradigm import AVEP, MI, P300, SSVEP

        return {
            "SSVEP": SSVEP,
            "P300": P300,
            "AVEP": AVEP,
            "MI": MI,
        }[name]
    raise AttributeError(name)
