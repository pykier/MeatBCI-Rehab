# -*- coding: utf-8 -*-
"""Reusable rehabilitation motor-imagery state machine for Brainstim."""

import json
import random
import socket
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional


class RehabMIPhase(str, Enum):
    START = "START"
    READY = "READY"
    TRIAL = "TRIAL"
    PROMPT = "PROMPT"
    MOTOR_IMAGERY = "MOTOR IMAGERY"
    FEEDBACK = "FEEDBACK"
    REST = "REST"
    STOP = "STOP"


@dataclass(frozen=True)
class RehabMITiming:
    start: float = 1.0
    ready: float = 10.0
    trial: float = 0.0
    prompt: float = 2.0
    motor_imagery: float = 5.0
    feedback: float = 8.0
    rest: float = 3.0

    def duration(self, phase: RehabMIPhase) -> float:
        mapping = {
            RehabMIPhase.START: self.start,
            RehabMIPhase.READY: self.ready,
            RehabMIPhase.TRIAL: self.trial,
            RehabMIPhase.PROMPT: self.prompt,
            RehabMIPhase.MOTOR_IMAGERY: self.motor_imagery,
            RehabMIPhase.FEEDBACK: self.feedback,
            RehabMIPhase.REST: self.rest,
            RehabMIPhase.STOP: 0.0,
        }
        return float(mapping[phase])


@dataclass(frozen=True)
class RehabMIEvent:
    phase: RehabMIPhase
    trial_id: int = 0
    target: Optional[str] = None
    prediction: Optional[str] = None
    marker: Optional[int] = None
    timestamp: Optional[float] = None

    def to_dict(self):
        payload = asdict(self)
        payload["phase"] = self.phase.value
        payload["time"] = (
            float(self.timestamp) if self.timestamp is not None else time.time()
        )
        return payload


class RehabMIParadigm:
    """Generate one authoritative phase stream for PsychoPy, LSL, and VR."""

    LABELS = {"left_hand": 1, "right_hand": 2}

    def __init__(
        self,
        n_repetitions: int,
        timing: Optional[RehabMITiming] = None,
        random_state: Optional[int] = None,
    ):
        self.n_repetitions = int(n_repetitions)
        self.timing = timing or RehabMITiming()
        self.random_state = random_state
        self._listeners: List[Callable[[RehabMIEvent], None]] = []

    def add_listener(self, listener: Callable[[RehabMIEvent], None]):
        self._listeners.append(listener)
        return listener

    def trial_targets(self) -> List[str]:
        targets = ["left_hand", "right_hand"] * self.n_repetitions
        random.Random(self.random_state).shuffle(targets)
        return targets

    def events(self) -> Iterable[RehabMIEvent]:
        yield RehabMIEvent(RehabMIPhase.START)
        yield RehabMIEvent(RehabMIPhase.READY)
        for trial_id, target in enumerate(self.trial_targets(), start=1):
            marker = self.LABELS[target]
            yield RehabMIEvent(RehabMIPhase.TRIAL, trial_id, target)
            yield RehabMIEvent(RehabMIPhase.PROMPT, trial_id, target)
            yield RehabMIEvent(
                RehabMIPhase.MOTOR_IMAGERY,
                trial_id,
                target,
                marker=marker,
            )
            yield RehabMIEvent(RehabMIPhase.FEEDBACK, trial_id, target)
            yield RehabMIEvent(RehabMIPhase.REST, trial_id, target)
        yield RehabMIEvent(
            RehabMIPhase.STOP,
            trial_id=self.n_repetitions * 2,
        )

    def publish(self, event: RehabMIEvent):
        for listener in tuple(self._listeners):
            listener(event)

    def run(self, renderer, sleep: Callable[[float], None] = time.sleep):
        for event in self.events():
            self.publish(event)
            renderer.render(event, self.timing.duration(event.phase))
            duration = self.timing.duration(event.phase)
            if duration > 0:
                sleep(duration)


class CompositeRenderer:
    def __init__(self, *renderers):
        self.renderers = [renderer for renderer in renderers if renderer is not None]

    def render(self, event: RehabMIEvent, duration: float):
        for renderer in self.renderers:
            renderer.render(event, duration)


class PsychoPyRenderer:
    """Adapter around a callable that performs the actual PsychoPy drawing."""

    def __init__(self, callback: Callable[[RehabMIEvent, float], None]):
        self.callback = callback

    def render(self, event: RehabMIEvent, duration: float):
        self.callback(event, duration)


class VREventSender:
    """Send the same Brainstim phase event to the browser VR backend."""

    def __init__(
        self,
        enabled=False,
        host="127.0.0.1",
        port=8765,
        source="brainstim_rehab_mi",
    ):
        self.enabled = bool(enabled)
        self.host = host
        self.port = int(port)
        self.source = source
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enabled else None

    def send(self, event, **payload):
        if not self.sock:
            return
        message = {
            "event": event,
            "source": self.source,
            "time": time.time(),
            **payload,
        }
        encoded = json.dumps(
            message,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            self.sock.sendto(encoded, (self.host, self.port))
        except OSError:
            return

    def __call__(self, event: RehabMIEvent):
        payload = event.to_dict()
        self.send(event.phase.value.lower().replace(" ", "_"), **payload)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None


class OnlineFeedbackReceiver:
    """Receive online decoder results for the local PsychoPy renderer."""

    def __init__(self, host="127.0.0.1", port=8764):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.results: Dict[int, dict] = {}
        self.error = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return self
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.port = int(self.sock.getsockname()[1])
        self.sock.settimeout(0.2)
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._receive_loop,
            name="rehab-mi-online-feedback",
            daemon=True,
        )
        self.thread.start()
        return self

    def _receive_loop(self):
        while not self.stop_event.is_set():
            try:
                payload, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self.stop_event.is_set():
                    self.error = exc
                break
            try:
                message = json.loads(payload.decode("utf-8"))
                trial_id = int(message.get("trial", 0))
            except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if trial_id <= 0 or message.get("event") != "feedback_sent":
                continue
            with self.lock:
                self.results[trial_id] = message

    def get(self, trial_id):
        with self.lock:
            result = self.results.get(int(trial_id))
            return dict(result) if result is not None else None

    def close(self):
        self.stop_event.set()
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None


class VRSceneRenderer:
    def __init__(self, sender: VREventSender):
        self.sender = sender

    def render(self, event: RehabMIEvent, duration: float):
        payload = event.to_dict()
        payload["duration"] = float(duration)
        self.sender.send(event.phase.value.lower().replace(" ", "_"), **payload)


class LSLMarkerPublisher:
    """Publish control markers and exactly one MI marker per trial."""

    def __init__(self, source_id="rehab_mi_marker_stream"):
        from pylsl import StreamInfo, StreamOutlet

        info = StreamInfo(
            name="MetaBCI_RehabMI_Markers",
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=source_id,
        )
        self.outlet = StreamOutlet(info)
        self._published_trials = set()

    def push_sample(self, sample):
        """Compatibility with a raw pylsl ``StreamOutlet``."""
        self.outlet.push_sample([str(sample[0])])

    def __call__(self, event: RehabMIEvent):
        if event.phase == RehabMIPhase.START:
            self.outlet.push_sample(["start"])
        elif event.phase == RehabMIPhase.STOP:
            self.outlet.push_sample(["stop"])
        elif event.phase == RehabMIPhase.MOTOR_IMAGERY:
            key = (event.trial_id, event.marker)
            if key not in self._published_trials:
                self.outlet.push_sample([str(event.marker)])
                self._published_trials.add(key)
