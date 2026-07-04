# -*- coding: utf-8 -*-
"""Reusable closed-loop feedback devices for rehabilitation BCI."""

import queue
import threading
import time
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

try:
    import serial
except ImportError:  # pragma: no cover - optional hardware dependency
    serial = None


LABEL_NAMES = {
    0: "left_hand",
    1: "right_hand",
}


class FeedbackDevice(metaclass=ABCMeta):
    def open(self):
        return self

    @abstractmethod
    def send(self, label):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()


class RobotHandSimulator(FeedbackDevice):
    def send(self, label):
        return f"robot:{LABEL_NAMES[int(label)]}:grasp_release:simulated"


class SerialRobotHandController(FeedbackDevice):
    """Own both serial ports and execute long motions asynchronously."""

    def __init__(
        self,
        left_port="COM4",
        right_port="COM3",
        baudrate=57600,
        timeout=0.2,
        init_command="A3",
        move_command="G5",
        reset_command="G5",
        line_ending="",
        dry_run=True,
        active_hands=None,
        async_mode=True,
    ):
        self.left_port = left_port
        self.right_port = right_port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.init_command = init_command
        self.move_command = move_command
        self.reset_command = reset_command
        self.line_ending = line_ending
        self.dry_run = bool(dry_run)
        self.active_hands = active_hands or ("left_hand", "right_hand")
        self.async_mode = bool(async_mode)
        self._ports = {}
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None

    def open(self):
        if not self.dry_run:
            if serial is None:
                raise ImportError("pyserial is required for robot serial mode.")
            try:
                if "left_hand" in self.active_hands:
                    self._ports["left_hand"] = serial.Serial(
                        self.left_port,
                        self.baudrate,
                        timeout=self.timeout,
                    )
                if "right_hand" in self.active_hands:
                    self._ports["right_hand"] = serial.Serial(
                        self.right_port,
                        self.baudrate,
                        timeout=self.timeout,
                    )
            except Exception:
                self.close()
                raise
        if self.async_mode:
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="metabci-robot-feedback",
                daemon=True,
            )
            self._thread.start()
        return self

    def close(self):
        if self._thread:
            self._queue.join()
            self._queue.put(None)
            self._thread.join(timeout=10.0)
            self._thread = None
        self._stop_event.set()
        for port in self._ports.values():
            if port and port.is_open:
                port.close()
        self._ports.clear()

    def send(self, label):
        label = int(label)
        hand = LABEL_NAMES[label]
        port_name = self.left_port if hand == "left_hand" else self.right_port
        if self.async_mode:
            self._queue.put(label)
            return f"robot:{hand}:{port_name}:queued"
        return self._execute(label)

    def _worker(self):
        while True:
            try:
                label = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if label is None:
                break
            try:
                self._execute(label)
            except Exception as exc:
                hand = LABEL_NAMES.get(int(label), str(label))
                port_name = self.left_port if hand == "left_hand" else self.right_port
                print(
                    f"Robot serial write failed: hand={hand}, "
                    f"port={port_name}, error={exc}",
                    flush=True,
                )
                self._recover_port(hand)
            finally:
                self._queue.task_done()

    def _recover_port(self, hand):
        if self.dry_run or serial is None:
            return
        port_name = self.left_port if hand == "left_hand" else self.right_port
        old_port = self._ports.get(hand)
        try:
            if old_port and old_port.is_open:
                old_port.close()
        except Exception:
            pass
        try:
            self._ports[hand] = serial.Serial(
                port_name,
                self.baudrate,
                timeout=self.timeout,
            )
            print(
                f"Robot serial port reopened: hand={hand}, port={port_name}",
                flush=True,
            )
        except Exception as exc:
            self._ports.pop(hand, None)
            print(
                f"Robot serial port reopen failed: hand={hand}, "
                f"port={port_name}, error={exc}",
                flush=True,
            )

    def _ensure_port(self, hand):
        if self.dry_run:
            return None
        port = self._ports.get(hand)
        if port is not None and getattr(port, "is_open", False):
            return port
        print(
            f"Robot serial port unavailable; trying to reopen: "
            f"hand={hand}, port={self.left_port if hand == 'left_hand' else self.right_port}",
            flush=True,
        )
        self._recover_port(hand)
        port = self._ports.get(hand)
        if port is not None and getattr(port, "is_open", False):
            return port
        return None

    def _write_command(self, hand, command):
        port_name = self.left_port if hand == "left_hand" else self.right_port
        port = self._ensure_port(hand)
        if port is None:
            raise RuntimeError(f"Robot serial port is not available: {port_name}")
        payload = f"{command}{self.line_ending}".encode("ascii")
        try:
            try:
                port.reset_output_buffer()
            except Exception:
                pass
            port.write(payload)
            port.flush()
            return
        except Exception as exc:
            print(
                f"Robot serial write failed; reopening and retrying once: "
                f"hand={hand}, port={port_name}, command={command}, error={exc}",
                flush=True,
            )
            self._recover_port(hand)
        port = self._ensure_port(hand)
        if port is None:
            raise RuntimeError(f"Robot serial port retry failed: {port_name}")
        try:
            try:
                port.reset_output_buffer()
            except Exception:
                pass
            port.write(payload)
            port.flush()
        except Exception as exc:
            raise RuntimeError(
                f"Robot serial retry write failed: hand={hand}, "
                f"port={port_name}, command={command}, error={exc}"
            ) from exc

    def _execute(self, label):
        hand = LABEL_NAMES[int(label)]
        port_name = self.left_port if hand == "left_hand" else self.right_port
        commands = (
            (self.init_command, 1.0),
            (self.move_command, 4.5),
            (self.reset_command, 2.0),
        )
        if self.dry_run:
            command_text = ",".join(command for command, _ in commands)
            return f"robot:{hand}:{port_name}:dry_run:{command_text}"
        if hand not in self._ports:
            self._recover_port(hand)
        if self._ensure_port(hand) is None:
            return f"robot:{hand}:{port_name}:skipped_port_not_available"
        for command, delay in commands:
            self._write_command(hand, command)
            time.sleep(delay)
        return f"robot:{hand}:{port_name}:sent"


class FESController(FeedbackDevice):
    """Safety-gated FES placeholder; real stimulation is disabled by default."""

    def __init__(self, enabled=False, confirmed=False):
        if enabled and not confirmed:
            raise ValueError("FES requires explicit safety confirmation.")
        self.enabled = bool(enabled)

    def send(self, label):
        hand = LABEL_NAMES[int(label)]
        state = "enabled" if self.enabled else "disabled"
        return f"fes:{hand}:pulse_train:{state}"


class VRFeedbackController(FeedbackDevice):
    def __init__(self, sender=None):
        self.sender = sender

    def send(self, label):
        hand = LABEL_NAMES[int(label)]
        if self.sender:
            self.sender.send(
                "feedback",
                phase="FEEDBACK",
                prediction=hand,
                control=hand,
            )
        return f"vr:{hand}:feedback"


@dataclass
class FeedbackResult:
    robot_command: str
    fes_command: str
    vr_command: str

    @property
    def mr_command(self):
        return self.vr_command


class ClosedLoopFeedback(FeedbackDevice):
    def __init__(self, robot=None, fes=None, vr=None, mr=None):
        self.robot = robot or RobotHandSimulator()
        self.fes = fes or FESController()
        self.vr = vr or mr or VRFeedbackController()

    def open(self):
        self.robot.open()
        self.fes.open()
        self.vr.open()
        return self

    def close(self):
        self.vr.close()
        self.fes.close()
        self.robot.close()

    def send(self, label):
        return FeedbackResult(
            robot_command=self.robot.send(label),
            fes_command=self.fes.send(label),
            vr_command=self.vr.send(label),
        )


FESStimulatorSimulator = FESController
MRFeedbackSimulator = VRFeedbackController
