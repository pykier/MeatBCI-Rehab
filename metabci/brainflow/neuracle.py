# -*- coding: utf-8 -*-
"""Reliable Neuracle DataService and LSL software-marker integration."""

import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

from .amplifiers import BaseAmplifier


MARKER_NAMES = {
    1: "left_hand",
    2: "right_hand",
    "start": "experiment_start",
    "feedback_start": "feedback_start",
    "stop": "experiment_stop",
}


def recv_exact(sock, byte_count, stop_event=None):
    """Receive exactly ``byte_count`` bytes unless the stream closes."""
    chunks = []
    received = 0
    while received < byte_count:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            chunk = sock.recv(byte_count - received)
        except socket.timeout:
            continue
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class SoftwareMarker:
    value: object
    timestamp: float
    event: str
    trial_id: Optional[int] = None


class LSLMarkerBridge:
    """Receive, deduplicate, and expose Brainstim software markers."""

    def __init__(
        self,
        source_id="rehab_mi_marker_stream",
        resolve_timeout=2.0,
        duplicate_window=0.1,
        verbose=False,
    ):
        self.source_id = source_id
        self.resolve_timeout = float(resolve_timeout)
        self.duplicate_window = float(duplicate_window)
        self.verbose = bool(verbose)
        self.stop_event = threading.Event()
        self.experiment_stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.queue = queue.Queue()
        self.rows: List[dict] = []
        self._pending: List[SoftwareMarker] = []
        self._lock = threading.Lock()
        self._last_key = None
        self._last_time = None
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="metabci-lsl-marker-bridge",
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def _parse_sample(self, raw, timestamp):
        trial_id = None
        value = raw
        if isinstance(raw, str) and ":" in raw:
            parts = raw.split(":")
            raw = parts[0]
            if len(parts) > 1:
                try:
                    trial_id = int(parts[1])
                except ValueError:
                    trial_id = None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = str(raw).lower()
        event = MARKER_NAMES.get(value, "unknown")
        return SoftwareMarker(value, float(timestamp), event, trial_id)

    def _is_duplicate(self, marker):
        key = (
            marker.trial_id if marker.trial_id is not None else marker.value,
            marker.event,
        )
        if (
            self._last_key == key
            and self._last_time is not None
            and marker.timestamp - self._last_time < self.duplicate_window
        ):
            return True
        self._last_key = key
        self._last_time = marker.timestamp
        return False

    def _run(self):
        streams = []
        while not self.stop_event.is_set():
            streams = resolve_byprop(
                "source_id",
                self.source_id,
                timeout=self.resolve_timeout,
            )
            if streams:
                break
        if not streams:
            return

        inlet = StreamInlet(streams[0])
        self.connected_event.set()
        while not self.stop_event.is_set():
            sample, timestamp = inlet.pull_sample(timeout=0.1)
            if not sample:
                continue
            marker = self._parse_sample(sample[0], timestamp)
            if self._is_duplicate(marker):
                continue
            row = {
                "lsl_time": marker.timestamp,
                "label": marker.value,
                "event": marker.event,
                "trial_id": marker.trial_id,
            }
            with self._lock:
                self.rows.append(row)
                self._pending.append(marker)
            self.queue.put(row)
            if self.verbose:
                trial_text = (
                    f" trial={marker.trial_id}"
                    if marker.trial_id is not None
                    else ""
                )
                print(
                    f"marker label={marker.value} event={marker.event}"
                    f"{trial_text} time={marker.timestamp:.6f}",
                    flush=True,
                )
            if marker.event == "experiment_stop":
                self.experiment_stop_event.set()

    def pop_until(self, timestamp):
        with self._lock:
            ready = [
                marker
                for marker in self._pending
                if marker.timestamp <= timestamp
            ]
            self._pending = [
                marker
                for marker in self._pending
                if marker.timestamp > timestamp
            ]
        return ready


class NeuracleDataService(BaseAmplifier):
    """MetaBCI ``BaseAmplifier`` implementation for Recorder DataService."""

    def __init__(
        self,
        device_address: Tuple[str, int] = ("127.0.0.1", 8712),
        srate=250.0,
        num_chans=17,
        eeg_chans=16,
        packet_duration=0.04,
        marker_bridge: Optional[LSLMarkerBridge] = None,
        timeout=0.5,
    ):
        super().__init__()
        self.device_address = device_address
        self.srate = float(srate)
        self.num_chans = int(num_chans)
        self.eeg_chans = int(eeg_chans)
        self.packet_duration = float(packet_duration)
        self.packet_samples = max(1, int(round(self.srate * self.packet_duration)))
        self.pkg_size = self.packet_samples * self.num_chans * 4
        self.timeout = float(timeout)
        self.marker_bridge = marker_bridge
        self.tcp_link = None
        self.error = None
        self.sample_count = 0
        self.connected_event = threading.Event()
        self.ready_event = threading.Event()

    def connect(self):
        if self.tcp_link is not None:
            return self
        self.tcp_link = socket.create_connection(
            self.device_address,
            timeout=5.0,
        )
        self.tcp_link.settimeout(self.timeout)
        self.connected_event.set()
        return self

    connect_tcp = connect

    def read_chunk(self):
        if self.tcp_link is None:
            raise RuntimeError("Neuracle DataService is not connected.")
        raw = recv_exact(self.tcp_link, self.pkg_size, self._exit)
        if not raw:
            raise ConnectionError("Neuracle DataService closed the connection.")

        bytes_per_sample = self.num_chans * 4
        usable = len(raw) - len(raw) % bytes_per_sample
        if usable <= 0:
            return np.empty((0, self.num_chans)), np.empty((0,))
        samples = np.frombuffer(raw[:usable], dtype="<f4").reshape(
            (-1, self.num_chans)
        ).copy()
        end_timestamp = local_clock()
        timestamps = end_timestamp - np.arange(
            len(samples) - 1,
            -1,
            -1,
        ) / self.srate
        self.sample_count += len(samples)
        self.ready_event.set()
        return samples, timestamps

    def _inject_markers(self, samples, timestamps):
        if self.marker_bridge is None or len(samples) == 0:
            return samples
        events = self.marker_bridge.pop_until(float(timestamps[-1]))
        for marker in events:
            if marker.value not in (1, 2):
                continue
            index = int(np.searchsorted(timestamps, marker.timestamp, side="left"))
            index = min(max(index, 0), len(samples) - 1)
            samples[index, -1] = int(marker.value)
        return samples

    def recv(self):
        samples, timestamps = self.read_chunk()
        samples = self._inject_markers(samples, timestamps)
        return samples.tolist()

    def _inner_loop(self):
        self._exit.clear()
        while not self._exit.is_set():
            try:
                samples = self.recv()
                if samples:
                    self._detect_event(samples)
            except Exception as exc:
                self.error = exc
                self._exit.set()

    def start_trans(self):
        self.connect()
        if self.marker_bridge:
            self.marker_bridge.start()
        self.start()

    def stop_trans(self):
        if hasattr(self, "_t_loop") and self._t_loop.is_alive():
            self.stop()
        if self.marker_bridge:
            self.marker_bridge.stop()
        self.close_connection()

    def close_connection(self):
        if self.tcp_link is not None:
            try:
                self.tcp_link.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.tcp_link.close()
            self.tcp_link = None


class NeuracleDataBuffer:
    """Timestamped rolling buffer used by the compatibility online demo."""

    def __init__(
        self,
        host="127.0.0.1",
        port=8712,
        srate=250.0,
        num_chans=17,
        eeg_chans=16,
        max_buffer_s=90.0,
    ):
        self.service = NeuracleDataService(
            device_address=(host, int(port)),
            srate=srate,
            num_chans=num_chans,
            eeg_chans=eeg_chans,
        )
        self.srate = float(srate)
        self.num_chans = int(num_chans)
        self.eeg_chans = int(eeg_chans)
        self.max_buffer_samples = int(max_buffer_s * self.srate)
        self.stop_event = threading.Event()
        self.connected_event = threading.Event()
        self.ready_event = threading.Event()
        self.lock = threading.Lock()
        self.data = np.empty((0, self.num_chans), dtype=np.float32)
        self.timestamps = np.empty((0,), dtype=np.float64)
        self.thread = threading.Thread(
            target=self._run,
            name="metabci-neuracle-data-buffer",
            daemon=True,
        )
        self.error = None
        self.sample_count = 0

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.service._exit.set()
        self.service.close_connection()
        self.thread.join(timeout=2.0)

    def _run(self):
        try:
            self.service.connect()
            self.connected_event.set()
            while not self.stop_event.is_set():
                samples, timestamps = self.service.read_chunk()
                if not len(samples):
                    continue
                with self.lock:
                    self.data = np.vstack((self.data, samples))
                    self.timestamps = np.concatenate(
                        (self.timestamps, timestamps)
                    )
                    self.sample_count += len(samples)
                    if len(self.timestamps) > self.max_buffer_samples:
                        self.data = self.data[-self.max_buffer_samples:]
                        self.timestamps = self.timestamps[-self.max_buffer_samples:]
                self.ready_event.set()
        except Exception as exc:
            self.error = exc
            self.connected_event.set()
            self.ready_event.set()
        finally:
            self.service.close_connection()

    def get_epoch(self, marker_time, tmin, tmax, timeout):
        n_samples = int(round((tmax - tmin) * self.srate))
        target_start = float(marker_time) + float(tmin)
        target_stop = float(marker_time) + float(tmax)
        deadline = time.time() + float(timeout)

        while time.time() < deadline:
            if self.error is not None:
                return None, f"data_stream_error_{self.error}"
            with self.lock:
                if len(self.timestamps) and self.timestamps[-1] >= target_stop:
                    timestamps = self.timestamps.copy()
                    data = self.data.copy()
                    break
            time.sleep(0.02)
        else:
            return None, "timeout_waiting_for_epoch"

        start_index = int(np.searchsorted(
            timestamps,
            target_start,
            side="left",
        ))
        stop_index = start_index + n_samples
        if stop_index > len(data):
            return None, f"not_enough_samples_{len(data) - start_index}_{n_samples}"
        epoch = data[start_index:stop_index, :self.eeg_chans].T
        return epoch[None, :, :], "ok"


class NeuracleRecorder:
    """Record timestamped DataService chunks until duration or STOP marker."""

    def __init__(
        self,
        service: NeuracleDataService,
        marker_bridge=None,
        max_reconnects=30,
        reconnect_delay=1.0,
    ):
        self.service = service
        self.marker_bridge = marker_bridge
        self.max_reconnects = int(max_reconnects)
        self.reconnect_delay = float(reconnect_delay)

    def record(self, duration=0.0):
        data_chunks = []
        timestamp_chunks = []
        if self.marker_bridge:
            self.marker_bridge.start()
        self.service.connect()
        start = time.time()
        reconnects = 0
        try:
            while True:
                if duration > 0 and time.time() - start >= duration:
                    break
                if (
                    self.marker_bridge
                    and self.marker_bridge.experiment_stop_event.is_set()
                ):
                    break
                try:
                    samples, timestamps = self.service.read_chunk()
                except (ConnectionError, OSError) as exc:
                    if (
                        self.marker_bridge
                        and self.marker_bridge.experiment_stop_event.is_set()
                    ):
                        print(
                            "DataService closed after experiment STOP; "
                            "saving the received recording."
                        )
                        break
                    reconnects += 1
                    if reconnects > self.max_reconnects:
                        if data_chunks:
                            print(
                                "Neuracle DataService disconnected and could not "
                                f"be restored after {self.max_reconnects} retries. "
                                "Saving the received partial recording.",
                                flush=True,
                            )
                            break
                        raise ConnectionError(
                            "Neuracle DataService disconnected and could not "
                            f"be restored after {self.max_reconnects} retries."
                        ) from exc
                    print(
                        "Neuracle DataService disconnected; reconnecting "
                        f"({reconnects}/{self.max_reconnects}) ..."
                    )
                    self.service.close_connection()
                    time.sleep(self.reconnect_delay)
                    self.service.connect()
                    continue
                reconnects = 0
                if len(samples):
                    data_chunks.append(samples)
                    timestamp_chunks.append(timestamps)
        except KeyboardInterrupt:
            pass
        finally:
            self.service.close_connection()
            if self.marker_bridge:
                self.marker_bridge.stop()
        if not data_chunks:
            raise RuntimeError("No EEG data received from Neuracle DataService.")
        return np.vstack(data_chunks), np.concatenate(timestamp_chunks)
