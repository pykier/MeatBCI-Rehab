"""Preflight checks for offline collection and online VR demonstration."""

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from serial.tools import list_ports

from metabci.brainflow.feedback import SerialRobotHandController
from metabci.brainstim.vr import get_ipv4_hints


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Neuracle EEG, robot hands, and optional VR scene."
    )
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--srate", type=float, default=250.0)
    parser.add_argument("--num-chans", type=int, default=17)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument(
        "--move-hands",
        action="store_true",
        help="Move both robot hands once after opening the serial ports.",
    )
    parser.add_argument("--vr-http-port", type=int, default=8766)
    parser.add_argument("--vr-udp-port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def check_eeg(args):
    bytes_per_sample = int(args.num_chans) * 4
    requested_bytes = max(
        bytes_per_sample,
        int(args.duration * args.srate * bytes_per_sample),
    )
    chunks = []
    deadline = time.monotonic() + max(args.duration, 1.0)

    print(f"[EEG] Connecting to Neuracle DataService {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5.0) as sock:
        sock.settimeout(0.5)
        while time.monotonic() < deadline and sum(map(len, chunks)) < requested_bytes:
            try:
                chunk = sock.recv(requested_bytes)
            except socket.timeout:
                continue
            if not chunk:
                break
            chunks.append(chunk)

    raw = b"".join(chunks)
    usable = len(raw) - len(raw) % bytes_per_sample
    if usable <= 0:
        raise RuntimeError(
            "Neuracle connected but no complete EEG sample was received. "
            "Check Recorder DataService and --num-chans."
        )

    data = np.frombuffer(raw[:usable], dtype="<f4").reshape((-1, args.num_chans))
    if not np.isfinite(data).all():
        raise RuntimeError("EEG stream contains NaN or infinite values.")
    print(
        f"[EEG] OK: samples={len(data)}, channels={data.shape[1]}, "
        f"mean(first 3)={np.round(data.mean(axis=0)[:3], 2).tolist()}"
    )


def check_robot(args):
    detected = {port.device.upper(): port.description for port in list_ports.comports()}
    missing = [
        port
        for port in (args.left_com, args.right_com)
        if port.upper() not in detected
    ]
    if missing:
        raise RuntimeError(
            f"Robot serial ports not detected: {', '.join(missing)}. "
            f"Detected ports: {', '.join(sorted(detected)) or 'none'}"
        )

    action = "open and move" if args.move_hands else "open"
    answer = input(
        f"[ROBOT] About to {action} left={args.left_com}, "
        f"right={args.right_com}. Keep hands clear. Type YES to continue: "
    )
    if answer != "YES":
        raise RuntimeError("Robot check cancelled.")

    with SerialRobotHandController(
        left_port=args.left_com,
        right_port=args.right_com,
        baudrate=args.baudrate,
        dry_run=False,
        active_hands=("left_hand", "right_hand"),
        async_mode=False,
    ) as robot:
        print("[ROBOT] Serial ports opened successfully.")
        if args.move_hands:
            with ThreadPoolExecutor(max_workers=2) as executor:
                left = executor.submit(robot.send, 0)
                right = executor.submit(robot.send, 1)
                print(f"[ROBOT] {left.result()}")
                print(f"[ROBOT] {right.result()}")
    print("[ROBOT] OK: ports released.")


def start_vr(args):
    demo_dir = Path(__file__).resolve().parent
    root = demo_dir.parents[1]
    command = [
        sys.executable,
        str(demo_dir / "vr_scene_server.py"),
        "--port",
        str(args.vr_http_port),
        "--udp-port",
        str(args.vr_udp_port),
        "--asset-dir",
        str(demo_dir / "assets"),
        "--quiet",
    ]
    print("[VR] Starting scene server ...")
    process = subprocess.Popen(command, cwd=str(root))
    time.sleep(1.2)
    if process.poll() is not None:
        raise RuntimeError(
            "VR server exited immediately. Ports 8765/8766 may already be occupied."
        )

    local_url = f"http://127.0.0.1:{args.vr_http_port}"
    print(f"[VR] Computer URL: {local_url}")
    for address in get_ipv4_hints():
        print(f"[VR] Headset URL: http://{address}:{args.vr_http_port}")
    if not args.no_browser:
        webbrowser.open(local_url)
    print("[VR] Server is running. Press Ctrl+C after checking the headset.")
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n[VR] Stopping scene server ...")
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    args = parse_args()
    check_eeg(args)
    check_robot(args)
    if args.mode == "online":
        start_vr(args)
    else:
        print("[PREFLIGHT] Offline collection hardware check passed.")


if __name__ == "__main__":
    main()
