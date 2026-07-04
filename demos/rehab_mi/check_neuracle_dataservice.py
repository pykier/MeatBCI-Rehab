import argparse
import socket
import time

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Check Neuracle Recorder DataService TCP output.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--srate", type=int, default=250)
    parser.add_argument("--num-chans", type=int, default=16)
    parser.add_argument("--duration", type=float, default=3.0)
    return parser.parse_args()


def recv_some(sock, max_bytes):
    sock.settimeout(2.0)
    chunks = []
    deadline = time.time() + 2.0
    while time.time() < deadline and sum(len(c) for c in chunks) < max_bytes:
        try:
            chunk = sock.recv(max_bytes)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def main():
    args = parse_args()
    bytes_per_sample = args.num_chans * 4
    max_bytes = int(args.duration * args.srate * bytes_per_sample)

    print(f"Connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        print("Connected. Reading data ...")
        raw = recv_some(sock, max_bytes)

    usable = len(raw) - (len(raw) % bytes_per_sample)
    if usable <= 0:
        print("No usable data received. Check Recorder DataService and channel count.")
        return

    data = np.frombuffer(raw[:usable], dtype="<f4").reshape((-1, args.num_chans))
    print(f"Received bytes: {len(raw)}")
    print(f"Samples x channels: {data.shape}")
    print(f"First row: {np.array2string(data[0], precision=4, suppress_small=True)}")
    print(f"Channel mean first 5: {np.array2string(data.mean(axis=0)[:5], precision=4, suppress_small=True)}")
    print("DataService check ok.")


if __name__ == "__main__":
    main()
