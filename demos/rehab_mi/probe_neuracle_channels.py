import argparse
import socket
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe Neuracle Recorder DataService channel count from raw TCP byte rate."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--srate", type=float, default=250.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--min-chans", type=int, default=8)
    parser.add_argument("--max-chans", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_chunks = []

    print(f"Connecting to {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=5) as sock:
        sock.settimeout(0.2)
        print(f"Reading raw TCP bytes for {args.duration:.1f}s ...")
        start = time.perf_counter()
        while time.perf_counter() - start < args.duration:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            raw_chunks.append(chunk)
        elapsed = time.perf_counter() - start

    byte_count = sum(len(chunk) for chunk in raw_chunks)
    float_count = byte_count // 4
    print(f"Received bytes: {byte_count}")
    print(f"Received float32 values: {float_count}")
    print(f"Wall duration: {elapsed:.3f}s")
    print("Candidate channel counts near target srate:")

    candidates = []
    for chans in range(args.min_chans, args.max_chans + 1):
        samples = float_count / chans
        estimated_srate = samples / elapsed if elapsed > 0 else 0.0
        error = abs(estimated_srate - args.srate)
        candidates.append((error, chans, estimated_srate))

    for error, chans, estimated_srate in sorted(candidates)[:8]:
        print(f"  channels={chans:2d}  estimated_srate={estimated_srate:8.2f} Hz  error={error:6.2f}")


if __name__ == "__main__":
    main()
