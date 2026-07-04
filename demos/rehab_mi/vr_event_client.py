import argparse

from metabci.brainstim.rehab_mi import VREventSender


def parse_args():
    parser = argparse.ArgumentParser(description="Send one MetaBCI VR scene event over UDP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--source", default="manual_test")
    parser.add_argument("--event", default="manual")
    parser.add_argument("--phase", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--prediction", default=None)
    parser.add_argument("--control", default=None)
    parser.add_argument("--trial", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    sender = VREventSender(enabled=True, host=args.host, port=args.port, source=args.source)
    sender.send(
        args.event,
        phase=args.phase,
        target=args.target,
        prediction=args.prediction,
        control=args.control,
        trial=args.trial,
    )
    sender.close()
    print(f"Sent VR event '{args.event}' to {args.host}:{args.port}")


if __name__ == "__main__":
    main()
