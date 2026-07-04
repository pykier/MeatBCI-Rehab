import argparse
from concurrent.futures import ThreadPoolExecutor

from device_adapters import SerialRobotHandController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move left and right rehab robot hands once."
    )
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Opening robot hands: left={args.left_com}, right={args.right_com}")
    print("Keep hands clear of the devices.")

    with SerialRobotHandController(
        left_port=args.left_com,
        right_port=args.right_com,
        baudrate=args.baudrate,
        dry_run=False,
        active_hands=("left_hand", "right_hand"),
    ) as robot:
        print("Move both hands once...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(robot.send, 0)
            right_future = executor.submit(robot.send, 1)
            print(left_future.result())
            print(right_future.result())

    print("Done.")


if __name__ == "__main__":
    main()
