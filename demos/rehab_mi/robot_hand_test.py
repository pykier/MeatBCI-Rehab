import argparse
from contextlib import nullcontext

from serial.tools import list_ports

from device_adapters import LABEL_NAMES, SerialRobotHandController


LINE_ENDINGS = {
    "none": "",
    "cr": "\r",
    "lf": "\n",
    "crlf": "\r\n",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test serial robot hand control.")
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--side", choices=["left", "right", "both"], default="left")
    parser.add_argument("--init-command", default="A3")
    parser.add_argument("--move-command", default="G5")
    parser.add_argument("--reset-command", default="G5")
    parser.add_argument("--line-ending", choices=LINE_ENDINGS.keys(), default="none")
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually write commands to the serial ports. Without this flag, the script only prints the planned commands.",
    )
    return parser.parse_args()


def print_ports():
    ports = list(list_ports.comports())
    if not ports:
        print("No serial ports detected.")
        return
    for port in ports:
        print(f"{port.device}\t{port.description}\t{port.hwid}")


def side_to_labels(side):
    if side == "left":
        return [0]
    if side == "right":
        return [1]
    return [0, 1]


def side_to_active_hands(side):
    if side == "left":
        return ("left_hand",)
    if side == "right":
        return ("right_hand",)
    return ("left_hand", "right_hand")


def main():
    args = parse_args()
    if args.list_ports:
        print_ports()
        if not args.open_only and not args.execute:
            return

    dry_run = not args.execute
    controller = SerialRobotHandController(
        left_port=args.left_com,
        right_port=args.right_com,
        baudrate=args.baudrate,
        init_command=args.init_command,
        move_command=args.move_command,
        reset_command=args.reset_command,
        line_ending=LINE_ENDINGS[args.line_ending],
        dry_run=dry_run,
        active_hands=side_to_active_hands(args.side),
    )

    context = controller if args.execute or args.open_only else nullcontext(controller)
    with context as robot:
        if args.open_only:
            mode = "opened" if args.execute else "dry-run open check skipped"
            print(f"left={args.left_com}, right={args.right_com}, baudrate={args.baudrate}, {mode}")
            return

        for label in side_to_labels(args.side):
            hand = LABEL_NAMES[label]
            print(f"Testing {hand}...")
            result = robot.send(label)
            print(result)

    if dry_run:
        print("Dry run only. Add --execute to actually send commands.")


if __name__ == "__main__":
    main()
