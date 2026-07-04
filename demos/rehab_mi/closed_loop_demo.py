import argparse
import csv
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from bnci_gdf_dataset import load_bnci_gdf_trials, select_balanced_indices
from device_adapters import (
    ClosedLoopFeedback,
    LABEL_NAMES,
    SerialRobotHandController,
)



LINE_ENDINGS = {
    "none": "",
    "cr": "\r",
    "lf": "\n",
    "crlf": "\r\n",
}


def parse_args():
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description="Simulate MetaBCI MI closed-loop control.")
    parser.add_argument("--data-dir", default=str(root / "BNCI2014004"))
    parser.add_argument("--model", default=str(out_dir / "model.pkl"))
    parser.add_argument("--log", default=str(out_dir / "closed_loop_log.csv"))
    parser.add_argument("--subjects", nargs="*", type=int, default=[1])
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--robot-mode", choices=["sim", "serial"], default="sim")
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--line-ending", choices=LINE_ENDINGS.keys(), default="none")
    parser.add_argument(
        "--control-source",
        choices=["prediction", "target"],
        default="prediction",
        help="Use model prediction or known target labels to drive feedback.",
    )
    parser.add_argument("--require-confirm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    estimator = bundle["estimator"]

    robot = None
    if args.robot_mode == "serial":
        robot = SerialRobotHandController(
            left_port=args.left_com,
            right_port=args.right_com,
            baudrate=args.baudrate,
            line_ending=LINE_ENDINGS[args.line_ending],
            dry_run=False,
        )
        if args.require_confirm:
            answer = input(
                f"About to control robot hands left={args.left_com}, right={args.right_com}. Type YES to continue: "
            )
            if answer != "YES":
                print("Cancelled.")
                return

    _, X, y, meta = load_bnci_gdf_trials(
        args.data_dir,
        subjects=args.subjects,
        window_s=bundle["window_s"],
    )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    feedback_context = robot if robot else None
    feedback = ClosedLoopFeedback(robot=robot)

    if feedback_context:
        feedback_context.__enter__()
    try:
        trial_indices = select_balanced_indices(y, args.limit)
        for i, trial_index in enumerate(trial_indices):
            pred_label = int(estimator.predict(X[trial_index : trial_index + 1])[0])
            true_label = int(y[trial_index])
            control_label = pred_label if args.control_source == "prediction" else true_label
            result = feedback.send(control_label)
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "trial": i + 1,
                "subject": meta.iloc[trial_index]["subject"],
                "event": meta.iloc[trial_index]["event"],
                "true_label": LABEL_NAMES[true_label],
                "pred_label": LABEL_NAMES[pred_label],
                "control_label": LABEL_NAMES[control_label],
                "correct": pred_label == true_label,
                "robot_command": result.robot_command,
                "fes_command": result.fes_command,
                "mr_command": result.mr_command,
            }
            rows.append(row)
            print(
                f"trial={row['trial']:02d} pred={row['pred_label']} control={row['control_label']} "
                f"robot={row['robot_command']} fes={row['fes_command']} mr={row['mr_command']}"
            )
    finally:
        if feedback_context:
            feedback_context.__exit__(None, None, None)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    accuracy = np.mean([row["correct"] for row in rows])
    print(f"Closed-loop simulation accuracy: {accuracy:.4f}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
