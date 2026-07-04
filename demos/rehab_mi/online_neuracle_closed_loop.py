"""Thin online entry point assembled from MetaBCI platform APIs."""

import argparse
import csv
import queue
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from metabci.brainda.algorithms.rehab import (
    load_model_bundle,
)
from metabci.brainflow.feedback import (
    ClosedLoopFeedback,
    LABEL_NAMES,
    SerialRobotHandController,
)
from metabci.brainflow.amplifiers import Marker
from metabci.brainflow.neuracle import (
    LSLMarkerBridge,
    NeuracleDataService,
)
from metabci.brainflow.rehab import RehabMIPredictionWorker
from metabci.brainstim.rehab_mi import VREventSender

from rehab_config import load_config, selected_model_path


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LINE_ENDINGS = {
    "none": "",
    "cr": "\r",
    "lf": "\n",
    "crlf": "\r\n",
}
MARKER_TO_LABEL = {1: 0, 2: 1}


def parse_args():
    parser = argparse.ArgumentParser(
        description="MetaBCI Neuracle + Marker + online MI closed loop."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--log",
        default=str(OUTPUT_DIR / "online_neuracle_closed_loop_log.csv"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--srate", type=float, default=None)
    parser.add_argument("--num-chans", type=int, default=17)
    parser.add_argument("--eeg-chans", type=int, default=None)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--lsl-source-id", default="rehab_mi_marker_stream")
    parser.add_argument("--max-trials", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--epoch-timeout", type=float, default=8.0)
    parser.add_argument(
        "--control-source",
        choices=["prediction", "target"],
        default="prediction",
    )
    parser.add_argument("--robot-mode", choices=["sim", "serial"], default="sim")
    parser.add_argument("--robot-side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--line-ending", choices=LINE_ENDINGS, default="none")
    parser.add_argument("--require-confirm", action="store_true")
    parser.add_argument("--vr-events", action="store_true")
    parser.add_argument("--vr-host", default="127.0.0.1")
    parser.add_argument("--vr-port", type=int, default=8765)
    parser.add_argument("--stim-feedback-host", default="127.0.0.1")
    parser.add_argument("--stim-feedback-port", type=int, default=8764)
    return parser.parse_args()


def make_feedback(args):
    if args.robot_mode == "sim":
        feedback = ClosedLoopFeedback()
        feedback.open()
        return feedback

    active_hands = {
        "left": ("left_hand",),
        "right": ("right_hand",),
        "both": ("left_hand", "right_hand"),
    }[args.robot_side]
    if args.require_confirm:
        answer = input(
            "About to control robot hands "
            f"left={args.left_com}, right={args.right_com}. Type YES to continue: "
        )
        if answer != "YES":
            raise SystemExit("Cancelled.")
    robot = SerialRobotHandController(
        left_port=args.left_com,
        right_port=args.right_com,
        baudrate=args.baudrate,
        line_ending=LINE_ENDINGS[args.line_ending],
        dry_run=False,
        active_hands=active_hands,
        async_mode=True,
    )
    feedback = ClosedLoopFeedback(robot=robot)
    feedback.open()
    return feedback


def main():
    args = parse_args()
    config = load_config(args.config)
    model_path = Path(args.model) if args.model else selected_model_path(config)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Selected model does not exist: {model_path}\n"
            f"Update selected_model in {config['_config_path']} or pass --model."
        )

    bundle = load_model_bundle(model_path)
    srate = float(args.srate or bundle.get("srate", 250.0))
    eeg_chans = int(args.eeg_chans or bundle.get("eeg_chans", 16))
    tmin = float(args.tmin if args.tmin is not None else bundle.get("tmin", 1.0))
    tmax = float(args.tmax if args.tmax is not None else bundle.get("tmax", 4.0))

    print(f"Config: {config['_config_path']}")
    print(f"Loaded model: {model_path}")
    print(
        "Model subject/sessions: "
        f"{bundle.get('subject', '-')}/{bundle.get('sessions', '-')}"
    )
    print(
        f"Online config: srate={srate}, num_chans={args.num_chans}, "
        f"eeg_chans={eeg_chans}, window={tmin}-{tmax}s"
    )
    print(f"Control source: {args.control_source}, robot mode: {args.robot_mode}")

    markers = LSLMarkerBridge(
        source_id=args.lsl_source_id,
        duplicate_window=0.1,
    )
    amplifier = NeuracleDataService(
        device_address=(args.host, args.port),
        srate=srate,
        num_chans=args.num_chans,
        eeg_chans=eeg_chans,
        marker_bridge=markers,
    )
    epoch_marker = Marker(
        interval=[tmin, tmax],
        srate=srate,
        events=[1, 2],
    )
    worker = RehabMIPredictionWorker(
        model_path=model_path,
        eeg_chans=eeg_chans,
    )
    amplifier.register_worker("rehab_mi_prediction", worker, epoch_marker)
    feedback = make_feedback(args)
    vr_sender = VREventSender(
        enabled=args.vr_events,
        host=args.vr_host,
        port=args.vr_port,
        source="brainflow_online_decoder",
    )
    stim_feedback_sender = VREventSender(
        enabled=True,
        host=args.stim_feedback_host,
        port=args.stim_feedback_port,
        source="brainflow_online_decoder",
    )
    rows = []

    worker.start()
    amplifier.start_trans()
    if not amplifier.ready_event.wait(timeout=10.0):
        raise RuntimeError(
            "Connected to Neuracle DataService but no EEG samples arrived."
        )
    if amplifier.error:
        raise RuntimeError(f"Neuracle DataService error: {amplifier.error}")
    print(f"Neuracle EEG stream ready. Received samples: {amplifier.sample_count}")

    vr_sender.send(
        "online_ready",
        phase="ONLINE READY",
        control_source=args.control_source,
        robot_mode=args.robot_mode,
    )
    start_time = time.time()
    trial_count = 0
    target_count = 0
    pending_targets = deque()
    pending_predictions = {}
    feedback_boundaries = set()
    stop_received = False

    def execute_ready_trials():
        nonlocal trial_count
        executed = []
        ready_trials = sorted(
            set(pending_predictions).intersection(feedback_boundaries)
        )
        for trial_id in ready_trials:
            item = pending_predictions.pop(trial_id)
            feedback_boundaries.discard(trial_id)
            target = item["target"]
            prediction = item["prediction"]
            marker_value = target["marker"]
            target_label = target["target_label"]
            target_name = target["target_name"]
            predicted_label = int(prediction["label"])
            control_label = (
                predicted_label
                if args.control_source == "prediction"
                else target_label
            )
            prediction_name = LABEL_NAMES[predicted_label]
            control_name = LABEL_NAMES[control_label]
            result = feedback.send(control_label)
            trial_count += 1

            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "trial": trial_id,
                "marker": marker_value,
                "event": target_name,
                "target_label": target_name,
                "pred_label": prediction_name,
                "control_label": control_name,
                "correct": predicted_label == target_label,
                "inference_ms": float(prediction.get("inference_ms", 0.0)),
                "prediction_delay_s": float(item["prediction_delay_s"]),
                "robot_command": result.robot_command,
                "fes_command": result.fes_command,
                "vr_command": result.vr_command,
            }
            rows.append(row)
            feedback_payload = {
                "phase": "FEEDBACK",
                "trial": trial_id,
                "target": target_name,
                "prediction": prediction_name,
                "control": control_name,
                "correct": row["correct"],
                "robot_command": result.robot_command,
            }
            vr_sender.send("feedback_sent", **feedback_payload)
            stim_feedback_sender.send("feedback_sent", **feedback_payload)
            print(
                f"trial={trial_id:02d} target={target_name} "
                f"pred={prediction_name} control={control_name} "
                f"epoch_wait={row['prediction_delay_s']:.3f}s "
                f"inference={row['inference_ms']:.1f}ms "
                f"robot={result.robot_command}"
            )
            executed.append(trial_id)
        return executed

    try:
        while True:
            if args.duration > 0 and time.time() - start_time >= args.duration:
                break
            if args.max_trials > 0 and trial_count >= args.max_trials:
                break
            while True:
                try:
                    marker = markers.queue.get_nowait()
                except queue.Empty:
                    break
                if marker["event"] == "experiment_stop":
                    stop_received = True
                    continue
                if marker["event"] == "feedback_start":
                    trial_id = marker.get("trial_id")
                    if trial_id is not None:
                        feedback_boundaries.add(int(trial_id))
                        execute_ready_trials()
                    continue
                if marker["event"] not in ("left_hand", "right_hand"):
                    continue
                marker_value = int(marker["label"])
                target_label = MARKER_TO_LABEL[marker_value]
                target_count += 1
                pending_targets.append(
                    {
                        "trial": target_count,
                        "marker": marker_value,
                        "target_label": target_label,
                        "target_name": LABEL_NAMES[target_label],
                        "received_at": time.monotonic(),
                    }
                )
                vr_sender.send(
                    "marker_received",
                    phase="MOTOR IMAGERY",
                    trial=target_count,
                    target=LABEL_NAMES[target_label],
                    marker=marker_value,
                )

            try:
                prediction = worker.result_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_received and not pending_targets:
                    break
                if amplifier.error:
                    raise RuntimeError(
                        f"Neuracle DataService error: {amplifier.error}"
                    )
                continue
            if not pending_targets:
                print("Prediction received without a matching trial marker; ignored.")
                continue

            target = pending_targets.popleft()
            trial_id = int(target["trial"])
            prediction_delay_s = time.monotonic() - target["received_at"]
            pending_predictions[trial_id] = {
                "target": target,
                "prediction": prediction,
                "prediction_delay_s": prediction_delay_s,
            }
            print(
                f"trial={trial_id:02d} prediction ready after "
                f"{prediction_delay_s:.3f}s; waiting for FEEDBACK boundary."
            )
            execute_ready_trials()
    except KeyboardInterrupt:
        print("Online loop interrupted by user.")
    finally:
        vr_sender.send("online_stopped", phase="STOP")
        vr_sender.close()
        stim_feedback_sender.close()
        if hasattr(amplifier, "_t_loop") and amplifier._t_loop.is_alive():
            amplifier.stop_trans()
        else:
            markers.stop()
            amplifier.close_connection()
        worker.stop()
        worker.join(timeout=5.0)
        feedback.close()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        accuracy = float(np.mean([row["correct"] for row in rows]))
        print(f"Online accuracy over collected trials: {accuracy:.4f}")
        print(f"Saved log: {log_path}")
    else:
        print("No online trials were processed; no log written.")


if __name__ == "__main__":
    main()
