"""One-command offline collection, epoching, and model training workflow.

Edit the constants below when changing the subject, trial count, or algorithm.
The online demo is intentionally not started here.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


# ===== Editable competition defaults =====
SUBJECT = "sub04"
SESSIONS = ("formal01", "formal02")

NREP = 15  # each repetition contains one left-hand and one right-hand trial
SRATE = 250
NUM_CHANS = 17
LSL_SOURCE_ID = "rehab_mi_marker_stream"

LEFT_COM = "COM4"
RIGHT_COM = "COM3"
ROBOT_FEEDBACK_DURING_OFFLINE = True


ALGORITHM = "eegnet"  # eegnet, secnet, fbmsnet, eegconformer, ifnet, mfanet, fbcspsvm, fbcspsvmrm, svc, centroid
TRAIN_EPOCHS = 350

EARLY_STOPPING_PATIENCE = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.005
DROPOUT = 0.5
VAL_RATIO = 0.2
RANDOM_STATE = 42

ALLOW_OVERWRITE_RECORDINGS = False
OVERWRITE_EPOCHS = True

COLLECT_TO_STIM_DELAY_SECONDS = 1.0
BETWEEN_SESSION_WAIT_SECONDS = 120.0
BETWEEN_STAGE_WAIT_SECONDS = 2.0
COLLECT_STOP_TIMEOUT_SECONDS = 60.0


def repo_root():
    return Path(__file__).resolve().parents[2]


def demo_script(name):
    return Path(__file__).resolve().parent / name


def add_if(command, condition, *args):
    if condition:
        command.extend(str(arg) for arg in args)


def format_command(command):
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def normalize_command(command):
    return [str(part) for part in command]


def print_step(title):
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)


def run_checked(name, command, cwd):
    command = normalize_command(command)
    print_step(f"RUN {name}")
    print(format_command(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")


def start_process(name, command, cwd):
    command = normalize_command(command)
    print_step(f"START {name}")
    print(format_command(command), flush=True)
    return subprocess.Popen(command, cwd=str(cwd))


def terminate_process(name, process):
    if process is None or process.poll() is not None:
        return
    print(f"Stopping {name} ...", flush=True)
    process.terminate()
    try:
        process.wait(timeout=6.0)
    except subprocess.TimeoutExpired:
        print(f"Killing {name} ...", flush=True)
        process.kill()


def build_collect_command(subject, session, args):
    command = [
        sys.executable,
        str(demo_script("collect_dataset.py")),
        "--subject",
        subject,
        "--session",
        session,
        "--srate",
        args.srate,
        "--num-chans",
        args.num_chans,
        "--lsl-source-id",
        args.lsl_source_id,
    ]
    add_if(command, args.overwrite_recordings, "--overwrite")
    return command


def build_stim_command(args):
    command = [
        sys.executable,
        str(demo_script("rehab_stim_demo.py")),
        "--direct",
        "--nrep",
        args.nrep,
        "--lsl-markers",
        "--lsl-source-id",
        args.lsl_source_id,
        "--feedback-mode",
        "target",
        "--left-com",
        args.left_com,
        "--right-com",
        args.right_com,
    ]
    add_if(command, args.robot_feedback, "--robot-feedback")
    return command


def build_epoch_command(subject, sessions, args):
    command = [
        sys.executable,
        str(demo_script("epoch_subject_sessions.py")),
        "--subject",
        subject,
        "--sessions",
        *sessions,
    ]
    add_if(command, args.overwrite_epochs, "--overwrite")
    return command


def build_train_command(subject, sessions, args):
    command = [
        sys.executable,
        str(demo_script("train_model.py")),
        "--subject",
        subject,
        "--sessions",
        *sessions,
        "--algorithm",
        args.algorithm,
        "--epochs",
        args.epochs,
        "--early-stopping-patience",
        args.early_stopping_patience,
        "--batch-size",
        args.batch_size,
        "--learning-rate",
        args.learning_rate,
        "--dropout",
        args.dropout,
        "--val-ratio",
        args.val_ratio,
        "--random-state",
        args.random_state,
    ]
    add_if(command, args.out is not None, "--out", args.out)
    return command


def run_one_session(subject, session, args, cwd):
    collect = None
    stim = None
    try:
        collect = start_process(
            f"collect {subject}/{session}",
            build_collect_command(subject, session, args),
            cwd,
        )
        time.sleep(args.collect_to_stim_delay)
        stim = start_process(
            f"stim {subject}/{session}",
            build_stim_command(args),
            cwd,
        )

        stim_code = stim.wait()
        if stim_code != 0:
            raise RuntimeError(
                f"Stimulus process failed for {subject}/{session} "
                f"with exit code {stim_code}."
            )

        try:
            collect_code = collect.wait(timeout=args.collect_stop_timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Collection process did not stop after stimulus finished "
                f"for {subject}/{session}. Check whether the STOP marker was sent."
            )
        if collect_code != 0:
            raise RuntimeError(
                f"Collection process failed for {subject}/{session} "
                f"with exit code {collect_code}."
            )
    finally:
        terminate_process("stim", stim)
        terminate_process("collect", collect)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run two-session offline collection, epoching, and training."
    )
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--sessions", nargs="+", default=list(SESSIONS))
    parser.add_argument("--nrep", type=int, default=NREP)
    parser.add_argument("--srate", type=int, default=SRATE)
    parser.add_argument("--num-chans", type=int, default=NUM_CHANS)
    parser.add_argument("--lsl-source-id", default=LSL_SOURCE_ID)
    parser.add_argument("--left-com", default=LEFT_COM)
    parser.add_argument("--right-com", default=RIGHT_COM)
    parser.add_argument("--robot-feedback", action="store_true", default=ROBOT_FEEDBACK_DURING_OFFLINE)
    parser.add_argument("--no-robot-feedback", dest="robot_feedback", action="store_false")
    parser.add_argument("--algorithm", default=ALGORITHM)
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", default=None)
    parser.add_argument("--overwrite-recordings", action="store_true", default=ALLOW_OVERWRITE_RECORDINGS)
    parser.add_argument("--overwrite-epochs", action="store_true", default=OVERWRITE_EPOCHS)
    parser.add_argument("--no-overwrite-epochs", dest="overwrite_epochs", action="store_false")
    parser.add_argument("--collect-to-stim-delay", type=float, default=COLLECT_TO_STIM_DELAY_SECONDS)
    parser.add_argument("--between-session-wait", type=float, default=BETWEEN_SESSION_WAIT_SECONDS)
    parser.add_argument("--between-stage-wait", type=float, default=BETWEEN_STAGE_WAIT_SECONDS)
    parser.add_argument("--collect-stop-timeout", type=float, default=COLLECT_STOP_TIMEOUT_SECONDS)
    return parser.parse_args()


def main():
    args = parse_args()
    cwd = repo_root()

    print_step("OFFLINE PIPELINE CONFIG")
    print(f"Project: {cwd}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"Subject: {args.subject}", flush=True)
    print(f"Sessions: {', '.join(args.sessions)}", flush=True)
    print(f"NREP: {args.nrep} ({args.nrep * 2} trials/session)", flush=True)
    print(f"Algorithm: {args.algorithm}", flush=True)
    print(
        f"Robot feedback: {args.robot_feedback} "
        f"(left={args.left_com}, right={args.right_com})",
        flush=True,
    )
    if args.robot_feedback:
        answer = input(
            "This workflow can move the robot hands during offline feedback. "
            "Keep hands clear. Type YES to continue: "
        )
        if answer != "YES":
            print("Cancelled.", flush=True)
            return

    for index, session in enumerate(args.sessions):
        run_one_session(args.subject, session, args, cwd)
        if index < len(args.sessions) - 1:
            print(
                f"Waiting {args.between_session_wait:.1f}s before next session ...",
                flush=True,
            )
            time.sleep(args.between_session_wait)

    print(
        f"Waiting {args.between_stage_wait:.1f}s before epoch cutting ...",
        flush=True,
    )
    time.sleep(args.between_stage_wait)
    run_checked(
        "epoch sessions",
        build_epoch_command(args.subject, args.sessions, args),
        cwd,
    )

    print(
        f"Waiting {args.between_stage_wait:.1f}s before model training ...",
        flush=True,
    )
    time.sleep(args.between_stage_wait)
    run_checked(
        "train model",
        build_train_command(args.subject, args.sessions, args),
        cwd,
    )

    print_step("OFFLINE PIPELINE FINISHED")
    print("You can now run run_online_demo.py with the trained model.", flush=True)


if __name__ == "__main__":
    main()
