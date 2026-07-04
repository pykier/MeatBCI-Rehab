import argparse
import subprocess
import sys
import time
from pathlib import Path

from rehab_config import load_config, selected_model_path












def parse_args():
    parser = argparse.ArgumentParser(
        description="One-command launcher for the MetaBCI rehab MI online demo."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--control-source", choices=["target", "prediction"], default="target")
    parser.add_argument("--robot-mode", choices=["sim", "serial"], default="serial")
    parser.add_argument("--robot-side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--num-chans", type=int, default=17)
    parser.add_argument("--eeg-chans", type=int, default=16)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--epoch-timeout", type=float, default=10.0)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--nrep", type=int, default=3)
    parser.add_argument("--lsl-source-id", default="rehab_mi_marker_stream")
    parser.add_argument("--feedback-mode", choices=["target", "random", "none"], default="none")
    parser.add_argument("--initial-wait-time", type=float, default=10.0)
    parser.add_argument("--cue-time", type=float, default=2.0)
    parser.add_argument("--imagery-time", type=float, default=5.0)
    parser.add_argument("--rest-time", type=float, default=3.0)
    parser.add_argument("--feedback-time", type=float, default=8.0)
    parser.add_argument("--left-image", default=None)
    parser.add_argument("--right-image", default=None)
    parser.add_argument("--left-gif", default=None)
    parser.add_argument("--right-gif", default=None)
    parser.add_argument("--vr", action="store_true", help="Start the VR web scene and send VR events.")
    parser.add_argument("--vr-http-port", type=int, default=8766)
    parser.add_argument("--vr-udp-port", type=int, default=8765)
    parser.add_argument("--stim-feedback-port", type=int, default=8764)
    parser.add_argument("--no-stim", action="store_true", help="Start only online decoder and optional VR scene.")
    parser.add_argument("--stim-delay", type=float, default=8.0)
    parser.add_argument("--win-size", nargs=2, type=int, default=None)
    parser.add_argument("--window-scale", type=float, default=0.8)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--screen-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[2]


def python_cmd(script, *args):
    return [sys.executable, str(script), *[str(arg) for arg in args]]


def add_if(cmd, condition, *args):
    if condition:
        cmd.extend(str(arg) for arg in args)


def start_process(name, cmd, cwd):
    print(f"\n[{name}]")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    return subprocess.Popen(cmd, cwd=str(cwd))


def stop_processes(processes):
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"Stopping {name}...")
            proc.terminate()
    deadline = time.time() + 6.0
    for name, proc in reversed(processes):
        if proc.poll() is None:
            timeout = max(0.1, deadline - time.time())
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"Killing {name}...")
                proc.kill()


def main():
    args = parse_args()
    config = load_config(args.config)
    args.model = args.model or str(selected_model_path(config))
    args.tmin = float(args.tmin if args.tmin is not None else config.get("tmin", 1.0))
    args.tmax = float(args.tmax if args.tmax is not None else config.get("tmax", 4.0))
    root = repo_root()
    demo_dir = Path(__file__).resolve().parent
    print(f"Config: {config['_config_path']}")
    print(f"Selected model: {args.model}")

    if args.robot_mode == "serial":
        print("This demo can move the robot hands. Keep hands clear of the devices.")
        answer = input(
            f"Robot config left={args.left_com}, right={args.right_com}, side={args.robot_side}. Type YES to continue: "
        )
        if answer != "YES":
            print("Cancelled.")
            return

    vr_cmd = python_cmd(
        demo_dir / "vr_scene_server.py",
        "--port",
        args.vr_http_port,
        "--udp-port",
        args.vr_udp_port,
        "--asset-dir",
        demo_dir / "assets",
        "--quiet",
    )

    online_cmd = python_cmd(
        demo_dir / "online_neuracle_closed_loop.py",
        "--config",
        config["_config_path"],
        "--model",
        args.model,
        "--control-source",
        args.control_source,
        "--robot-mode",
        args.robot_mode,
        "--robot-side",
        args.robot_side,
        "--left-com",
        args.left_com,
        "--right-com",
        args.right_com,
        "--num-chans",
        args.num_chans,
        "--eeg-chans",
        args.eeg_chans,
        "--max-trials",
        args.max_trials,
        "--epoch-timeout",
        args.epoch_timeout,
        "--tmin",
        args.tmin,
        "--tmax",
        args.tmax,
        "--lsl-source-id",
        args.lsl_source_id,
        "--stim-feedback-port",
        args.stim_feedback_port,
    )
    add_if(online_cmd, args.vr, "--vr-events", "--vr-port", args.vr_udp_port)

    stim_cmd = python_cmd(
        demo_dir / "rehab_stim_demo.py",
        "--direct",
        "--nrep",
        args.nrep,
        "--lsl-markers",
        "--lsl-source-id",
        args.lsl_source_id,
        "--feedback-mode",
        args.feedback_mode,
        "--initial-wait-time",
        args.initial_wait_time,
        "--cue-time",
        args.cue_time,
        "--imagery-time",
        args.imagery_time,
        "--rest-time",
        args.rest_time,
        "--feedback-time",
        args.feedback_time,
        "--window-scale",
        args.window_scale,
        "--screen-id",
        args.screen_id,
        "--online-feedback",
        "--online-feedback-port",
        args.stim_feedback_port,
    )
    add_if(stim_cmd, args.win_size is not None, "--win-size", *(args.win_size or []))
    add_if(stim_cmd, args.left_image is not None, "--left-image", args.left_image)
    add_if(stim_cmd, args.right_image is not None, "--right-image", args.right_image)
    add_if(stim_cmd, args.left_gif is not None, "--left-gif", args.left_gif)
    add_if(stim_cmd, args.right_gif is not None, "--right-gif", args.right_gif)
    add_if(stim_cmd, args.fullscreen, "--fullscreen")
    add_if(stim_cmd, args.vr, "--vr-events", "--vr-port", args.vr_udp_port)

    if args.dry_run:
        print("Dry run commands:")
        if args.vr:
            print("VR:", " ".join(vr_cmd))
        print("ONLINE:", " ".join(online_cmd))
        if not args.no_stim:
            print("STIM:", " ".join(stim_cmd))
        return

    processes = []
    try:
        if args.vr:
            processes.append(("vr", start_process("vr", vr_cmd, root)))
            time.sleep(1.5)
        processes.append(("online", start_process("online", online_cmd, root)))
        if not args.no_stim:
            print(f"\nWaiting {args.stim_delay:.1f}s before starting Brainstim...")
            time.sleep(args.stim_delay)
            online_proc = next((proc for name, proc in processes if name == "online"), None)
            if online_proc and online_proc.poll() is not None:
                print("Online decoder exited before Brainstim started. Fix the online error above and rerun.")
                return
            processes.append(("stim", start_process("stim", stim_cmd, root)))

        while processes:
            alive = [(name, proc) for name, proc in processes if proc.poll() is None]
            if not alive:
                break
            if not args.no_stim:
                stim_proc = next((proc for name, proc in processes if name == "stim"), None)
                online_proc = next((proc for name, proc in processes if name == "online"), None)
                if stim_proc and online_proc and stim_proc.poll() is not None and online_proc.poll() is not None:
                    break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
