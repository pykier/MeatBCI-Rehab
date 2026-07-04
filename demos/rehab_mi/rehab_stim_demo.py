import argparse
import csv
import random
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence
from psychopy import core, event, monitors, visual

from metabci.brainstim.framework import Experiment
from metabci.brainstim.rehab_mi import (
    LSLMarkerPublisher,
    OnlineFeedbackReceiver,
    RehabMIParadigm,
    VREventSender,
)
from metabci.brainstim.utils import NeuraclePort, NeuroScanPort
from metabci.brainflow.feedback import SerialRobotHandController


LABEL_NAMES = {
    0: "left_hand",
    1: "right_hand",
}

LINE_ENDINGS = {
    "none": "",
    "cr": "\r",
    "lf": "\n",
    "crlf": "\r\n",
}


def texture_dir():
    return Path(__file__).resolve().parents[2] / "metabci" / "brainstim" / "textures"


def asset_dir():
    return Path(__file__).resolve().parent / "assets"


def default_asset(name):
    return texture_dir() / name


def default_stim_asset(asset_name, texture_name):
    local_asset = asset_dir() / asset_name
    if local_asset.exists():
        return local_asset
    return default_asset(texture_name)


def first_existing_asset(names, fallback):
    for name in names:
        local_asset = asset_dir() / name
        if local_asset.exists():
            return local_asset
    return fallback


def resolve_asset(path, fallback):
    if path:
        asset_path = Path(path)
        if not asset_path.exists():
            raise FileNotFoundError(f"Asset does not exist: {asset_path}")
        return asset_path
    return fallback


def extract_gif_frames(gif_path, cache_dir, max_frames=90):
    if not gif_path:
        return []

    gif_path = Path(gif_path)
    if not gif_path.exists():
        raise FileNotFoundError(f"GIF does not exist: {gif_path}")

    out_dir = cache_dir / gif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("frame_*.png"))
    if existing:
        return existing[:max_frames]

    frames = []
    with Image.open(gif_path) as image:
        for i, frame in enumerate(ImageSequence.Iterator(image)):
            if i >= max_frames:
                break
            frame_path = out_dir / f"frame_{i:03d}.png"
            frame.convert("RGBA").save(frame_path)
            frames.append(frame_path)
    return frames


def fit_image_size(image_path, max_size, zoom=1.0):
    max_width, max_height = max_size
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        return max_size
    scale = min(max_width / width, max_height / height) * zoom
    return (width * scale, height * scale)


def make_port(device_type, port_addr):
    if not port_addr:
        return None

    if device_type == "NeuroScan":
        return NeuroScanPort(port_addr, use_serial=True)
    if device_type == "Neuracle":
        return NeuraclePort(port_addr)
    raise ValueError(f"Unsupported device type: {device_type}")


def make_lsl_marker_outlet(enabled, source_id):
    if not enabled:
        return None
    return LSLMarkerPublisher(source_id=source_id)


def push_lsl_marker(outlet, label):
    if outlet is not None:
        outlet.push_sample([str(label)])


def send_control_marker(outlet, label):
    push_lsl_marker(outlet, label)


class AsyncRobotFeedback:
    def __init__(self, robot):
        self.robot = robot
        self.thread = None
        self.lock = threading.Lock()
        self.last_result = None
        self.last_error = None

    def trigger(self, label):
        if self.robot is None:
            return None
        with self.lock:
            if self.thread and self.thread.is_alive():
                return "robot_busy"

            def _run():
                try:
                    self.last_result = self.robot.send(label)
                    self.last_error = None
                except Exception as exc:
                    self.last_error = exc
                    self.last_result = f"robot_error:{exc}"
                    print(
                        f"Offline robot feedback failed but stimulus will continue: {exc}",
                        flush=True,
                    )

            self.thread = threading.Thread(target=_run, name="offline-robot-feedback", daemon=True)
            self.thread.start()
            return "robot_started"

    def wait(self, timeout=None):
        with self.lock:
            thread = self.thread
        if thread:
            thread.join(timeout=timeout)


def get_screen_size():
    try:
        root = tk.Tk()
        root.withdraw()
        size = [root.winfo_screenwidth(), root.winfo_screenheight()]
        root.destroy()
        return size
    except Exception:
        return [1280, 720]


def resolve_window_size(win_size, window_scale):
    if win_size:
        return list(win_size)
    screen_w, screen_h = get_screen_size()
    scale = max(0.2, min(float(window_scale), 1.0))
    width = int(screen_w * scale)
    height = int(screen_h * scale)
    # Keep a 16:9 window inside the requested screen fraction.
    if width / height > 16 / 9:
        width = int(height * 16 / 9)
    else:
        height = int(width * 9 / 16)
    return [width, height]


class BoxStim:
    def __init__(self, win, pos, size, color, line_width=4):
        x, y = pos
        width, height = size
        left = x - width / 2
        right = x + width / 2
        bottom = y - height / 2
        top = y + height / 2
        self.lines = [
            visual.Line(win, start=(left, top), end=(right, top), units="pix", lineColor=color, lineWidth=line_width),
            visual.Line(win, start=(right, top), end=(right, bottom), units="pix", lineColor=color, lineWidth=line_width),
            visual.Line(win, start=(right, bottom), end=(left, bottom), units="pix", lineColor=color, lineWidth=line_width),
            visual.Line(win, start=(left, bottom), end=(left, top), units="pix", lineColor=color, lineWidth=line_width),
        ]

    def draw(self):
        for line in self.lines:
            line.draw()


class ProgressLineStim:
    def __init__(self, win, pos, width, color, back_color="#555555", line_width=12):
        self.pos = pos
        self.width = width
        self.back = visual.Line(
            win,
            start=(pos[0] - width / 2, pos[1]),
            end=(pos[0] + width / 2, pos[1]),
            units="pix",
            lineColor=back_color,
            lineWidth=line_width,
        )
        self.bar = visual.Line(
            win,
            start=(pos[0] - width / 2, pos[1]),
            end=(pos[0] - width / 2 + 1, pos[1]),
            units="pix",
            lineColor=color,
            lineWidth=line_width,
        )

    def set_value(self, value):
        value = min(max(float(value), 0.0), 1.0)
        start = (self.pos[0] - self.width / 2, self.pos[1])
        end = (self.pos[0] - self.width / 2 + self.width * value, self.pos[1])
        self.bar.start = start
        self.bar.end = end

    def draw_back(self):
        self.back.draw()

    def draw(self):
        self.back.draw()
        self.bar.draw()


def make_stimuli(
    win,
    left_image,
    right_image,
    left_prompt_image,
    right_prompt_image,
    win_size,
    image_zoom=1.3,
    prompt_image_zoom=1.05,
):
    width, height = win_size
    image_box = (min(width * 0.30, 500), min(height * 0.34, 380))
    prompt_image_box = (min(width * 0.30, 500), min(height * 0.38, 420))
    left_image_size = fit_image_size(left_image, image_box, image_zoom)
    right_image_size = fit_image_size(right_image, image_box, image_zoom)
    left_prompt_image_size = fit_image_size(left_prompt_image, prompt_image_box, prompt_image_zoom)
    right_prompt_image_size = fit_image_size(right_prompt_image, prompt_image_box, prompt_image_zoom)
    left_border_size = (left_image_size[0] * 1.08, left_image_size[1] * 1.08)
    right_border_size = (right_image_size[0] * 1.08, right_image_size[1] * 1.08)
    left_feedback_size = (left_image_size[0] * 1.14, left_image_size[1] * 1.14)
    right_feedback_size = (right_image_size[0] * 1.14, right_image_size[1] * 1.14)
    left_pos = (-width * 0.22, -height * 0.02)
    right_pos = (width * 0.22, -height * 0.02)

    stims = {
        "title": visual.TextStim(
            win,
            text="念力搬砖BCI赛队作品",
            units="pix",
            pos=(0, height * 0.07),
            height=38,
            color="#f2f2f2",
            bold=True,
            font="Microsoft YaHei",
        ),
        "phase": visual.TextStim(
            win,
            text="",
            units="pix",
            pos=(0, height * 0.29),
            height=30,
            color="#f2f2f2",
            bold=True,
        ),
        "hint": visual.TextStim(
            win,
            text="请根据提示进行对应的康复机械手运动想象",
            units="pix",
            pos=(0, -height * 0.04),
            height=30,
            color="#f2f2f2",
            bold=True,
            font="Microsoft YaHei",
        ),
        "online_actual": visual.TextStim(
            win,
            text="",
            units="pix",
            pos=(0, height * 0.04),
            height=38,
            color="#f2f2f2",
            bold=True,
            font="Microsoft YaHei",
        ),
        "online_prediction": visual.TextStim(
            win,
            text="",
            units="pix",
            pos=(0, -height * 0.06),
            height=38,
            color="#f2f2f2",
            bold=True,
            font="Microsoft YaHei",
        ),
        "fixation": visual.TextStim(
            win,
            text="+",
            units="pix",
            pos=(0, 0),
            height=78,
            color="#f2f2f2",
            bold=True,
        ),
        "left_label": visual.TextStim(
            win,
            text="LEFT",
            units="pix",
            pos=(left_pos[0], left_pos[1] - image_box[1] * 0.62),
            height=25,
            color="#cccccc",
            bold=True,
        ),
        "right_label": visual.TextStim(
            win,
            text="RIGHT",
            units="pix",
            pos=(right_pos[0], right_pos[1] - image_box[1] * 0.62),
            height=25,
            color="#cccccc",
            bold=True,
        ),
        "left_image": visual.ImageStim(
            win,
            image=str(left_image),
            units="pix",
            pos=left_pos,
            size=left_image_size,
            opacity=0.42,
        ),
        "right_image": visual.ImageStim(
            win,
            image=str(right_image),
            units="pix",
            pos=right_pos,
            size=right_image_size,
            opacity=0.42,
        ),
        "left_prompt_image": visual.ImageStim(
            win,
            image=str(left_prompt_image),
            units="pix",
            pos=left_pos,
            size=left_prompt_image_size,
            opacity=0.42,
        ),
        "right_prompt_image": visual.ImageStim(
            win,
            image=str(right_prompt_image),
            units="pix",
            pos=right_pos,
            size=right_prompt_image_size,
            opacity=0.42,
        ),
        "left_border": BoxStim(
            win, left_pos, left_border_size, "#4dd2ff", line_width=5
        ),
        "right_border": BoxStim(
            win, right_pos, right_border_size, "#4dd2ff", line_width=5
        ),
        "feedback_left": BoxStim(
            win, left_pos, left_feedback_size, "#61d394", line_width=7
        ),
        "feedback_right": BoxStim(
            win, right_pos, right_feedback_size, "#61d394", line_width=7
        ),
        "progress": ProgressLineStim(
            win, (0, -height * 0.39), width * 0.5, "#61d394", back_color="#555555", line_width=12
        ),
        "gif": visual.ImageStim(
            win,
            image=str(left_image),
            units="pix",
            pos=(0, -height * 0.02),
            size=(image_box[0] * 1.1, image_box[1] * 1.1),
            opacity=1.0,
        ),
    }
    return stims


def set_progress(stims, value):
    stims["progress"].set_value(value)


def draw_scene(
    win,
    stims,
    active=None,
    phase="",
    hint="",
    progress=0.0,
    feedback=None,
    image_set="imagery",
):
    stims["phase"].text = phase
    set_progress(stims, progress)

    stims["phase"].draw()

    display_side = feedback or active
    if display_side in ("left", "right"):
        image_stim = (
            stims[f"{display_side}_prompt_image"]
            if image_set == "prompt"
            else stims[f"{display_side}_image"]
        )
        image_stim.opacity = 1.0
        image_stim.draw()
        stims[f"{display_side}_label"].draw()

    if image_set == "imagery":
        if display_side == "left":
            stims["left_border"].draw()
        elif display_side == "right":
            stims["right_border"].draw()

    if feedback == "left":
        stims["feedback_left"].draw()
    elif feedback == "right":
        stims["feedback_right"].draw()

    stims["progress"].draw()
    win.flip()


def display_side_name(value):
    return str(value or "").replace("_hand", "").lower()


def draw_online_feedback(win, stims, target, prediction, progress):
    actual_text = display_side_name(target) or "-"
    prediction_text = display_side_name(prediction) or "等待中"
    stims["online_actual"].text = f"实际指令：{actual_text}"
    stims["online_prediction"].text = f"预测结果：{prediction_text}"
    set_progress(stims, progress)
    stims["online_actual"].draw()
    stims["online_prediction"].draw()
    stims["progress"].draw()
    win.flip()


def draw_for_seconds(win, seconds, fps, draw_func):
    if seconds <= 0:
        return True
    total_frames = max(1, int(seconds * fps))
    clock = core.Clock()
    frame = 0
    while True:
        elapsed = clock.getTime()
        if elapsed >= seconds:
            break
        keys = event.getKeys(["q", "escape"])
        if keys:
            print(f"Stimulus stopped by keyboard input: {keys}", flush=True)
            return False
        draw_func(frame, total_frames)
        frame += 1
    return True


def play_preview(win, stims, side, frame_paths, seconds, fps):
    if not frame_paths:
        return draw_for_seconds(
            win,
            seconds,
            fps,
            lambda frame, total: draw_scene(
                win,
                stims,
                active=side,
                phase="ACTION PREVIEW",
                hint=f"Watch the {side} rehabilitation movement.",
                progress=(frame + 1) / total,
            ),
        )

    side_image = stims[f"{side}_image"]
    return draw_for_seconds(
        win,
        seconds,
        fps,
        lambda frame, total: draw_gif_preview(
            win,
            stims,
            side,
            side_image,
            frame_paths[frame % len(frame_paths)],
            frame,
            total,
        ),
    )


def draw_gif_preview(win, stims, side, side_image, frame_path, frame, total):
    stims["phase"].text = "ACTION PREVIEW"
    set_progress(stims, (frame + 1) / total)

    stims["phase"].draw()

    stims["gif"].image = str(frame_path)
    stims["gif"].draw()
    if side == "left":
        stims["left_border"].draw()
    else:
        stims["right_border"].draw()
    stims[f"{side}_label"].draw()
    side_image.opacity = 1.0
    stims["progress"].draw()
    win.flip()


def rehab_mi_paradigm(
    win,
    stims,
    fps,
    nrep,
    prepare_time,
    rest_time,
    cue_time,
    preview_time,
    imagery_time,
    feedback_time,
    start_time=1.0,
    initial_wait_time=10.0,
    port=None,
    lsl_outlet=None,
    feedback_mode="target",
    left_frames=None,
    right_frames=None,
    log_path=None,
    vr_sender=None,
    robot_feedback=None,
    robot_feedback_source="target",
    online_feedback_receiver=None,
):
    left_frames = left_frames or []
    right_frames = right_frames or []
    rows = []

    state_machine = RehabMIParadigm(n_repetitions=nrep)
    trials = [
        {
            "id": 0 if target == "left_hand" else 1,
            "side": target.replace("_hand", ""),
        }
        for target in state_machine.trial_targets()
    ]

    if lsl_outlet is not None:
        send_control_marker(lsl_outlet, "start")
    if vr_sender:
        vr_sender.send("experiment_start", phase="START", trial=0)
    ok = draw_for_seconds(
        win,
        start_time,
        fps,
        lambda frame, total: draw_phase_only(win, stims, "START"),
    )
    if not ok:
        if lsl_outlet is not None:
            send_control_marker(lsl_outlet, "stop")
        return

    if vr_sender:
        vr_sender.send("initial_wait", phase="READY", trial=0)
    ok = draw_for_seconds(
        win,
        initial_wait_time,
        fps,
        lambda frame, total: draw_ready(
            win, stims, progress=(frame + 1) / total
        ),
    )
    if not ok:
        if lsl_outlet is not None:
            send_control_marker(lsl_outlet, "stop")
        return

    for trial_no, trial in enumerate(trials, 1):
        side = trial["side"]
        label = int(trial["id"])
        target = LABEL_NAMES[label]
        frames = left_frames if side == "left" else right_frames

        if vr_sender:
            vr_sender.send("trial_start", phase="TRIAL", trial=trial_no, target=target)
        ok = draw_for_seconds(
            win,
            prepare_time,
            fps,
            lambda frame, total: draw_phase_only(win, stims, f"TRIAL {trial_no}"),
        )
        if not ok:
            break

        if vr_sender:
            vr_sender.send("prompt", phase="PROMPT", trial=trial_no, target=target)
        ok = draw_for_seconds(
            win,
            cue_time,
            fps,
            lambda frame, total: draw_scene(
                win,
                stims,
                active=side,
                phase="PROMPT",
                hint=f"Prepare for {side} motor imagery.",
                progress=(frame + 1) / total,
                image_set="prompt",
            ),
        )
        if not ok:
            break

        if preview_time > 0 and vr_sender:
            vr_sender.send("preview", phase="ACTION PREVIEW", trial=trial_no, target=target)
        ok = play_preview(win, stims, side, frames, preview_time, fps)
        if not ok:
            break

        ok = draw_imagery(
            win,
            stims,
            side,
            label,
            imagery_time,
            fps,
            port,
            lsl_outlet,
            trial_no=trial_no,
            vr_sender=vr_sender,
        )
        if not ok:
            break

        pred_label = predict_feedback(label, feedback_mode)
        pred_name = LABEL_NAMES[pred_label] if pred_label is not None else None
        pred_side = pred_name.replace("_hand", "") if pred_name else None
        robot_label = label if robot_feedback_source == "target" or pred_label is None else pred_label
        if vr_sender:
            payload = {
                "phase": "FEEDBACK",
                "trial": trial_no,
                "target": target,
            }
            if feedback_mode == "none":
                # The decoder may finish before the imagery display ends.
                # This event is the authoritative boundary at which VR may
                # reveal a cached online prediction.
                if lsl_outlet is not None:
                    send_control_marker(
                        lsl_outlet,
                        f"feedback_start:{trial_no}",
                    )
                vr_sender.send("feedback_start", **payload)
            else:
                if pred_name:
                    payload["feedback"] = pred_name
                vr_sender.send("stim_feedback", **payload)
        if feedback_time > 0:
            robot_status = None
            if robot_feedback is not None:
                robot_status = robot_feedback.trigger(robot_label)
            if feedback_mode == "none":
                def draw_live_feedback(frame, total):
                    result = (
                        online_feedback_receiver.get(trial_no)
                        if online_feedback_receiver is not None
                        else None
                    )
                    prediction = result.get("prediction") if result else None
                    draw_online_feedback(
                        win,
                        stims,
                        target=target,
                        prediction=prediction,
                        progress=(frame + 1) / total,
                    )

                ok = draw_for_seconds(
                    win,
                    feedback_time,
                    fps,
                    draw_live_feedback,
                )
            else:
                ok = draw_for_seconds(
                    win,
                    feedback_time,
                    fps,
                    lambda frame, total: draw_scene(
                        win,
                        stims,
                        active=side,
                        feedback=pred_side,
                        phase="FEEDBACK",
                        hint=(
                            f"Executing rehab motion: {LABEL_NAMES[robot_label].replace('_hand', '')}."
                            if robot_status == "robot_started"
                            else (f"Predicted feedback: {pred_side}." if pred_side else "Waiting for online feedback.")
                        ),
                        progress=(frame + 1) / total,
                        image_set="prompt",
                    ),
                )
            if not ok:
                break

        if vr_sender:
            vr_sender.send("rest", phase="REST", trial=trial_no, target=target)
        ok = draw_for_seconds(
            win,
            rest_time,
            fps,
            lambda frame, total: draw_rest(win, stims, frame, total),
        )
        if not ok:
            break

        rows.append(
            {
                "time": datetime.now().isoformat(timespec="seconds"),
                "trial": trial_no,
                "target": side,
                "trigger": label + 1,
                "feedback": pred_side or "",
                "feedback_mode": feedback_mode,
            }
        )

    if log_path and rows:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if robot_feedback is not None:
        robot_feedback.wait(timeout=10.0)
    if lsl_outlet is not None:
        send_control_marker(lsl_outlet, "stop")
    if vr_sender:
        vr_sender.send("experiment_stop", phase="STOP", trial=len(rows))


def draw_rest(win, stims, frame, total):
    stims["phase"].text = "REST"
    set_progress(stims, (frame + 1) / total)
    stims["phase"].draw()
    stims["fixation"].draw()
    stims["progress"].draw()
    win.flip()


def draw_ready(win, stims, progress):
    stims["phase"].text = "READY"
    set_progress(stims, progress)
    stims["phase"].draw()
    stims["title"].draw()
    stims["hint"].draw()
    stims["progress"].draw()
    win.flip()


def draw_phase_only(win, stims, phase):
    stims["phase"].text = phase
    stims["phase"].draw()
    win.flip()


def draw_imagery(
    win,
    stims,
    side,
    label,
    seconds,
    fps,
    port,
    lsl_outlet=None,
    trial_no=None,
    vr_sender=None,
):
    trigger_frame = max(1, int(0.05 * fps))

    def draw(frame, total):
        if frame == 0 and port:
            win.callOnFlip(port.setData, label + 1)
        if frame == 0 and lsl_outlet is not None:
            win.callOnFlip(push_lsl_marker, lsl_outlet, label + 1)
        if frame == 0 and vr_sender:
            win.callOnFlip(
                vr_sender.send,
                "imagery",
                phase="MOTOR IMAGERY",
                trial=trial_no,
                target=LABEL_NAMES[label],
                marker=label + 1,
            )
        if frame == trigger_frame and port:
            port.setData(0)
        draw_scene(
            win,
            stims,
            active=side,
            phase="MOTOR IMAGERY",
            hint=f"Imagine repeatedly moving your {side} hand.",
            progress=(frame + 1) / total,
        )

    return draw_for_seconds(win, seconds, fps, draw)


def predict_feedback(label, mode):
    if mode == "none":
        return None
    if mode == "random":
        return random.choice([0, 1])
    return label


def parse_args():
    out_dir = Path(__file__).resolve().parent / "outputs"
    parser = argparse.ArgumentParser(description="MetaBCI Brainstim rehab MI scene.")
    parser.add_argument("--left-image", default=None)
    parser.add_argument("--right-image", default=None)
    parser.add_argument("--left-prompt-image", default=None)
    parser.add_argument("--right-prompt-image", default=None)
    parser.add_argument("--left-gif", default=None)
    parser.add_argument("--right-gif", default=None)
    parser.add_argument("--nrep", type=int, default=3)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--win-size", nargs=2, type=int, default=None)
    parser.add_argument("--window-scale", type=float, default=0.8)
    parser.add_argument("--image-zoom", type=float, default=1.15)
    parser.add_argument("--prompt-image-zoom", type=float, default=1.05)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--screen-id", type=int, default=0)
    parser.add_argument("--direct", action="store_true", help="Start paradigm without the menu.")
    parser.add_argument("--port-addr", default=None)
    parser.add_argument("--device-type", choices=["NeuroScan", "Neuracle"], default="Neuracle")
    parser.add_argument("--lsl-markers", action="store_true")
    parser.add_argument("--lsl-source-id", default="rehab_mi_marker_stream")
    parser.add_argument("--feedback-mode", choices=["target", "random", "none"], default="target")
    parser.add_argument("--start-time", type=float, default=1.0)
    parser.add_argument("--initial-wait-time", type=float, default=10.0)
    parser.add_argument("--prepare-time", type=float, default=1.0)
    parser.add_argument("--rest-time", type=float, default=3.0)
    parser.add_argument("--cue-time", type=float, default=2.0)
    parser.add_argument("--preview-time", type=float, default=0.0)
    parser.add_argument("--imagery-time", type=float, default=5.0)
    parser.add_argument("--feedback-time", type=float, default=8.0)
    parser.add_argument("--log", default=str(out_dir / "rehab_stim_log.csv"))
    parser.add_argument("--robot-feedback", action="store_true")
    parser.add_argument("--robot-feedback-source", choices=["target", "feedback"], default="target")
    parser.add_argument("--left-com", default="COM4")
    parser.add_argument("--right-com", default="COM3")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--line-ending", choices=LINE_ENDINGS.keys(), default="none")
    parser.add_argument("--vr-events", action="store_true")
    parser.add_argument("--vr-host", default="127.0.0.1")
    parser.add_argument("--vr-port", type=int, default=8765)
    parser.add_argument("--online-feedback", action="store_true")
    parser.add_argument("--online-feedback-host", default="127.0.0.1")
    parser.add_argument("--online-feedback-port", type=int, default=8764)
    return parser.parse_args()


def main():
    args = parse_args()
    args.win_size = resolve_window_size(args.win_size, args.window_scale)
    print(
        "Preparing rehab MI stimulus: "
        f"direct={args.direct}, nrep={args.nrep}, "
        f"feedback_mode={args.feedback_mode}, robot_feedback={args.robot_feedback}",
        flush=True,
    )
    left_image = resolve_asset(args.left_image, default_stim_asset("left_robot_hand.png", "left_hand22.png"))
    right_image = resolve_asset(args.right_image, default_stim_asset("right_robot_hand.png", "right_hand22.png"))
    left_prompt_image = resolve_asset(
        args.left_prompt_image,
        first_existing_asset(["left_prompt_ready.jpg", "left_promt_ready.jpg"], left_image),
    )
    right_prompt_image = resolve_asset(
        args.right_prompt_image,
        first_existing_asset(["right_prompt_ready.jpg", "right_promt_ready.jpg"], right_image),
    )
    left_frames = extract_gif_frames(args.left_gif, Path(__file__).resolve().parent / "outputs" / "gif_cache")
    right_frames = extract_gif_frames(args.right_gif, Path(__file__).resolve().parent / "outputs" / "gif_cache")
    port = make_port(args.device_type, args.port_addr)
    lsl_outlet = make_lsl_marker_outlet(args.lsl_markers, args.lsl_source_id)
    robot_controller = None
    robot_feedback = None
    if args.robot_feedback:
        print(
            f"Opening robot serial ports: left={args.left_com}, right={args.right_com}",
            flush=True,
        )
        robot_controller = SerialRobotHandController(
            left_port=args.left_com,
            right_port=args.right_com,
            baudrate=args.baudrate,
            line_ending=LINE_ENDINGS[args.line_ending],
            dry_run=False,
            active_hands=("left_hand", "right_hand"),
        )
        try:
            robot_controller.__enter__()
            robot_feedback = AsyncRobotFeedback(robot_controller)
            print("Robot serial ports opened.", flush=True)
        except Exception as exc:
            print(
                "WARNING: robot serial ports could not be opened. "
                "The stimulus and markers will continue without robot feedback. "
                f"error={exc}",
                flush=True,
            )
            robot_controller = None
            robot_feedback = None
    vr_sender = VREventSender(
        enabled=args.vr_events,
        host=args.vr_host,
        port=args.vr_port,
        source="brainstim",
    )
    online_feedback_receiver = None
    if args.online_feedback:
        online_feedback_receiver = OnlineFeedbackReceiver(
            host=args.online_feedback_host,
            port=args.online_feedback_port,
        ).start()

    mon = monitors.Monitor(name="rehab_monitor", width=59.6, distance=60, verbose=False)
    mon.setSizePix(args.win_size)
    ex = Experiment(
        monitor=mon,
        bg_color_warm=np.array([-0.85, -0.85, -0.85]),
        screen_id=args.screen_id,
        win_size=np.array(args.win_size),
        is_fullscr=args.fullscreen,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()
    print(f"PsychoPy window ready: size={args.win_size}", flush=True)
    win.color = np.array([-0.85, -0.85, -0.85])
    stims = make_stimuli(
        win,
        left_image,
        right_image,
        left_prompt_image,
        right_prompt_image,
        args.win_size,
        image_zoom=args.image_zoom,
        prompt_image_zoom=args.prompt_image_zoom,
    )

    kwargs = dict(
        win=win,
        stims=stims,
        fps=args.fps,
        nrep=args.nrep,
        prepare_time=args.prepare_time,
        rest_time=args.rest_time,
        cue_time=args.cue_time,
        preview_time=args.preview_time,
        imagery_time=args.imagery_time,
        feedback_time=args.feedback_time,
        start_time=args.start_time,
        initial_wait_time=args.initial_wait_time,
        port=port,
        lsl_outlet=lsl_outlet,
        feedback_mode=args.feedback_mode,
        left_frames=left_frames,
        right_frames=right_frames,
        log_path=Path(args.log),
        vr_sender=vr_sender,
        robot_feedback=robot_feedback,
        robot_feedback_source=args.robot_feedback_source,
        online_feedback_receiver=online_feedback_receiver,
    )

    try:
        if args.direct:
            # ``Experiment.run`` normally performs this initialization.
            # Direct mode bypasses it, so stale q/escape events from a
            # previous PsychoPy window must be cleared explicitly.
            event.clearEvents(eventType="keyboard")
            print(
                "Starting direct rehab MI paradigm: "
                f"nrep={args.nrep}, trials={args.nrep * 2}, "
                f"feedback_mode={args.feedback_mode}",
                flush=True,
            )
            rehab_mi_paradigm(**kwargs)
            print("Direct rehab MI paradigm finished.", flush=True)
            win.close()
        else:
            ex.register_paradigm("rehab MI scene", rehab_mi_paradigm, **kwargs)
            ex.run()
    finally:
        vr_sender.close()
        if online_feedback_receiver is not None:
            online_feedback_receiver.close()
        if robot_controller is not None:
            robot_controller.__exit__(None, None, None)


if __name__ == "__main__":
    main()
