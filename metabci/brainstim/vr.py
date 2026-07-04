import argparse
import json
import mimetypes
import queue
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


DEFAULT_ASSET_DIR = Path(__file__).resolve().parent / "assets"

PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>MetaBCI VR Rehab Scene</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #081012;
      --panel: rgba(20, 31, 34, 0.74);
      --text: #eef7f5;
      --muted: #95aaa8;
      --cyan: #37c8c1;
      --green: #8be06e;
      --orange: #ffb454;
      --red: #ff6b6b;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      overflow-x: hidden;
      overflow-y: auto;
      background:
        radial-gradient(circle at 50% 120%, rgba(55, 200, 193, 0.18), transparent 44%),
        linear-gradient(180deg, #081012 0%, #102022 100%);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
    }

    .scene {
      position: relative;
      width: min(92vw, 1720px);
      min-height: 100vh;
      display: grid;
      grid-template-rows: 1fr;
      margin: 0 auto;
      padding:
        max(18px, env(safe-area-inset-top))
        max(28px, calc(env(safe-area-inset-right) + 28px))
        max(18px, env(safe-area-inset-bottom))
        max(28px, calc(env(safe-area-inset-left) + 28px));
      gap: clamp(12px, 2vw, 24px);
    }

    .stage {
      display: grid;
      grid-template-columns: minmax(220px, 0.9fr) minmax(240px, 32vw) minmax(220px, 0.9fr);
      align-items: center;
      gap: clamp(12px, 2vw, 28px);
      min-height: 0;
    }

    .stage.center-only {
      grid-template-columns: minmax(260px, 42vw);
      justify-content: center;
    }

    .stage.single-hand {
      grid-template-columns: minmax(240px, 0.9fr) minmax(260px, 0.9fr);
      justify-content: center;
    }

    .hand {
      height: min(58vh, 620px);
      min-height: 260px;
      border: 3px solid transparent;
      border-radius: 8px;
      background: transparent;
      display: grid;
      place-items: center;
    }

    .hand.hidden {
      display: none;
    }

    .hand.active {
      border-color: var(--cyan);
      background: rgba(55, 200, 193, 0.06);
    }

    .hand.feedback {
      border-color: transparent;
      background: transparent;
    }

    .hand svg {
      width: min(70%, 300px);
      max-height: 70%;
      fill: rgba(238,247,245,0.86);
      filter: drop-shadow(0 20px 35px rgba(0,0,0,0.32));
      transition: transform 320ms ease;
    }

    .hand img {
      width: min(92%, 520px);
      height: min(88%, 520px);
      object-fit: contain;
      object-position: center;
      filter: drop-shadow(0 20px 35px rgba(0,0,0,0.32));
    }

    .left img {
      transform: scaleX(-1);
    }

    body.prompt-mode .left img {
      transform: none;
    }

    .left.feedback img {
      transform: scaleX(-1);
    }

    body.prompt-mode .left.feedback img {
      transform: none;
    }

    .hand.feedback svg {
      transform: none;
    }

    .left svg {
      --rot: -8deg;
      transform: scaleX(-1);
    }

    .left.feedback svg {
      transform: scaleX(-1);
    }

    .right svg {
      --rot: 8deg;
    }

    .center {
      align-self: stretch;
      border: 0;
      background: transparent;
      display: grid;
      grid-template-rows: auto 1fr auto;
      align-items: center;
      padding: clamp(18px, 3vw, 34px);
      text-align: center;
      min-height: min(62vh, 640px);
    }

    .stage.center-only .center {
      min-height: auto;
      align-self: center;
      padding: 0;
    }

    .phase {
      font-size: clamp(28px, 5vw, 74px);
      line-height: 1;
      font-weight: 800;
      letter-spacing: 0;
    }

    .cue {
      font-size: clamp(24px, 4vw, 54px);
      color: var(--orange);
      font-weight: 800;
      min-height: 1.1em;
    }

    .feedback-text {
      font-size: clamp(28px, 4vw, 56px);
      color: var(--text);
      font-weight: 800;
      line-height: 1.8;
      min-height: 3.6em;
    }

    .fixation {
      width: clamp(46px, 6vw, 86px);
      height: clamp(46px, 6vw, 86px);
      margin: 0 auto;
      position: relative;
    }

    .fixation.hidden,
    .cue.hidden,
    .feedback-text.hidden {
      display: none;
    }

    .fixation::before,
    .fixation::after {
      content: "";
      position: absolute;
      background: var(--cyan);
      border-radius: 2px;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
    }

    .fixation::before {
      width: 100%;
      height: 8px;
    }

    .fixation::after {
      width: 8px;
      height: 100%;
    }

    @media (max-width: 1400px) {
      .scene {
        width: min(94vw, 1440px);
        padding:
          max(16px, env(safe-area-inset-top))
          max(20px, calc(env(safe-area-inset-right) + 20px))
          max(16px, env(safe-area-inset-bottom))
          max(20px, calc(env(safe-area-inset-left) + 20px));
      }

      .stage {
        grid-template-columns: minmax(180px, 0.85fr) minmax(220px, 34vw) minmax(180px, 0.85fr);
      }

      .stage.center-only {
        grid-template-columns: minmax(260px, 48vw);
      }

      .stage.single-hand {
        grid-template-columns: minmax(220px, 0.9fr) minmax(240px, 0.9fr);
      }

      .hand {
        height: min(46vh, 500px);
        min-height: 220px;
      }
    }

    @media (max-width: 900px) {
      .scene {
        overflow: auto;
        height: auto;
        min-height: 100vh;
      }

      .stage {
        grid-template-columns: 1fr;
      }

      .stage.center-only {
        grid-template-columns: 1fr;
      }

      .stage.single-hand {
        grid-template-columns: 1fr;
      }

      .hand {
        height: 28vh;
        min-height: 180px;
      }

      .center {
        min-height: 260px;
      }
    }
  </style>
</head>

<body>
  <main class="scene">
    <section id="stage" class="stage center-only">
      <div id="left" class="hand left hidden" aria-label="left hand">
        <img id="left-img"
             src="/assets/left_vr_hand.png"
             data-imagery-src="/assets/left_vr_hand.png"
             data-prompt-src="/assets/left_promt_ready.jpg"
             alt=""
             onload="this.style.display='block'; this.nextElementSibling.style.display='none'"
             onerror="this.style.display='none'; this.nextElementSibling.style.display='block'">
        <svg viewBox="0 0 120 160" role="img" aria-hidden="true">
          <path d="M48 16c6 0 10 4 10 10v48h4V18c0-6 5-10 10-10s10 4 10 10v56h4V30c0-6 4-10 10-10s10 4 10 10v76c0 24-20 44-44 44H46c-20 0-36-16-36-36V78c0-7 5-12 12-12s12 5 12 12v18h4V26c0-6 4-10 10-10z"/>
        </svg>
      </div>

      <div class="center">
        <div id="phase" class="phase">READY</div>
        <div>
          <div id="fixation" class="fixation hidden"></div>
          <div id="cue" class="cue hidden"></div>
        </div>
        <div id="feedback" class="feedback-text hidden"></div>
      </div>

      <div id="right" class="hand right hidden" aria-label="right hand">
        <img id="right-img"
             src="/assets/right_vr_hand.png"
             data-imagery-src="/assets/right_vr_hand.png"
             data-prompt-src="/assets/right_promt_ready.jpg"
             alt=""
             onload="this.style.display='block'; this.nextElementSibling.style.display='none'"
             onerror="this.style.display='none'; this.nextElementSibling.style.display='block'">
        <svg viewBox="0 0 120 160" role="img" aria-hidden="true">
          <path d="M48 16c6 0 10 4 10 10v48h4V18c0-6 5-10 10-10s10 4 10 10v56h4V30c0-6 4-10 10-10s10 4 10 10v76c0 24-20 44-44 44H46c-20 0-36-16-36-36V78c0-7 5-12 12-12s12 5 12 12v18h4V26c0-6 4-10 10-10z"/>
        </svg>
      </div>
    </section>
  </main>

  <script>
    var stage = document.getElementById("stage");
    var left = document.getElementById("left");
    var right = document.getElementById("right");
    var leftImg = document.getElementById("left-img");
    var rightImg = document.getElementById("right-img");
    var phase = document.getElementById("phase");
    var fixation = document.getElementById("fixation");
    var cue = document.getElementById("cue");
    var feedback = document.getElementById("feedback");
    var latestEventTime = 0;
    var displayedPhase = "READY";
    var displayedTrial = 0;
    var pendingFeedback = {};

    function addClass(element, name) {
      if (!element) return;
      if (element.classList) {
        element.classList.add(name);
      } else if ((" " + element.className + " ").indexOf(" " + name + " ") < 0) {
        element.className += " " + name;
      }
    }

    function removeClass(element, name) {
      if (!element) return;
      if (element.classList) {
        element.classList.remove(name);
      } else {
        element.className = (" " + element.className + " ")
          .replace(" " + name + " ", " ")
          .replace(/^\s+|\s+$/g, "");
      }
    }

    function toggleClass(element, name, enabled) {
      if (enabled) addClass(element, name);
      else removeClass(element, name);
    }

    function sideName(value) {
      if (!value) return "";
      return String(value).replace("_hand", "").toUpperCase();
    }

    function clearHands() {
      removeClass(left, "active");
      removeClass(left, "feedback");
      removeClass(right, "active");
      removeClass(right, "feedback");
      addClass(left, "hidden");
      addClass(right, "hidden");
    }

    function markHand(side, cls) {
      if (side === "left" || side === "left_hand") addClass(left, cls);
      if (side === "right" || side === "right_hand") addClass(right, cls);
    }

    function showCenterOnly() {
      addClass(stage, "center-only");
      removeClass(stage, "single-hand");
      addClass(left, "hidden");
      addClass(right, "hidden");
    }

    function showSingleHand(side, cls) {
      var isLeft = side === "left" || side === "left_hand";
      var isRight = side === "right" || side === "right_hand";

      removeClass(stage, "center-only");
      addClass(stage, "single-hand");
      toggleClass(left, "hidden", !isLeft);
      toggleClass(right, "hidden", !isRight);

      if (isLeft && cls) addClass(left, cls);
      if (isRight && cls) addClass(right, cls);
    }

    function showFixation(enabled) {
      toggleClass(fixation, "hidden", !enabled);
    }

    function showCue(enabled) {
      toggleClass(cue, "hidden", !enabled);
    }

    function showFeedback(enabled) {
      toggleClass(feedback, "hidden", !enabled);
    }

    function setImageMode(mode) {
      var promptMode = mode === "prompt";
      var images = [leftImg, rightImg];
      var i;

      toggleClass(document.body, "prompt-mode", promptMode);

      for (i = 0; i < images.length; i += 1) {
        var img = images[i];
        var nextSrc = promptMode
          ? img.getAttribute("data-prompt-src")
          : img.getAttribute("data-imagery-src");

        if (img.getAttribute("src") !== nextSrc) {
          img.style.display = "block";
          img.nextElementSibling.style.display = "block";
          img.setAttribute("src", nextSrc);
        }
      }
    }

    function normalizePhase(value) {
      return String(value || "READY").toUpperCase();
    }

    function isReadyEvent(evt, currentPhase) {
      return (
        evt === "ready" ||
        evt === "initial_wait" ||
        currentPhase === "READY"
      );
    }

    function isRestEvent(evt, currentPhase) {
      return (
        evt === "rest" ||
        currentPhase === "REST"
      );
    }

    function applyReadyView(currentPhase) {
      displayedPhase = currentPhase || "READY";
      phase.textContent = currentPhase || "READY";
      cue.textContent = "";
      feedback.textContent = "";
      clearHands();
      showCenterOnly();
      showFixation(false);
      showCue(false);
      showFeedback(false);
    }

    function applyRestView() {
      displayedPhase = "REST";
      phase.textContent = "REST";
      cue.textContent = "";
      feedback.textContent = "";
      clearHands();
      showCenterOnly();
      showFixation(false);
      showCue(false);
      showFeedback(false);
    }

    function applySingleHandView(currentPhase, side, imageMode, handClass) {
      displayedPhase = currentPhase;
      phase.textContent = currentPhase;
      cue.textContent = "";
      feedback.textContent = "";
      clearHands();
      setImageMode(imageMode);
      showSingleHand(side, handClass);
      showFixation(false);
      showCue(false);
      showFeedback(false);
    }

    function applyOnlineFeedbackView(target, prediction) {
      displayedPhase = "FEEDBACK";
      phase.textContent = "";
      cue.textContent = "";
      feedback.innerHTML =
        "实际指令：" + (sideName(target).toLowerCase() || "-") +
        "<br>预测结果：" + (sideName(prediction).toLowerCase() || "等待中");
      clearHands();
      showCenterOnly();
      showFixation(false);
      showCue(false);
      showFeedback(true);
    }

    function applyEvent(data) {
      var eventTime = Number(data.time || data.server_time || 0);
      if (eventTime && eventTime <= latestEventTime) return;
      if (eventTime) latestEventTime = eventTime;

      var evt = data.event || "";
      var currentPhase = normalizePhase(data.phase || evt);

      if (evt === "experiment_start" || currentPhase === "START") {
        displayedTrial = 0;
        applyReadyView("START");
        return;
      }

      if (isReadyEvent(evt, currentPhase) || evt === "online_ready") {
        setImageMode("prompt");
        applyReadyView("READY");
        return;
      }

      if (evt === "trial_start" || currentPhase === "TRIAL") {
        displayedTrial = Number(data.trial || 0);
        applyReadyView("TRIAL " + (data.trial || ""));
        return;
      }

      if (isRestEvent(evt, currentPhase)) {
        setImageMode("imagery");
        applyRestView();
        return;
      }

      if (evt === "prompt" || currentPhase === "PROMPT") {
        displayedTrial = Number(data.trial || displayedTrial);
        applySingleHandView("PROMPT", data.target, "prompt", "");
        return;
      }

      if (
        evt === "imagery" ||
        evt === "marker_received" ||
        currentPhase === "MOTOR IMAGERY"
      ) {
        displayedTrial = Number(data.trial || displayedTrial);
        applySingleHandView("MOTOR IMAGERY", data.target, "imagery", "active");
        return;
      }

      if (
        evt === "prediction" ||
        evt === "feedback_sent"
      ) {
        var resultTrial = Number(data.trial || 0);
        pendingFeedback[resultTrial] = data;
        if (displayedPhase === "FEEDBACK" && displayedTrial === resultTrial) {
          applyOnlineFeedbackView(
            data.target || data.control,
            data.prediction || data.feedback || data.control
          );
        }
        return;
      }

      if (evt === "feedback_start" || currentPhase === "PREDICTION") {
        displayedTrial = Number(data.trial || displayedTrial);
        var cached = pendingFeedback[displayedTrial] || data;
        applyOnlineFeedbackView(
          cached.target || data.target || cached.control,
          cached.prediction || cached.feedback || cached.control
        );
        return;
      }

      if (evt === "stim_feedback" || currentPhase === "FEEDBACK") {
        displayedTrial = Number(data.trial || displayedTrial);
        var predictedSide = data.prediction || data.feedback || data.control;
        applySingleHandView("FEEDBACK", predictedSide, "prompt", "feedback");
        return;
      }

      if (
        evt === "experiment_stop" ||
        evt === "online_stopped" ||
        currentPhase === "STOP" ||
        currentPhase === "STOPPED"
      ) {
        applyReadyView("STOP");
        return;
      }

      if (evt === "trial_skipped" || currentPhase === "SKIPPED") {
        applyReadyView("TRIAL " + (data.trial || "") + " SKIPPED");
        return;
      }

      applyReadyView(currentPhase || "READY");
    }

    function pollState() {
      var request = new XMLHttpRequest();

      request.onreadystatechange = function () {
        if (request.readyState !== 4) return;
        if (request.status >= 200 && request.status < 300) {
          try {
            applyEvent(JSON.parse(request.responseText));
          } catch (_) {}
        }
      };

      request.open("GET", "/state?_=" + new Date().getTime(), true);
      request.setRequestHeader("Cache-Control", "no-cache");
      request.send(null);
    }

    if (window.EventSource) {
      var events = new EventSource("/events");

      events.addEventListener("metabci", function (event) {
        try {
          applyEvent(JSON.parse(event.data));
        } catch (_) {}
      });
    }

    setInterval(pollState, 500);
    pollState();
  </script>
</body>
</html>
"""


class EventHub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()
        self.latest = {"event": "ready", "phase": "READY", "time": time.time()}
        self.feedback_by_trial = {}

    def subscribe(self):
        client = queue.Queue(maxsize=128)
        with self.lock:
            self.clients.append(client)
            latest = dict(self.latest)
        client.put(latest)
        return client

    def unsubscribe(self, client):
        with self.lock:
            if client in self.clients:
                self.clients.remove(client)

    def publish(self, event):
        event = dict(event)
        event.setdefault("server_time", time.time())
        with self.lock:
            trial_id = int(event.get("trial") or 0)
            if event.get("event") == "feedback_sent" and trial_id > 0:
                self.feedback_by_trial[trial_id] = dict(event)
            elif event.get("event") == "feedback_start" and trial_id > 0:
                cached = self.feedback_by_trial.get(trial_id)
                if cached:
                    for key in (
                        "prediction",
                        "control",
                        "correct",
                        "robot_command",
                    ):
                        if key in cached:
                            event[key] = cached[key]
            self.latest = dict(event)
            clients = list(self.clients)

        for client in clients:
            try:
                client.put_nowait(event)
            except queue.Full:
                pass


def make_handler(hub, quiet=False, asset_dir=None):
    asset_root = Path(asset_dir or DEFAULT_ASSET_DIR).resolve()

    class VRSceneHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            route_path = urlsplit(self.path).path

            if route_path.startswith("/assets/"):
                rel = unquote(route_path[len("/assets/") :]).replace("\\", "/")
                asset_path = (asset_root / rel).resolve()

                if not str(asset_path).startswith(str(asset_root)) or not asset_path.is_file():
                    self.send_error(404)
                    return

                content_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(asset_path.read_bytes())
                return

            if route_path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(PAGE.encode("utf-8"))
                return

            if route_path == "/state":
                data = json.dumps(hub.latest, ensure_ascii=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data)
                return

            if route_path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                client = hub.subscribe()

                try:
                    while True:
                        try:
                            event = client.get(timeout=15)
                            payload = json.dumps(event, ensure_ascii=True, separators=(",", ":"))
                            self.wfile.write(f"event: metabci\ndata: {payload}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except queue.Empty:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    hub.unsubscribe(client)

                return

            self.send_error(404)

        def log_message(self, fmt, *args):
            if not quiet:
                super().log_message(fmt, *args)

    return VRSceneHandler


def udp_loop(hub, host, port, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.5)

    print(f"VR UDP event input: {host}:{port}")

    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue

        try:
            event = json.loads(data.decode("utf-8"))
        except Exception as exc:
            print(f"Ignore invalid UDP event from {addr}: {exc}")
            continue

        event.setdefault("remote", f"{addr[0]}:{addr[1]}")
        hub.publish(event)

    sock.close()


def get_lan_hint():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def get_ipv4_hints():
    hints = set()

    try:
        hints.add(get_lan_hint())
    except OSError:
        pass

    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            hints.add(item[4][0])
    except OSError:
        pass

    hints.discard("127.0.0.1")
    return sorted(hints)


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a browser-based MetaBCI VR rehab scene.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--udp-host", default="0.0.0.0")
    parser.add_argument("--udp-port", type=int, default=8765)
    parser.add_argument("--asset-dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    hub = EventHub()
    stop_event = threading.Event()

    udp_thread = threading.Thread(
        target=udp_loop,
        args=(hub, args.udp_host, args.udp_port, stop_event),
        name="vr-udp-listener",
        daemon=True,
    )
    udp_thread.start()

    handler = make_handler(
        hub,
        quiet=args.quiet,
        asset_dir=args.asset_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"VR scene server: http://{args.host}:{args.port}")

    hints = get_ipv4_hints()
    if hints:
        print("Open one of these URLs on the VR headset if it is on the same network:")
        for hint in hints:
            print(f"  http://{hint}:{args.port}")
    else:
        print("No LAN IPv4 address detected. Run ipconfig and use the PC IPv4 address on the VR network.")

    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping VR scene server...")
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
