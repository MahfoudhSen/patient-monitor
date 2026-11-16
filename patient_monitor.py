#!/usr/bin/env python3
"""
Patient Stand-Up / Bed-Exit / Fall Monitor  (PROTOTYPE - not a medical device)

Runs 100% locally: camera -> MediaPipe pose -> posture state machine -> alerts.
Optional: a free local vision LLM (Ollama + moondream/llava) double-checks alerts
to cut false alarms. No video ever leaves the machine.

Usage:
    python patient_monitor.py --room "ER Bay 12"
    python patient_monitor.py --camera 1 --ntfy-topic my-er-alerts --llm moondream
    python patient_monitor.py --video test.mp4          # test on a recorded clip

Keys while running:
    b = draw bed zone (drag a box, press ENTER)   c = clear bed zone
    p = toggle privacy blur                       m = mute/unmute sound
    q = quit
"""
import argparse
import base64
import csv
import math
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import requests
from mediapipe.tasks.python import BaseOptions, vision

# Pose model files (downloaded automatically on first run, ~5-30 MB)
MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


class L:  # MediaPipe pose landmark indices
    NOSE = 0
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28


SKELETON = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


def load_pose_model(complexity):
    url = MODEL_URLS[complexity]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(url))
    if not os.path.exists(path):
        print(f"Downloading pose model -> {path}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    return vision.PoseLandmarker.create_from_options(opts)

# ----------------------------------------------------------------------------- geometry

def angle_3pt(a, b, c):
    """Angle ABC in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def torso_tilt(shoulder_mid, hip_mid):
    """0 deg = perfectly upright, 90 deg = horizontal (lying)."""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]          # image y grows downward
    return abs(math.degrees(math.atan2(dx, dy)))


def classify_posture(lm, w, h, min_vis=0.5):
    """Return (posture, info_dict). posture in LYING / SITTING / STANDING / UNKNOWN."""
    def p(idx):
        k = lm[idx]
        return (k.x * w, k.y * h), (k.visibility or 0.0)

    (ls, vls), (rs, vrs) = p(L.LEFT_SHOULDER), p(L.RIGHT_SHOULDER)
    (lh, vlh), (rh, vrh) = p(L.LEFT_HIP), p(L.RIGHT_HIP)
    (lk, vlk), (rk, vrk) = p(L.LEFT_KNEE), p(L.RIGHT_KNEE)
    (la, vla), (ra, vra) = p(L.LEFT_ANKLE), p(L.RIGHT_ANKLE)

    if min(vls, vrs, vlh, vrh) < min_vis:
        return "UNKNOWN", {}

    sh_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
    hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    tilt = torso_tilt(sh_mid, hip_mid)
    info = {"tilt": tilt, "hip": hip_mid, "shoulder": sh_mid}

    if tilt > 55:
        return "LYING", info

    # Upright torso: decide sitting vs standing using knee angles (legs straight = standing)
    knee_angles = []
    if vlh > min_vis and vlk > min_vis and vla > min_vis:
        knee_angles.append(angle_3pt(lh, lk, la))
    if vrh > min_vis and vrk > min_vis and vra > min_vis:
        knee_angles.append(angle_3pt(rh, rk, ra))

    if knee_angles:
        knee = max(knee_angles)
        info["knee"] = knee
        if knee > 150:
            return "STANDING", info
        if knee < 125:
            return "SITTING", info
        return "SITTING", info  # in-between: treat as sitting / rising

    # Legs hidden (blanket, bed rail): upright torso with hidden legs ~ sitting up
    return "SITTING", info


# ----------------------------------------------------------------------------- alerts

class Alerter:
    def __init__(self, room, ntfy_topic=None, ntfy_server="https://ntfy.sh",
                 webhook=None, cooldown=30, log_path="alerts_log.csv"):
        self.room = room
        self.ntfy_topic = ntfy_topic
        self.ntfy_server = ntfy_server.rstrip("/")
        self.webhook = webhook
        self.cooldown = cooldown
        self.last_sent = {}
        self.muted = False
        self.log_path = log_path
        if not os.path.exists(log_path):
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "room", "event", "detail", "llm_check"])

    def _beep(self):
        if self.muted:
            return
        try:
            if sys.platform.startswith("win"):
                import winsound
                for _ in range(3):
                    winsound.Beep(1500, 250)
            elif sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
            else:
                print("\a", end="", flush=True)
                os.system("(paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga "
                          "|| aplay -q /usr/share/sounds/alsa/Front_Center.wav) >/dev/null 2>&1 &")
        except Exception:
            pass

    def send(self, event, detail="", llm_check="", priority="high"):
        now = time.time()
        if now - self.last_sent.get(event, 0) < self.cooldown:
            return False
        self.last_sent[event] = now
        msg = f"[{self.room}] {event}" + (f" - {detail}" if detail else "")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n*** ALERT {stamp}: {msg}  {('LLM: ' + llm_check) if llm_check else ''}")

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([stamp, self.room, event, detail, llm_check])

        self._beep()
        # Text only - NO images or patient identifiers are ever sent off the machine.
        threading.Thread(target=self._push, args=(event, msg, priority), daemon=True).start()
        return True

    def _push(self, event, msg, priority):
        try:
            if self.ntfy_topic:
                requests.post(f"{self.ntfy_server}/{self.ntfy_topic}", data=msg.encode(),
                              headers={"Title": f"Patient alert: {event}",
                                       "Priority": "urgent" if event == "FALL DETECTED" else priority,
                                       "Tags": "rotating_light"}, timeout=5)
            if self.webhook:
                requests.post(self.webhook, json={"room": self.room, "event": event, "text": msg},
                              timeout=5)
        except Exception as e:
            print(f"  (push failed: {e})")


# ----------------------------------------------------------------------------- optional local LLM

class LLMVerifier:
    """Asks a free local vision model (Ollama) what the patient is doing. Runs in background."""

    PROMPT = ("You are monitoring a hospital patient for safety. Look at the image and answer with "
              "exactly ONE word from this list: LYING, SITTING, STANDING, FLOOR, NOBODY. "
              "FLOOR means the person is on the ground / has fallen.")

    def __init__(self, model, url="http://localhost:11434"):
        self.model = model
        self.url = url.rstrip("/") + "/api/generate"
        self.ok = self._check()

    def _check(self):
        try:
            requests.get(self.url.replace("/api/generate", "/api/tags"), timeout=3)
            return True
        except Exception:
            print(f"[LLM] Ollama not reachable at {self.url}; continuing without LLM check.")
            return False

    def ask(self, frame, callback):
        if not self.ok:
            callback("")
            return

        def run():
            try:
                small = cv2.resize(frame, (448, int(448 * frame.shape[0] / frame.shape[1])))
                _, buf = cv2.imencode(".jpg", small)
                r = requests.post(self.url, json={
                    "model": self.model, "prompt": self.PROMPT, "stream": False,
                    "images": [base64.b64encode(buf).decode()],
                    "options": {"temperature": 0}}, timeout=30)
                ans = r.json().get("response", "").strip().upper()
                for word in ("FLOOR", "STANDING", "SITTING", "LYING", "NOBODY"):
                    if word in ans:
                        callback(word)
                        return
                callback(ans[:20])
            except Exception as e:
                print(f"[LLM] error: {e}")
                callback("")

        threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------------- monitor

class PatientMonitor:
    def __init__(self, args):
        self.args = args
        src = args.video if args.video else args.camera
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            sys.exit(f"Could not open camera/video: {src}")

        self.pose = load_pose_model(args.model_complexity)
        self.t0 = time.time()
        self.alerter = Alerter(args.room, args.ntfy_topic, args.ntfy_server, args.webhook,
                               args.cooldown)
        self.llm = LLMVerifier(args.llm, args.ollama_url) if args.llm else None

        self.bed_zone = None            # (x, y, w, h)
        self.privacy = args.privacy
        self.history = deque(maxlen=int(args.smooth_frames))   # recent posture labels
        self.hip_track = deque(maxlen=60)                     # (t, hip_y_norm, posture)
        self.state = "UNKNOWN"
        self.prev_state = "UNKNOWN"
        self.banner = ("", 0)            # (text, until_time)
        self.no_person_since = None

    # -- helpers
    def _stable_state(self, posture):
        self.history.append(posture)
        known = [p for p in self.history if p != "UNKNOWN"]
        if len(known) < max(3, len(self.history) // 2):
            return self.state
        winner = max(set(known), key=known.count)
        return winner if known.count(winner) >= 0.6 * len(known) else self.state

    def _in_bed(self, pt):
        if not self.bed_zone:
            return True
        x, y, w, h = self.bed_zone
        return x <= pt[0] <= x + w and y <= pt[1] <= y + h

    def _alert(self, event, detail, frame):
        def fire(llm_answer):
            # If the LLM clearly disagrees, downgrade (still logged) to reduce false alarms.
            if llm_answer and self.args.llm_veto:
                agree = {
                    "PATIENT STANDING UP": {"STANDING"},
                    "PATIENT SITTING UP": {"SITTING", "STANDING"},
                    "LEFT BED ZONE": {"STANDING", "SITTING", "FLOOR", "NOBODY"},
                    "FALL DETECTED": {"FLOOR", "LYING"},
                    "PATIENT OUT OF VIEW": {"NOBODY", "STANDING", "FLOOR"},
                }.get(event)
                if agree and llm_answer not in agree:
                    print(f"[LLM] vetoed '{event}' (model saw {llm_answer})")
                    return
            if self.alerter.send(event, detail, llm_answer):
                self.banner = (f"{event}", time.time() + 8)

        if self.llm and self.llm.ok and event != "FALL DETECTED":
            self.llm.ask(frame.copy(), fire)
        else:
            fire("")   # falls always alert immediately; never wait on the LLM

    # -- main loop
    def run(self):
        print(f"Monitoring '{self.args.room}'. Press b=bed zone, p=privacy, m=mute, q=quit.")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.args.video:
                    break
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            now = time.time()
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = self.pose.detect_for_video(img, int((now - self.t0) * 1000))
            landmarks = res.pose_landmarks[0] if res.pose_landmarks else None
            posture, info = "UNKNOWN", {}

            if landmarks:
                self.no_person_since = None
                posture, info = classify_posture(landmarks, w, h)
            else:
                self.no_person_since = self.no_person_since or now

            self.prev_state, self.state = self.state, self._stable_state(posture)

            # ---- fall detection: fast hip drop followed by lying/low posture
            if info.get("hip"):
                self.hip_track.append((now, info["hip"][1] / h, posture))
                recent = [(t, y) for t, y, _ in self.hip_track if now - t <= 1.0]
                if len(recent) >= 4:
                    drop = recent[-1][1] - min(y for _, y in recent)
                    was_up = any(p in ("STANDING", "SITTING") for t, _, p in self.hip_track
                                 if now - t <= 2.0)
                    on_floor_level = info["hip"][1] / h > self.args.floor_line
                    out_of_bed = not self._in_bed(info["hip"])
                    if drop > self.args.fall_drop and was_up and posture == "LYING" \
                            and (out_of_bed or on_floor_level or not self.bed_zone):
                        self._alert("FALL DETECTED", "rapid drop to ground", frame)

            # ---- transitions
            if self.state != self.prev_state:
                if self.state == "STANDING":
                    self._alert("PATIENT STANDING UP", "", frame)
                elif self.state == "SITTING" and self.prev_state == "LYING" and self.args.alert_sitting:
                    self._alert("PATIENT SITTING UP", "early warning", frame)

            if info.get("hip") and self.bed_zone and not self._in_bed(info["hip"]) \
                    and self.state in ("STANDING", "SITTING"):
                self._alert("LEFT BED ZONE", "", frame)

            if self.no_person_since and now - self.no_person_since > self.args.absent_secs \
                    and self.prev_state != "UNKNOWN":
                self._alert("PATIENT OUT OF VIEW", f">{self.args.absent_secs}s", frame)

            if not self.args.headless:
                self._draw(frame, landmarks, info, now)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    roi = cv2.selectROI("Draw bed zone - ENTER to confirm", frame, False)
                    cv2.destroyWindow("Draw bed zone - ENTER to confirm")
                    self.bed_zone = roi if roi[2] > 0 else None
                if key == ord("c"):
                    self.bed_zone = None
                if key == ord("p"):
                    self.privacy = not self.privacy
                if key == ord("m"):
                    self.alerter.muted = not self.alerter.muted

        self.cap.release()
        cv2.destroyAllWindows()

    def _draw(self, frame, landmarks, info, now):
        h, w = frame.shape[:2]
        if self.privacy:   # blur the whole scene, keep the skeleton visible
            frame[:] = cv2.GaussianBlur(frame, (0, 0), 25)
        if landmarks:
            pts = [(int(k.x * w), int(k.y * h), (k.visibility or 0)) for k in landmarks]
            for a, b in SKELETON:
                if pts[a][2] > 0.5 and pts[b][2] > 0.5:
                    cv2.line(frame, pts[a][:2], pts[b][:2], (255, 255, 255), 3)
            for x, y, v in pts[11:29]:
                if v > 0.5:
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        if self.bed_zone:
            x, y, bw, bh = self.bed_zone
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 200, 0), 2)
            cv2.putText(frame, "BED", (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        fy = int(self.args.floor_line * h)
        cv2.line(frame, (0, fy), (w, fy), (80, 80, 80), 1)

        color = {"LYING": (0, 200, 0), "SITTING": (0, 200, 255),
                 "STANDING": (0, 0, 255)}.get(self.state, (200, 200, 200))
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        txt = f"{self.args.room} | {self.state}"
        if "tilt" in info:
            txt += f" | torso {info['tilt']:.0f}deg"
        if "knee" in info:
            txt += f" | knee {info['knee']:.0f}deg"
        if self.alerter.muted:
            txt += " | MUTED"
        cv2.putText(frame, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        text, until = self.banner
        if text and now < until and int(now * 3) % 2 == 0:
            cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 255), -1)
            cv2.putText(frame, f"ALERT: {text}", (15, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (255, 255, 255), 3)
        cv2.imshow("Patient Monitor", frame)


def main():
    ap = argparse.ArgumentParser(description="Local AI patient stand-up / fall monitor")
    ap.add_argument("--room", default="Room 1", help="label shown in alerts (no patient names!)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--video", help="use a video file instead of a camera (for testing)")
    ap.add_argument("--ntfy-topic", help="ntfy topic for phone push alerts")
    ap.add_argument("--ntfy-server", default="https://ntfy.sh",
                    help="use your hospital's self-hosted ntfy server in production")
    ap.add_argument("--webhook", help="POST JSON alerts here (Teams/Slack/nurse-call bridge)")
    ap.add_argument("--llm", help="Ollama vision model to verify alerts, e.g. moondream or llava")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--llm-veto", action="store_true",
                    help="suppress an alert if the LLM clearly disagrees (falls are never vetoed)")
    ap.add_argument("--alert-sitting", action="store_true", default=True,
                    help="early warning when patient sits up from lying (default on)")
    ap.add_argument("--no-alert-sitting", dest="alert_sitting", action="store_false")
    ap.add_argument("--cooldown", type=int, default=30, help="seconds between repeat alerts")
    ap.add_argument("--absent-secs", type=int, default=20)
    ap.add_argument("--fall-drop", type=float, default=0.18,
                    help="hip drop (fraction of frame height) within 1s that counts as a fall")
    ap.add_argument("--floor-line", type=float, default=0.80,
                    help="hips below this fraction of frame height = floor level")
    ap.add_argument("--smooth-frames", type=int, default=10)
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--privacy", action="store_true", help="start with video blurred")
    ap.add_argument("--headless", action="store_true", help="no window (server/kiosk mode)")
    PatientMonitor(ap.parse_args()).run()


if __name__ == "__main__":
    main()
