from pathlib import Path
from collections import deque
from datetime import datetime
import subprocess
import sys
import threading
import time

import cv2
import joblib
import numpy as np
import serial
import torch
import torch.nn as nn
from torchvision import models, transforms


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

CAMERA_CHECKPOINT = BASE_DIR / "best.pth"
RADAR_MODEL_PATH = BASE_DIR / "live_radar_svm_temporal.pkl"
CALL_SCRIPT = BASE_DIR / "call_care.py"

VIDEO_SOURCE = 0                 # 0 = webcam; or use a video file path
CAMERA_SEQ_LEN = 16
IMG_SIZE = 224
FALL_INDEX = 1
CAMERA_FALL_THRESHOLD = 0.70
CAMERA_CONFIRM_PREDS = 3

RADAR_PORT = "COM3"
RADAR_BAUD_RATE = 115200
RADAR_WINDOW_SIZE = 20
RADAR_CONFIRM_PREDS = 3
RADAR_EXPECTED_FEATURES = 42

GET_UP_SECONDS = 5
SENSOR_STALE_SECONDS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set True only if you want every line printed by call_care.py
SHOW_CALL_SCRIPT_OUTPUT = True


# =========================================================
# 2. NEAT TERMINAL OUTPUT
# =========================================================

def print_header():
    print("\n" + "=" * 64)
    print("                         ELDERGUARD AI")
    print("                         SYSTEM RUNNING")
    print("=" * 64)
    print("AI-Based Elderly Fall Detection and Emergency Alert System")
    print("-" * 64)


def section(title):
    print(f"\n{title}")
    print("-" * 48)


def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# =========================================================
# 3. CAMERA CNN-LSTM MODEL
# Keep this architecture consistent with the trained model.
# =========================================================

class CNNLSTM(nn.Module):
    def __init__(self, hidden=128, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden,
            batch_first=True
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        features = self.cnn(x.flatten(0, 1))
        features = self.pool(features).flatten(1)
        features = features.view(batch_size, seq_len, -1)
        output, _ = self.lstm(features)
        return self.fc(output[:, -1])


def load_camera_model():
    if not CAMERA_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Camera checkpoint not found:\n{CAMERA_CHECKPOINT}"
        )

    model = CNNLSTM()
    checkpoint = torch.load(CAMERA_CHECKPOINT, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        raise KeyError(
            "best.pth does not contain 'model_state_dict'. "
            "Use the checkpoint format from your training code."
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


camera_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# =========================================================
# 4. RADAR SVM MODEL AND FEATURES
# =========================================================

def load_radar_model():
    if not RADAR_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Radar model not found:\n{RADAR_MODEL_PATH}"
        )

    model = joblib.load(RADAR_MODEL_PATH)
    if getattr(model, "n_features_in_", None) != RADAR_EXPECTED_FEATURES:
        raise ValueError("Radar model must accept exactly 42 features.")
    return model


def parse_radar_line(line):
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 6:
        return None

    try:
        values = np.asarray(parts, dtype=float)
    except ValueError:
        return None

    if not np.all(np.isfinite(values)):
        return None
    return values


def extract_radar_features(values):
    if values.shape != (RADAR_WINDOW_SIZE, 6):
        raise ValueError(
            f"Expected ({RADAR_WINDOW_SIZE}, 6), got {values.shape}"
        )

    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    mean_abs_change = np.abs(np.diff(values, axis=0)).mean(axis=0)
    start_to_end = values[-1] - values[0]

    sample_indices = np.arange(RADAR_WINDOW_SIZE, dtype=float)
    centered_indices = sample_indices - sample_indices.mean()
    slopes = np.array([
        np.polyfit(centered_indices, values[:, j], 1)[0]
        for j in range(6)
    ])

    features = np.concatenate([
        means, stds, mins, maxs,
        mean_abs_change, start_to_end, slopes
    ])

    if features.size != RADAR_EXPECTED_FEATURES:
        raise ValueError(f"Expected 42 features, got {features.size}")

    return features.reshape(1, -1)


# =========================================================
# 5. SHARED SENSOR / EVENT STATE
# =========================================================

lock = threading.Lock()
stop_event = threading.Event()

sensor_state = {"camera": None, "radar": None}
sensor_last_update = {"camera": None, "radar": None}
fall_hits = {"camera": 0, "radar": 0}
last_printed_sensor_state = {"camera": None, "radar": None}

event = {
    "active": False,
    "start": None,
    "wallclock": None,
    "both_normal_since": None,
    "call_process": None,
    "call_finished_reported": False,
}


def show_sensor_change(sensor_name, prediction, detail=""):
    label = "FALL DETECTED" if prediction == 1 else "NORMAL"
    with lock:
        changed = last_printed_sensor_state[sensor_name] != prediction
        last_printed_sensor_state[sensor_name] = prediction

    if changed:
        suffix = f" | {detail}" if detail else ""
        log(f"{sensor_name.upper():<7} STATUS: {label}{suffix}")


def start_caregiver_process():
    if not CALL_SCRIPT.is_file():
        log(f"ERROR: caregiver script not found: {CALL_SCRIPT}")
        return None

    try:
        stdout_target = None if SHOW_CALL_SCRIPT_OUTPUT else subprocess.DEVNULL
        process = subprocess.Popen(
            [sys.executable, "-u", str(CALL_SCRIPT)],
            cwd=str(BASE_DIR),
            stdout=stdout_target,
            stderr=stdout_target
        )
        log("Caregiver calling process started.")
        return process
    except Exception as exc:
        log(f"Could not start caregiver calling process: {exc}")
        return None


def update_sensor(sensor_name, prediction, detail=""):
    should_start_alert = False

    with lock:
        sensor_last_update[sensor_name] = time.monotonic()
        sensor_state[sensor_name] = prediction

        if prediction == 1:
            fall_hits[sensor_name] += 1
        else:
            fall_hits[sensor_name] = 0

        required_hits = (
            CAMERA_CONFIRM_PREDS if sensor_name == "camera"
            else RADAR_CONFIRM_PREDS
        )
        confirmed = fall_hits[sensor_name] >= required_hits

        if confirmed and not event["active"]:
            event["active"] = True
            event["start"] = time.monotonic()
            event["wallclock"] = datetime.now()
            event["both_normal_since"] = None
            event["call_finished_reported"] = False
            should_start_alert = True

    show_sensor_change(sensor_name, prediction, detail)

    if should_start_alert:
        with lock:
            event_time = event["wallclock"]

        print("\n" + "!" * 64)
        print("                 EMERGENCY EVENT CONFIRMED")
        print("!" * 64)
        log(f"Fall confirmed by {sensor_name.upper()}.")
        log(f"Event time: {event_time.strftime('%d-%m-%Y %H:%M:%S')}")
        section("CAREGIVER CALL SEQUENCE")

        process = start_caregiver_process()
        with lock:
            event["call_process"] = process


# =========================================================
# 6. RECOVERY MONITOR
# Recovery is not allowed to close the event while the
# caregiver calling script is still running.
# =========================================================

def alert_monitor():
    while not stop_event.wait(0.25):
        with lock:
            if not event["active"]:
                continue

            now = time.monotonic()

            both_normal_and_fresh = all(
                sensor_state[name] == 0
                and sensor_last_update[name] is not None
                and now - sensor_last_update[name] <= SENSOR_STALE_SECONDS
                for name in ("camera", "radar")
            )

            call_process = event["call_process"]
            call_finished = (
                call_process is None or call_process.poll() is not None
            )

            if call_finished and not event["call_finished_reported"]:
                event["call_finished_reported"] = True
                call_exit_code = (
                    None if call_process is None else call_process.returncode
                )
                report_call_finished = True
            else:
                report_call_finished = False
                call_exit_code = None

            if both_normal_and_fresh:
                if event["both_normal_since"] is None:
                    event["both_normal_since"] = now
            else:
                event["both_normal_since"] = None

            recovery_elapsed = (
                0 if event["both_normal_since"] is None
                else now - event["both_normal_since"]
            )

            can_reset = (
                both_normal_and_fresh
                and recovery_elapsed >= GET_UP_SECONDS
                and call_finished
            )

        if report_call_finished:
            if call_exit_code == 0:
                log("Caregiver calling process completed.")
            elif call_exit_code is None:
                log("Caregiver calling process was not started.")
            else:
                log(
                    "Caregiver calling process ended "
                    f"with exit code {call_exit_code}."
                )

        if can_reset:
            print("\n" + "-" * 64)
            section("RECOVERY MONITORING")
            log("Both sensors report NORMAL.")
            log("NORMAL condition maintained for 5 seconds.")
            log("Caregiver calling process has finished.")
            log("Monitoring event reset.")
            print("-" * 64 + "\n")

            with lock:
                event["active"] = False
                event["start"] = None
                event["wallclock"] = None
                event["both_normal_since"] = None
                event["call_process"] = None
                event["call_finished_reported"] = False
                fall_hits["camera"] = 0
                fall_hits["radar"] = 0


# =========================================================
# 7. RADAR WORKER
# =========================================================

def radar_worker(radar_model):
    rows = deque(maxlen=RADAR_WINDOW_SIZE)

    try:
        with serial.Serial(
            RADAR_PORT,
            RADAR_BAUD_RATE,
            timeout=1
        ) as ser:
            time.sleep(2)
            ser.reset_input_buffer()
            log(f"ESP32 connected on {RADAR_PORT}.")

            while not stop_event.is_set():
                raw_line = ser.readline().decode(
                    "utf-8", errors="ignore"
                ).strip()

                if not raw_line:
                    continue

                values = parse_radar_line(raw_line)
                if values is None:
                    continue

                rows.append(values)
                if len(rows) < RADAR_WINDOW_SIZE:
                    continue

                features = extract_radar_features(
                    np.asarray(rows, dtype=float)
                )
                prediction = int(radar_model.predict(features)[0])

                if prediction in (0, 1):
                    update_sensor("radar", prediction)

    except serial.SerialException as exc:
        log(f"RADAR SERIAL ERROR: {exc}")
        log("Check COM3 and close Arduino IDE Serial Monitor.")
    except Exception as exc:
        log(f"RADAR WORKER ERROR: {exc!r}")


# =========================================================
# 8. CAMERA LOOP
# =========================================================

def run_camera(camera_model):
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera/video: {VIDEO_SOURCE}")

    frame_buffer = deque(maxlen=CAMERA_SEQ_LEN)

    try:
        log("Camera/video opened.")

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                log("Video ended or camera frame unavailable.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(camera_transform(rgb))

            label = "Collecting camera frames"
            color = (220, 220, 220)

            if len(frame_buffer) == CAMERA_SEQ_LEN:
                clip = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    probabilities = torch.softmax(camera_model(clip), dim=1)[0]

                p_fall = probabilities[FALL_INDEX].item()
                prediction = int(p_fall >= CAMERA_FALL_THRESHOLD)

                update_sensor(
                    "camera",
                    prediction,
                    detail=f"fall confidence {p_fall:.2f}"
                )

                label = (
                    f"CAMERA FALL {p_fall:.2f}"
                    if prediction == 1
                    else f"CAMERA NORMAL {1 - p_fall:.2f}"
                )
                color = (0, 0, 255) if prediction == 1 else (0, 200, 0)

            with lock:
                event_active = event["active"]

            if event_active:
                label = "ELDERGUARD: EMERGENCY EVENT ACTIVE"
                color = (0, 0, 255)

            cv2.putText(
                frame, label, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
            )
            cv2.imshow("ElderGuard AI - Camera + Radar", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log("Stop requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# 9. MAIN APPLICATION
# =========================================================

def main():
    print_header()
    print(f"Device       : {DEVICE}")
    print(f"Camera input : {VIDEO_SOURCE}")
    print(f"Radar port   : {RADAR_PORT}")
    print("Fall rule    : Either camera OR radar can confirm a fall")

    section("SYSTEM INITIALIZATION")

    log("Loading camera CNN-LSTM model...")
    camera_model = load_camera_model()
    log("Camera CNN-LSTM model loaded successfully.")

    log("Loading radar SVM model...")
    radar_model = load_radar_model()
    log("Radar SVM model loaded successfully.")

    log("Models ready.")

    section("LIVE MONITORING")
    log("Camera status: waiting for frames.")
    log("Radar status : waiting for ESP32 data.")
    log("Press Q in the camera window to stop monitoring.")

    monitor_thread = threading.Thread(
        target=alert_monitor,
        daemon=True
    )
    radar_thread = threading.Thread(
        target=radar_worker,
        args=(radar_model,),
        daemon=True
    )

    monitor_thread.start()
    radar_thread.start()

    try:
        run_camera(camera_model)
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        radar_thread.join(timeout=2)
        monitor_thread.join(timeout=2)
        log("ElderGuard AI stopped.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nMonitoring interrupted by user.")
    except Exception as error:
        stop_event.set()
        print(f"\nSYSTEM ERROR: {error}")
        raise
