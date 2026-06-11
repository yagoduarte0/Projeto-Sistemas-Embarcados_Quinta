# -*- coding: utf-8 -*-
"""
Dashboard web em tempo real para a deteccao de sonolencia do motorista.

Mostra, ao vivo:
  - video da camera com os olhos marcados (verde=aberto, vermelho=fechado)
  - pontuacao de sonolencia (0-100) com medidor e grafico historico
  - nivel de alerta (NORMAL / ATENCAO / ALERTA / PERIGO)
  - EAR, PERCLOS, FPS, n. de bocejos, tempo de sessao
  - botoes: recalibrar e ligar/desligar som

Arquitetura (Flask puro, sem socketio):
  - uma thread de fundo (DrowsinessEngine) le a camera e processa cada frame,
    guardando o ultimo JPEG anotado + as metricas sob um lock;
  - /video_feed  -> stream MJPEG do ultimo frame;
  - /metrics     -> JSON com as metricas (a pagina faz polling ~6x/s);
  - /recalibrate, /toggle_sound -> acoes dos botoes.

Reaproveita toda a logica de drowsiness_detector.py.

Uso:
    pip install -r requirements.txt
    python dashboard.py            # camera 0
    python dashboard.py 1          # outra camera (ex.: a do retrovisor)
  depois abra http://127.0.0.1:5000 no navegador.
"""

import sys
import time
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template

import drowsiness_detector as dd  # reusa constantes, geometria e alarme


# ----------------------------------------------------------------------------
# MOTOR DE PROCESSAMENTO (roda numa thread de fundo)
# ----------------------------------------------------------------------------
class DrowsinessEngine:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._jpeg = None              # ultimo frame anotado (bytes JPEG)
        self._metrics = self._empty_metrics()
        self._stop = False

        # estado da deteccao
        self._score = 0.0
        self._ear_open = None
        self._calibrating = True
        self._calib_start = None
        self._calib_samples = []
        self._perclos_buf = deque()
        self._yawn_active = False
        self._yawn_count = 0
        self._session_start = None
        self._recalib_request = False
        self._paused = False

        # Registro p/ avaliacao do artigo (Metodo A=EAR, Metodo B=blendshape)
        self._recording = False
        self._logger = None
        self._ground_truth = False
        self._blink = 0.0
        self._ear_ratio = 1.0

        # Atencao/distracao e fadiga
        self._headpose = dd.HeadPoseMonitor()
        self._fatigue = dd.FatigueMeters()
        self._headpose_samples = []
        self._yaw_dev = 0.0
        self._pitch_dev = 0.0
        self._distracted = False

        self.alarm = dd.AlarmManager()
        self._thread = threading.Thread(target=self._run, daemon=True)

    # ---- API publica ----
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self.alarm.stop()
        if self._logger is not None:
            self._logger.close()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_metrics(self):
        with self._lock:
            return dict(self._metrics)

    def request_recalibration(self):
        self._recalib_request = True

    def toggle_sound(self):
        return self.alarm.toggle()

    def toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.alarm.set_level(0)   # silencia ao pausar
        return self._paused

    def toggle_record(self):
        self._recording = not self._recording
        if self._recording and self._logger is None:
            self._logger = dd.CsvLogger()
        return {"recording": self._recording,
                "path": self._logger.path if self._logger else None}

    def toggle_truth(self):
        self._ground_truth = not self._ground_truth
        return self._ground_truth

    # ---- interno ----
    @staticmethod
    def _empty_metrics():
        return {
            "score": 0.0, "level": 0, "level_label": "INICIANDO",
            "state": "...", "ear": 0.0, "blink": 0.0, "perclos": 0.0, "yawns": 0,
            "fps": 0.0, "sound": True, "face": False,
            "calibrating": True, "recording": False, "ground_truth": False,
            "paused": False, "distracted": False, "yaw": 0, "microsleep": 0,
            "microsleep_active": False, "blink_rate": 0, "latency": None,
            "elapsed": 0.0,
        }

    def _run(self):
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            with self._lock:
                self._metrics["level_label"] = "ERRO: SEM CAMERA"
            return

        landmarker = dd.create_landmarker()

        self._session_start = time.time()
        self._calib_start = time.time()
        start_t = self._session_start
        prev_t = start_t
        last_ts = -1

        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            dt = now - prev_t
            prev_t = now
            fps = 1.0 / dt if dt > 0 else 0.0

            # ---------- PAUSA: congela deteccao, silencia, mostra PAUSADO ----------
            if self._paused:
                self.alarm.set_level(0)
                pframe = frame.copy()
                ov = pframe.copy()
                cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(ov, 0.5, pframe, 0.5, 0, pframe)
                cv2.putText(pframe, "PAUSADO", (w // 2 - 170, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
                ok_jpg, buf = cv2.imencode(".jpg", pframe,
                                           [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self._lock:
                    if ok_jpg:
                        self._jpeg = buf.tobytes()
                    self._metrics["paused"] = True
                    self._metrics["level"] = 0
                    self._metrics["level_label"] = "PAUSADO"
                time.sleep(0.03)
                continue

            if self._recalib_request:
                self._calibrating = True
                self._calib_start = now
                self._calib_samples = []
                self._score = 0.0
                self._recalib_request = False

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = max(last_ts + 1, int((now - start_t) * 1000))
            last_ts = ts_ms
            res = landmarker.detect_for_video(mp_image, ts_ms)

            ear = 0.0
            state = "SEM ROSTO"
            perclos = 0.0
            yawning = False
            face = bool(res.face_landmarks)

            if face:
                pts = dd.landmarks_to_array(res.face_landmarks[0], w, h)
                ear = (dd.eye_aspect_ratio(pts, dd.LEFT_EYE) +
                       dd.eye_aspect_ratio(pts, dd.RIGHT_EYE)) / 2.0
                mar = dd.mouth_aspect_ratio(pts, dd.MOUTH)
                self._blink = (dd.blink_score(res.face_blendshapes[0])
                               if res.face_blendshapes else 0.0)
                yaw = pitch = 0.0
                if res.facial_transformation_matrixes:
                    yaw, pitch, _ = dd.head_euler_angles(
                        res.facial_transformation_matrixes[0])

                if self._calibrating:
                    self._calib_samples.append(ear)
                    self._headpose_samples.append((yaw, pitch))
                    if now - self._calib_start >= dd.CALIB_SECONDS:
                        arr = np.array(self._calib_samples)
                        self._ear_open = float(
                            np.median(arr[arr > np.percentile(arr, 40)]))
                        hp = np.array(self._headpose_samples)
                        self._headpose.set_baseline(float(np.median(hp[:, 0])),
                                                    float(np.median(hp[:, 1])))
                        self._calibrating = False
                    state = "CALIBRANDO"
                    closed_flag = False
                else:
                    r = ear / self._ear_open if self._ear_open else 1.0
                    self._ear_ratio = r
                    if r < dd.RATIO_CLOSED:
                        state, closed_flag = "OLHOS FECHADOS", True
                        self._score += dd.SCORE_RATE_CLOSED * dt
                    elif r < dd.RATIO_HALF:
                        state, closed_flag = "SEMICERRADO", True
                        self._score += dd.SCORE_RATE_HALF * dt
                    else:
                        state, closed_flag = "OLHOS ABERTOS", False
                        self._score -= dd.SCORE_RECOVERY * dt

                    if mar > dd.MAR_YAWN:
                        yawning = True
                        if not self._yawn_active:
                            self._score += dd.SCORE_YAWN_BUMP
                            self._yawn_count += 1
                            self._yawn_active = True
                    else:
                        self._yawn_active = False

                # PERCLOS + distracao + fadiga (so apos calibrar)
                if not self._calibrating:
                    self._perclos_buf.append((now, closed_flag))
                    while self._perclos_buf and now - self._perclos_buf[0][0] > dd.PERCLOS_WINDOW:
                        self._perclos_buf.popleft()
                    if self._perclos_buf:
                        perclos = sum(1 for _, c in self._perclos_buf if c) / len(self._perclos_buf)

                    self._yaw_dev, self._pitch_dev, self._distracted = \
                        self._headpose.update(now, yaw, pitch)
                    if self._distracted:
                        self._score += dd.SCORE_RATE_DISTRACTED * dt
                        state = "DISTRAIDO"
                    self._fatigue.update(now, closed_flag)

                self._draw_eyes(frame, pts, state)
            else:
                # sem rosto = possivel cabeca baixa
                if not self._calibrating:
                    self._score += dd.SCORE_RATE_HALF * dt * 0.5

            self._score = float(np.clip(self._score, 0.0, 100.0))
            if self._calibrating:
                level = 0
            elif self._score >= dd.LEVEL_DANGER:
                level = 3
            elif self._score >= dd.LEVEL_ALERT:
                level = 2
            elif self._score >= dd.LEVEL_ATTENTION:
                level = 1
            else:
                level = 0
            self.alarm.set_level(level)
            if level >= 1:
                self._fatigue.register_alert(now)

            label = "CALIBRANDO" if self._calibrating else dd.LEVEL_LABELS[level]

            # registro em CSV (so apos calibrar)
            if self._recording and self._logger is not None and not self._calibrating:
                self._logger.log(now - self._session_start, ear, self._ear_ratio,
                                 self._blink, state, self._score, level, perclos,
                                 self._yawn_count, self._yaw_dev, self._pitch_dev,
                                 self._distracted, self._fatigue.microsleep_count,
                                 self._fatigue.blink_rate, self._ground_truth)

            ok_jpg, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                if ok_jpg:
                    self._jpeg = buf.tobytes()
                self._metrics = {
                    "score": round(self._score, 1),
                    "level": level,
                    "level_label": label,
                    "state": state,
                    "ear": round(ear, 3),
                    "blink": round(self._blink, 3),
                    "perclos": round(perclos * 100, 1),
                    "yawns": self._yawn_count,
                    "fps": round(fps, 1),
                    "sound": self.alarm._enabled,
                    "face": face,
                    "calibrating": self._calibrating,
                    "recording": self._recording,
                    "ground_truth": self._ground_truth,
                    "paused": False,
                    "distracted": self._distracted,
                    "yaw": round(self._yaw_dev, 0),
                    "microsleep": self._fatigue.microsleep_count,
                    "microsleep_active": self._fatigue.microsleep_active,
                    "blink_rate": round(self._fatigue.blink_rate, 0),
                    "latency": (round(self._fatigue.avg_latency, 2)
                                if self._fatigue.avg_latency is not None else None),
                    "elapsed": round(now - self._session_start, 1),
                }

        cap.release()
        landmarker.close()

    @staticmethod
    def _draw_eyes(frame, pts, state):
        color = (0, 0, 255) if "FECHAD" in state or "SEMI" in state else (0, 220, 0)
        for idx in (dd.LEFT_EYE, dd.RIGHT_EYE):
            poly = pts[idx].astype(np.int32)
            cv2.polylines(frame, [poly], True, color, 2)
        for idx in dd.MOUTH:
            cv2.circle(frame, tuple(pts[idx].astype(int)), 2, (255, 200, 0), -1)


# ----------------------------------------------------------------------------
# APP FLASK
# ----------------------------------------------------------------------------
app = Flask(__name__)
engine = None  # criado no main


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/metrics")
def metrics():
    return jsonify(engine.get_metrics())


@app.route("/video_feed")
def video_feed():
    def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpg = engine.get_jpeg()
            if jpg is not None:
                yield boundary + jpg + b"\r\n"
            time.sleep(0.03)  # ~30 fps maximo
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/recalibrate", methods=["POST"])
def recalibrate():
    engine.request_recalibration()
    return jsonify({"ok": True})


@app.route("/toggle_pause", methods=["POST"])
def toggle_pause():
    return jsonify({"paused": engine.toggle_pause()})


@app.route("/toggle_sound", methods=["POST"])
def toggle_sound():
    return jsonify({"sound": engine.toggle_sound()})


@app.route("/toggle_record", methods=["POST"])
def toggle_record():
    return jsonify(engine.toggle_record())


@app.route("/toggle_truth", methods=["POST"])
def toggle_truth():
    return jsonify({"ground_truth": engine.toggle_truth()})


if __name__ == "__main__":
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    engine = DrowsinessEngine(cam)
    engine.start()
    print("Dashboard em http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
