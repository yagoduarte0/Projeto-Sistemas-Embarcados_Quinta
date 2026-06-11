# -*- coding: utf-8 -*-
"""
Sistema de deteccao de sonolencia para motoristas (caminhoes).
Camera montada no espelho retrovisor interno.

Tecnica:
  - MediaPipe FaceMesh (478 landmarks) -> olhos e boca
  - EAR (Eye Aspect Ratio)  -> abertura dos olhos
  - MAR (Mouth Aspect Ratio)-> bocejo
  - PERCLOS                 -> % de fecho ocular numa janela de tempo (metrica padrao de fadiga)
  - Pontuacao de sonolencia 0-100 que sobe enquanto o olho fica fechado/semicerrado
    e desce quando o motorista volta a abrir os olhos.
  - Alarme sonoro que ESCALA em 3 niveis conforme a pontuacao sobe.

Calibracao automatica no inicio (3s) -> mede o EAR de "olho aberto" do motorista.
Isso adapta os limiares ao angulo do retrovisor e a cada pessoa.

Controles:
  q / ESC  -> sair
  c        -> recalibrar
  m        -> liga/desliga som

Autor: projeto Sistemas Embarcados - Eye Tracking / Deteccao de Sonolencia
"""

import csv
import time
import threading
from collections import deque

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Arquivo de modelo do FaceLandmarker (vai junto do projeto).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "face_landmarker.task")

# winsound so existe no Windows. Em outro SO, o alarme vira so visual.
try:
    import winsound
    _HAS_SOUND = True
except ImportError:
    _HAS_SOUND = False


# ----------------------------------------------------------------------------
# 1. INDICES DOS LANDMARKS (MediaPipe FaceMesh)
# ----------------------------------------------------------------------------
# Cada olho usa 6 pontos para o calculo do EAR.
#   EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
# Ordem: [canto_externo, topo1, topo2, canto_interno, baixo2, baixo1]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Boca (para bocejo): cantos + labio superior/inferior internos
MOUTH = [78, 308, 13, 14]  # [canto_esq, canto_dir, labio_sup, labio_inf]


# ----------------------------------------------------------------------------
# 2. PARAMETROS DE COMPORTAMENTO  (ajuste fino aqui)
# ----------------------------------------------------------------------------
# Os limiares de olho sao RELATIVOS ao EAR calibrado (olho aberto = 1.0):
RATIO_CLOSED = 0.55   # abaixo disso = olho FECHADO
RATIO_HALF   = 0.78   # entre CLOSED e HALF = SEMICERRADO; acima = ABERTO

MAR_YAWN = 0.6        # MAR acima disso = bocejo

# Evolucao da pontuacao de sonolencia (0-100), por segundo:
SCORE_RATE_CLOSED = 45.0   # quao rapido sobe com olho fechado
SCORE_RATE_HALF   = 18.0   # sobe mais devagar com olho semicerrado
SCORE_RECOVERY    = 25.0   # quao rapido desce com olho aberto
SCORE_YAWN_BUMP   = 8.0    # acrescimo instantaneo por bocejo

# Limiares dos niveis de alerta (sobre a pontuacao 0-100):
LEVEL_ATTENTION = 30   # nivel 1: atencao
LEVEL_ALERT     = 60   # nivel 2: alerta
LEVEL_DANGER    = 85   # nivel 3: perigo

PERCLOS_WINDOW = 60.0  # janela do PERCLOS em segundos
CALIB_SECONDS  = 3.0   # duracao da calibracao

# --- Monitoramento de atencao / distracao (head pose) ---
YAW_AWAY_DEG        = 22.0  # desvio de yaw (graus) p/ "olhando para o lado"
DISTRACTION_SECONDS = 1.5   # tempo olhando para o lado p/ marcar DISTRAIDO
SCORE_RATE_DISTRACTED = 18.0  # distracao contribui na pontuacao (por segundo)

# --- Microssono e piscadas ---
MICROSLEEP_SECONDS = 2.0   # olhos fechados continuos >= isto = microssono
BLINK_MAX_SECONDS  = 0.4   # fecho mais curto que isto conta como piscada
FATIGUE_WINDOW     = 60.0  # janela p/ taxa de piscadas (segundos)


# ----------------------------------------------------------------------------
# 3. ALARME SONORO ESCALONADO (thread separada p/ nao travar o video)
# ----------------------------------------------------------------------------
class AlarmManager:
    """Toca um beep cuja frequencia/cadencia aumentam com o nivel (1..3)."""

    # (frequencia_hz, duracao_ms, pausa_entre_beeps_s)
    PROFILES = {
        1: (880,  150, 1.20),   # atencao   - beep suave e espacado
        2: (1320, 250, 0.55),   # alerta    - mais agudo e frequente
        3: (2000, 450, 0.12),   # perigo    - quase continuo e estridente
    }

    def __init__(self):
        self._level = 0
        self._enabled = True
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_level(self, level: int):
        with self._lock:
            self._level = level

    def toggle(self):
        self._enabled = not self._enabled
        return self._enabled

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            with self._lock:
                level = self._level if self._enabled else 0
            if level <= 0:
                time.sleep(0.1)
                continue
            freq, dur, pause = self.PROFILES[level]
            if _HAS_SOUND:
                try:
                    winsound.Beep(freq, dur)
                except RuntimeError:
                    pass
            else:
                # Sem winsound: usa o bell do terminal como fallback.
                print("\a", end="", flush=True)
                time.sleep(dur / 1000.0)
            time.sleep(pause)


# ----------------------------------------------------------------------------
# 4. FUNCOES GEOMETRICAS
# ----------------------------------------------------------------------------
def _dist(a, b):
    return np.linalg.norm(a - b)


def eye_aspect_ratio(landmarks, idx):
    p = [landmarks[i] for i in idx]
    vert = _dist(p[1], p[5]) + _dist(p[2], p[4])
    horiz = 2.0 * _dist(p[0], p[3])
    return vert / horiz if horiz > 1e-6 else 0.0


def mouth_aspect_ratio(landmarks, idx):
    p = [landmarks[i] for i in idx]
    horiz = _dist(p[0], p[1])
    vert = _dist(p[2], p[3])
    return vert / horiz if horiz > 1e-6 else 0.0


def landmarks_to_array(face_landmarks, w, h):
    """Converte a lista de landmarks normalizados em coordenadas de pixel (Nx2).

    Na Tasks API, face_landmarks ja e uma lista de NormalizedLandmark.
    """
    return np.array([(lm.x * w, lm.y * h) for lm in face_landmarks],
                    dtype=np.float32)


def create_landmarker():
    """Cria o FaceLandmarker (Tasks API) em modo VIDEO."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modelo nao encontrado: {MODEL_PATH}\n"
            "Baixe face_landmarker.task de "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task"
        )
    # Lemos o modelo em bytes no Python e passamos via model_asset_buffer.
    # Motivo: o loader nativo do MediaPipe NAO abre caminhos com acentos no
    # Windows (ex.: "Cosmeticos", "Area"). O open() do Python lida com unicode.
    with open(MODEL_PATH, "rb") as f:
        model_bytes = f.read()

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,   # habilita eyeBlinkLeft/Right (metodo B)
        output_facial_transformation_matrixes=True,  # pose da cabeca (distracao)
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def blink_score(blend_result):
    """Metodo B (ML): score de olho fechado dado pelos blendshapes do MediaPipe.

    Recebe res.face_blendshapes[0] (lista de Category) e devolve o maior entre
    eyeBlinkLeft e eyeBlinkRight, em [0,1] (0 = aberto, 1 = totalmente fechado).
    """
    d = {c.category_name: c.score for c in blend_result}
    return max(d.get("eyeBlinkLeft", 0.0), d.get("eyeBlinkRight", 0.0))


def head_euler_angles(matrix):
    """Extrai (yaw, pitch, roll) em graus da matriz 4x4 de transformacao facial.

    yaw   = virar a cabeca para os lados (esquerda/direita) -> distracao
    pitch = inclinar para cima/baixo (cabeca caindo) -> sono
    """
    R = np.array(matrix, dtype=np.float64)[:3, :3]
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0
    return float(yaw), float(pitch), float(roll)


class HeadPoseMonitor:
    """Detecta DISTRACAO (olhos fora da via) por desvio de yaw da cabeca.

    A linha de base (frente) e capturada na calibracao, deixando o sistema
    robusto ao angulo do retrovisor.
    """
    def __init__(self):
        self.yaw0 = 0.0
        self.pitch0 = 0.0
        self.calibrated = False
        self._away_start = None

    def set_baseline(self, yaw, pitch):
        self.yaw0, self.pitch0 = yaw, pitch
        self.calibrated = True

    def update(self, now, yaw, pitch):
        yaw_dev = yaw - self.yaw0
        pitch_dev = pitch - self.pitch0
        looking_away = abs(yaw_dev) > YAW_AWAY_DEG
        if looking_away:
            if self._away_start is None:
                self._away_start = now
            away_dur = now - self._away_start
        else:
            self._away_start = None
            away_dur = 0.0
        distracted = away_dur >= DISTRACTION_SECONDS
        return yaw_dev, pitch_dev, distracted


class FatigueMeters:
    """Microssono, taxa de piscadas, duracao do fecho e latencia do alerta.

    Chame update(now, closed_flag) a cada frame; e register_alert(now) quando
    o alarme dispara (level>=1) para medir a latencia do alerta.
    """
    def __init__(self):
        self._closed = False
        self._closed_start = None
        self._ms_flagged = False
        self._alert_recorded = False
        self.microsleep_count = 0
        self.microsleep_active = False
        self.closed_duration = 0.0
        self._blinks = deque()          # timestamps de piscadas completas
        self._blink_durs = deque()      # (t, duracao)
        self.last_latency = None
        self._latencies = []

    def update(self, now, closed_flag):
        if closed_flag and not self._closed:        # olhos acabaram de fechar
            self._closed = True
            self._closed_start = now
            self._ms_flagged = False
            self._alert_recorded = False
        elif closed_flag and self._closed:          # continuam fechados
            self.closed_duration = now - self._closed_start
            if self.closed_duration >= MICROSLEEP_SECONDS and not self._ms_flagged:
                self.microsleep_count += 1
                self._ms_flagged = True
            self.microsleep_active = self.closed_duration >= MICROSLEEP_SECONDS
        elif (not closed_flag) and self._closed:    # olhos abriram -> fim do fecho
            dur = now - self._closed_start
            if dur < BLINK_MAX_SECONDS:
                self._blinks.append(now)
                self._blink_durs.append((now, dur))
            self._closed = False
            self.closed_duration = 0.0
            self.microsleep_active = False

        while self._blinks and now - self._blinks[0] > FATIGUE_WINDOW:
            self._blinks.popleft()
        while self._blink_durs and now - self._blink_durs[0][0] > FATIGUE_WINDOW:
            self._blink_durs.popleft()

    def register_alert(self, now):
        if self._closed and self._closed_start is not None and not self._alert_recorded:
            self.last_latency = now - self._closed_start
            self._latencies.append(self.last_latency)
            self._alert_recorded = True

    @property
    def blink_rate(self):
        """Piscadas por minuto (na janela FATIGUE_WINDOW)."""
        return len(self._blinks) * (60.0 / FATIGUE_WINDOW)

    @property
    def mean_blink_ms(self):
        if not self._blink_durs:
            return 0.0
        return 1000.0 * sum(d for _, d in self._blink_durs) / len(self._blink_durs)

    @property
    def avg_latency(self):
        if not self._latencies:
            return None
        return sum(self._latencies) / len(self._latencies)


# ----------------------------------------------------------------------------
# 4b. REGISTRO EM CSV (para a avaliacao quantitativa do artigo)
# ----------------------------------------------------------------------------
class CsvLogger:
    """Grava uma linha por frame em logs/session_AAAAMMDD_HHMMSS.csv.

    Colunas:
      t            tempo (s) desde o inicio da sessao
      ear          EAR medio (Metodo A, geometrico)
      ear_ratio    EAR / EAR_calibrado  (1.0 = totalmente aberto)
      blink_blend  blendshape eyeBlink  (Metodo B, ML; 0=aberto, 1=fechado)
      state        estado textual do olho
      score        pontuacao de sonolencia 0-100
      level        nivel de alerta 0-3
      perclos      PERCLOS [0,1]
      yawns        contagem acumulada de bocejos
      ground_truth rotulo verdadeiro: 1 = olhos fechados/sonolento (tecla 'g')
    """
    HEADER = ["t", "ear", "ear_ratio", "blink_blend", "state",
              "score", "level", "perclos", "yawns",
              "yaw_dev", "pitch_dev", "distracted", "microsleep", "blink_rate",
              "ground_truth"]

    def __init__(self, folder=None):
        if folder is None:
            folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(folder, exist_ok=True)
        name = "session_%s.csv" % time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(folder, name)
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(self.HEADER)

    def log(self, t, ear, ear_ratio, blink_blend, state, score, level, perclos,
            yawns, yaw_dev, pitch_dev, distracted, microsleep, blink_rate,
            ground_truth):
        self._w.writerow([round(t, 3), round(ear, 4), round(ear_ratio, 4),
                          round(blink_blend, 4), state, round(score, 2), level,
                          round(perclos, 4), yawns, round(yaw_dev, 1),
                          round(pitch_dev, 1), int(distracted), microsleep,
                          round(blink_rate, 1), int(ground_truth)])

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 5. DESENHO DO PAINEL (HUD)
# ----------------------------------------------------------------------------
LEVEL_COLORS = {
    0: (0, 200, 0),      # verde
    1: (0, 215, 255),    # amarelo
    2: (0, 140, 255),    # laranja
    3: (0, 0, 255),      # vermelho
}
LEVEL_LABELS = {
    0: "NORMAL",
    1: "ATENCAO",
    2: "ALERTA",
    3: "PERIGO - PARE O VEICULO",
}


def draw_hud(frame, score, level, ear, state, perclos, yawning, calibrating, fps):
    h, w = frame.shape[:2]
    color = LEVEL_COLORS[level]

    # Faixa superior translucida
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    if calibrating:
        cv2.putText(frame, "CALIBRANDO... mantenha os olhos abertos e olhe pra frente",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return

    cv2.putText(frame, f"Estado: {state}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}   PERCLOS: {perclos*100:4.1f}%   FPS: {fps:4.1f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    status = LEVEL_LABELS[level]
    if yawning:
        status += "  (BOCEJO)"
    cv2.putText(frame, status, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Barra de sonolencia
    bx, by, bw, bh = 20, h - 50, w - 40, 26
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), 2)
    fill = int(bw * min(score, 100) / 100.0)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), color, -1)
    cv2.putText(frame, f"Sonolencia: {score:5.1f}/100", (bx + 8, by + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Moldura vermelha pulsante em perigo
    if level >= 3:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)


# ----------------------------------------------------------------------------
# 6. LOOP PRINCIPAL
# ----------------------------------------------------------------------------
def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Nao consegui abrir a camera index={camera_index}")

    landmarker = create_landmarker()

    alarm = AlarmManager()

    # Estado da calibracao
    calibrating = True
    calib_start = time.time()
    calib_samples = []
    ear_open = None  # EAR de referencia (olho aberto)

    score = 0.0
    perclos_buf = deque()  # (timestamp, fechado?)
    start_t = time.time()
    prev_t = start_t
    last_ts = -1           # timestamp (ms) estritamente crescente p/ modo VIDEO
    yawn_active = False
    yawn_count = 0

    # Atencao/distracao e fadiga
    headpose = HeadPoseMonitor()
    fatigue = FatigueMeters()
    headpose_samples = []

    # Registro p/ avaliacao do artigo
    logger = None
    recording = False
    ground_truth = False   # tecla 'g': rotulo "olhos fechados/sonolento"

    print("Iniciando. Calibrando por %.0f s - mantenha os olhos abertos." % CALIB_SECONDS)
    print("Teclas: q=sair  c=recalibrar  m=som  r=gravar CSV  g=rotulo verdadeiro")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # espelha (mais natural na tela)
        h, w = frame.shape[:2]

        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 1.0 / dt if dt > 0 else 0.0

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = max(last_ts + 1, int((now - start_t) * 1000))
        last_ts = ts_ms
        res = landmarker.detect_for_video(mp_image, ts_ms)

        ear = 0.0
        ear_ratio = 1.0
        blink = 0.0
        yaw_dev = 0.0
        pitch_dev = 0.0
        distracted = False
        state = "SEM ROSTO"
        level = 0
        perclos = 0.0
        yawning = False

        if res.face_landmarks:
            pts = landmarks_to_array(res.face_landmarks[0], w, h)
            ear_l = eye_aspect_ratio(pts, LEFT_EYE)
            ear_r = eye_aspect_ratio(pts, RIGHT_EYE)
            ear = (ear_l + ear_r) / 2.0
            mar = mouth_aspect_ratio(pts, MOUTH)
            if res.face_blendshapes:
                blink = blink_score(res.face_blendshapes[0])  # Metodo B (ML)

            # Pose da cabeca (yaw/pitch) p/ deteccao de distracao
            yaw = pitch = 0.0
            if res.facial_transformation_matrixes:
                yaw, pitch, _ = head_euler_angles(
                    res.facial_transformation_matrixes[0])

            # ---------- CALIBRACAO ----------
            if calibrating:
                calib_samples.append(ear)
                headpose_samples.append((yaw, pitch))
                if now - calib_start >= CALIB_SECONDS:
                    arr = np.array(calib_samples)
                    # mediana dos quadros bons (descarta piscadas)
                    ear_open = float(np.median(arr[arr > np.percentile(arr, 40)]))
                    hp = np.array(headpose_samples)
                    headpose.set_baseline(float(np.median(hp[:, 0])),
                                          float(np.median(hp[:, 1])))
                    calibrating = False
                    print(f"Calibracao concluida. EAR aberto = {ear_open:.3f}")
                draw_hud(frame, 0, 0, ear, "", 0, False, True, fps)
                cv2.imshow("Deteccao de Sonolencia - Motorista", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
                continue

            # ---------- CLASSIFICACAO DO OLHO ----------
            r = ear / ear_open if ear_open else 1.0
            ear_ratio = r
            if r < RATIO_CLOSED:
                state, closed_flag = "OLHOS FECHADOS", True
                score += SCORE_RATE_CLOSED * dt
            elif r < RATIO_HALF:
                state, closed_flag = "SEMICERRADO", True
                score += SCORE_RATE_HALF * dt
            else:
                state, closed_flag = "OLHOS ABERTOS", False
                score -= SCORE_RECOVERY * dt

            # ---------- BOCEJO ----------
            if mar > MAR_YAWN:
                yawning = True
                if not yawn_active:        # conta so na transicao
                    score += SCORE_YAWN_BUMP
                    yawn_count += 1
                    yawn_active = True
            else:
                yawn_active = False

            # ---------- PERCLOS (janela deslizante) ----------
            perclos_buf.append((now, closed_flag))
            while perclos_buf and now - perclos_buf[0][0] > PERCLOS_WINDOW:
                perclos_buf.popleft()
            if perclos_buf:
                perclos = sum(1 for _, c in perclos_buf if c) / len(perclos_buf)

            # ---------- DISTRACAO (olhos fora da via) ----------
            yaw_dev, pitch_dev, distracted = headpose.update(now, yaw, pitch)
            if distracted:
                score += SCORE_RATE_DISTRACTED * dt
                state = "DISTRAIDO"

            # ---------- MICROSSONO / PISCADAS ----------
            fatigue.update(now, closed_flag)

        else:
            # Sem rosto: pode ser cabeca baixa (dormindo) -> sobe devagar
            score += SCORE_RATE_HALF * dt * 0.5

        # ---------- LIMITA E DEFINE NIVEL ----------
        score = float(np.clip(score, 0.0, 100.0))
        if score >= LEVEL_DANGER:
            level = 3
        elif score >= LEVEL_ALERT:
            level = 2
        elif score >= LEVEL_ATTENTION:
            level = 1
        else:
            level = 0
        alarm.set_level(level)
        if level >= 1:
            fatigue.register_alert(now)

        # ---------- REGISTRO EM CSV ----------
        if recording and logger is not None:
            logger.log(now - start_t, ear, ear_ratio, blink, state, score, level,
                       perclos, yawn_count, yaw_dev, pitch_dev, distracted,
                       fatigue.microsleep_count, fatigue.blink_rate, ground_truth)

        draw_hud(frame, score, level, ear, state, perclos, yawning, False, fps)
        # Indicadores de gravacao e rotulo verdadeiro
        if recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 90, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ground_truth:
            cv2.putText(frame, "ROTULO: SONOLENTO", (w - 320, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # Linha com metricas de fadiga
        info = "Piscadas/min: %.0f   Microssonos: %d   Yaw: %+.0f deg" % (
            fatigue.blink_rate, fatigue.microsleep_count, yaw_dev)
        cv2.putText(frame, info, (20, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        if distracted:
            cv2.putText(frame, "DISTRAIDO - OLHE A VIA", (w // 2 - 220, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        if fatigue.microsleep_active:
            cv2.putText(frame, "MICROSSONO!", (w // 2 - 150, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.imshow("Deteccao de Sonolencia - Motorista", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            calibrating = True
            calib_start = time.time()
            calib_samples = []
            score = 0.0
            print("Recalibrando...")
        elif key == ord('m'):
            on = alarm.toggle()
            print("Som", "ligado" if on else "desligado")
        elif key == ord('r'):
            recording = not recording
            if recording and logger is None:
                logger = CsvLogger()
                print("Gravando em:", logger.path)
            print("Gravacao", "LIGADA" if recording else "pausada")
        elif key == ord('g'):
            ground_truth = not ground_truth
            print("Rotulo verdadeiro:", "SONOLENTO" if ground_truth else "alerta")

    alarm.stop()
    if logger is not None:
        logger.close()
        print("CSV salvo em:", logger.path)
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    import sys
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(cam)
