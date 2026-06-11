# -*- coding: utf-8 -*-
"""
TESTE GUIADO — coleta + avaliacao + FPS/latencia em um unico comando.

Faz, automaticamente, os 3 fatores que entram no artigo:

  1) COLETA com ground truth:
     conduz um protocolo cronometrado, alternando fases
       - "OLHOS ABERTOS"  (rotulo ALERTA  = 0)
       - "FECHE OS OLHOS" (rotulo SONOLENTO = 1)
     marcando o rotulo verdadeiro sozinho e gravando o CSV em logs/.

  2) AVALIACAO:
     ao terminar, chama o evaluate.py no CSV recem-gravado e gera
     a tabela (figures/metrics_table.tex) e as figuras (figures/*.pdf).

  3) FPS e LATENCIA do alarme:
     mede o FPS medio durante a sessao e a latencia media do alerta
     (tempo do fecho dos olhos ate o primeiro apito) — sem cronometro.

Uso:
    python coletar_e_avaliar.py            # camera 0
    python coletar_e_avaliar.py 1          # outra camera

Teclas durante a coleta:  q = abortar (mesmo assim avalia o que coletou)
                          m = liga/desliga o som

Ajuste o protocolo nas constantes CYCLES e PHASE_SECONDS abaixo.
"""

import sys
import time

import cv2
import numpy as np
import mediapipe as mp

import drowsiness_detector as dd
import evaluate

# ---- Protocolo (ajuste aqui) ----
CYCLES = 3            # quantas vezes repetir o par (ABERTO, FECHADO)
PHASE_SECONDS = 20.0  # duracao de cada fase (s). 3 ciclos x 2 x 20s = 120s


def _draw_center(frame, text, y, scale, color, thick=2):
    w = frame.shape[1]
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.putText(frame, text, ((w - tw) // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Nao consegui abrir a camera index={camera_index}")

    landmarker = dd.create_landmarker()
    alarm = dd.AlarmManager()
    fatigue = dd.FatigueMeters()
    logger = dd.CsvLogger()

    # plano de fases: (instrucao, ground_truth)
    phases = []
    for _ in range(CYCLES):
        phases.append(("MANTENHA OS OLHOS ABERTOS", 0))
        phases.append(("FECHE / PISQUE LENTAMENTE", 1))

    calibrating = True
    calib_start = time.time()
    calib_samples = []
    ear_open = None

    start_t = time.time()
    prev_t = start_t
    last_ts = -1
    score = 0.0
    yawn_active = False
    yawn_count = 0
    fps_samples = []

    phase_idx = -1          # -1 = ainda calibrando
    phase_start = None

    print("Protocolo: %d ciclos x 2 fases x %.0fs (~%.0fs de coleta)."
          % (CYCLES, PHASE_SECONDS, CYCLES * 2 * PHASE_SECONDS))
    print("Calibrando %.0fs: mantenha os olhos abertos olhando para frente."
          % dd.CALIB_SECONDS)

    aborted = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
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
        state = "SEM ROSTO"
        closed_flag = False

        if res.face_landmarks:
            pts = dd.landmarks_to_array(res.face_landmarks[0], w, h)
            ear = (dd.eye_aspect_ratio(pts, dd.LEFT_EYE) +
                   dd.eye_aspect_ratio(pts, dd.RIGHT_EYE)) / 2.0
            mar = dd.mouth_aspect_ratio(pts, dd.MOUTH)
            if res.face_blendshapes:
                blink = dd.blink_score(res.face_blendshapes[0])

            # ----- calibracao -----
            if calibrating:
                calib_samples.append(ear)
                if now - calib_start >= dd.CALIB_SECONDS:
                    arr = np.array(calib_samples)
                    ear_open = float(np.median(arr[arr > np.percentile(arr, 40)]))
                    calibrating = False
                    phase_idx = 0
                    phase_start = now
                    print("Calibracao OK (EAR aberto = %.3f). Iniciando coleta."
                          % ear_open)
            else:
                r = ear / ear_open if ear_open else 1.0
                ear_ratio = r
                if r < dd.RATIO_CLOSED:
                    state, closed_flag = "OLHOS FECHADOS", True
                    score += dd.SCORE_RATE_CLOSED * dt
                elif r < dd.RATIO_HALF:
                    state, closed_flag = "SEMICERRADO", True
                    score += dd.SCORE_RATE_HALF * dt
                else:
                    state, closed_flag = "OLHOS ABERTOS", False
                    score -= dd.SCORE_RECOVERY * dt
                if mar > dd.MAR_YAWN and not yawn_active:
                    score += dd.SCORE_YAWN_BUMP
                    yawn_count += 1
                    yawn_active = True
                elif mar <= dd.MAR_YAWN:
                    yawn_active = False
                fatigue.update(now, closed_flag)

        # ----- nivel / alarme / latencia -----
        score = float(np.clip(score, 0.0, 100.0))
        if calibrating:
            level = 0
        elif score >= dd.LEVEL_DANGER:
            level = 3
        elif score >= dd.LEVEL_ALERT:
            level = 2
        elif score >= dd.LEVEL_ATTENTION:
            level = 1
        else:
            level = 0
        alarm.set_level(level)
        if level >= 1:
            fatigue.register_alert(now)

        # ----- fase atual / ground truth -----
        if calibrating:
            cv2.rectangle(frame, (0, 0), (w, 140), (0, 0, 0), -1)
            _draw_center(frame, "CALIBRANDO...", 70, 1.2, (0, 255, 255), 3)
            _draw_center(frame, "olhos abertos, olhando para frente",
                         110, 0.7, (200, 200, 200))
        else:
            # avanca de fase quando o tempo acaba
            if now - phase_start >= PHASE_SECONDS:
                phase_idx += 1
                phase_start = now
                if phase_idx >= len(phases):
                    break  # fim do protocolo

            instr, gt = phases[phase_idx]
            remaining = PHASE_SECONDS - (now - phase_start)
            fps_samples.append(fps)

            # grava sempre durante a coleta (ground truth = gt da fase)
            logger.log(now - start_t, ear, ear_ratio, blink, state, score,
                       level, 0.0, yawn_count, 0.0, 0.0, False,
                       fatigue.microsleep_count, fatigue.blink_rate, gt)

            # painel superior com a instrucao e contagem
            band = (0, 110, 0) if gt == 0 else (0, 0, 140)
            cv2.rectangle(frame, (0, 0), (w, 150), band, -1)
            tag = "ALERTA" if gt == 0 else "SONOLENTO"
            _draw_center(frame, "%s  (rotulo: %s)" % (instr, tag),
                         55, 1.0, (255, 255, 255), 3)
            _draw_center(frame, "Fase %d/%d   tempo: %4.1fs" % (
                phase_idx + 1, len(phases), max(0.0, remaining)),
                100, 0.8, (230, 230, 230))
            # barra de progresso da fase
            pw = int(w * (now - phase_start) / PHASE_SECONDS)
            cv2.rectangle(frame, (0, 140), (pw, 150), (255, 255, 255), -1)
            cv2.putText(frame, "EAR %.2f  blink %.2f  score %.0f  FPS %.0f" % (
                ear, blink, score, fps), (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Coleta guiada - Sonolencia", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            aborted = True
            break
        elif key == ord('m'):
            alarm.toggle()

    # ---- finalizacao ----
    alarm.stop()
    logger.close()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    avg_fps = float(np.mean(fps_samples)) if fps_samples else 0.0
    lat = fatigue.avg_latency
    print("\n" + "=" * 60)
    print("COLETA %s. CSV: %s" % ("ABORTADA" if aborted else "concluida",
                                  logger.path))
    print("FPS medio: %.1f" % avg_fps)
    if lat is not None:
        print("Latencia media do alerta: %.2f s" % lat)
    else:
        print("Latencia do alerta: (nenhum alerta disparado nesta sessao)")
    print("Bocejos: %d   Microssonos: %d" % (yawn_count, fatigue.microsleep_count))
    print("=" * 60 + "\n")

    # ---- AVALIACAO automatica ----
    print(">>> Rodando evaluate.py no CSV coletado...\n")
    sys.argv = ["evaluate.py", logger.path]
    evaluate.main()

    print("\nPara o artigo:")
    print("  - Tabela 1:  figures/metrics_table.tex (Acc/Prec/Rec/F1 dos 2 metodos)")
    print("  - Fig. 2:    figures/confusion_matrices.pdf")
    print("  - Fig. 3:    figures/metrics_comparison.pdf")
    print("  - Results:   FPS medio = %.1f   |   Latencia media = %s" % (
        avg_fps, ("%.2f s" % lat) if lat is not None else "N/D"))


if __name__ == "__main__":
    cam = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(cam)
