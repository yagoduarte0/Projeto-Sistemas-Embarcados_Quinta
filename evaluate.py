# -*- coding: utf-8 -*-
"""
Avaliacao quantitativa COMPARANDO DOIS METODOS de deteccao de olho fechado,
usando os CSV gravados pelo sistema (dashboard.py / drowsiness_detector.py).

  Metodo A (geometrico): EAR  -> olho fechado se ear_ratio < limiar
  Metodo B (ML):         blendshape eyeBlink do MediaPipe -> fechado se blink > limiar

O rotulo verdadeiro (ground_truth) e a coluna marcada por voce durante a coleta
(1 = olhos fechados/sonolento).

Saidas (pasta figures/):
  - confusion_matrices.png/.pdf  : matrizes de confusao dos 2 metodos
  - metrics_comparison.png/.pdf  : barras de Acuracia/Precisao/Recall/F1
  - threshold_sweep.png/.pdf     : F1 vs limiar (escolha do operacional)
  - metrics_table.tex            : tabela LaTeX pronta para o artigo

Uso:
    python evaluate.py                  # usa o CSV mais recente em logs/
    python evaluate.py logs/sessao.csv  # usa um CSV especifico
    python evaluate.py logs/a.csv logs/b.csv ...  # junta varios CSV
"""

import os
import sys
import csv
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sem janela; so salva arquivos
import matplotlib.pyplot as plt

import drowsiness_detector as dd

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(HERE, "logs")
FIGS = os.path.join(HERE, "figures")


# ----------------------------------------------------------------------------
# Leitura dos dados
# ----------------------------------------------------------------------------
def load_csvs(paths):
    ear_ratio, blink, truth = [], [], []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ear = float(row["ear"])
                    if ear <= 0:           # sem rosto -> descarta
                        continue
                    ear_ratio.append(float(row["ear_ratio"]))
                    blink.append(float(row["blink_blend"]))
                    truth.append(int(row["ground_truth"]))
                except (KeyError, ValueError):
                    continue
    return (np.array(ear_ratio), np.array(blink), np.array(truth))


def newest_csv():
    files = sorted(glob.glob(os.path.join(LOGS, "*.csv")))
    if not files:
        return None
    return files[-1]


# ----------------------------------------------------------------------------
# Metricas
# ----------------------------------------------------------------------------
def confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return tp, fp, fn, tn


def metrics(y_true, y_pred):
    tp, fp, fn, tn = confusion(y_true, y_pred)
    n = max(tp + fp + fn + tn, 1)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "cm": (tp, fp, fn, tn)}


def best_threshold(score, y_true, thresholds, closed_when_below):
    """Varre limiares e devolve (melhor_limiar, melhor_metrica)."""
    best_t, best = thresholds[0], None
    for t in thresholds:
        y_pred = (score < t).astype(int) if closed_when_below else (score > t).astype(int)
        m = metrics(y_true, y_pred)
        if best is None or m["f1"] > best["f1"]:
            best, best_t = m, t
    return best_t, best


# ----------------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------------
def plot_confusions(mA, mB, outbase):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, m, title in zip(axes, [mA, mB],
                            ["Metodo A: EAR (geometrico)",
                             "Metodo B: Blendshape (ML)"]):
        tp, fp, fn, tn = m["cm"]
        cm = np.array([[tn, fp], [fn, tp]])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Alerta", "Fechado"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Alerta", "Fechado"])
        ax.set_xlabel("Predito"); ax.set_ylabel("Verdadeiro")
        thr = cm.max() / 2 if cm.max() else 0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thr else "black",
                        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=150)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)


def plot_bars(mA, mB, outbase):
    labels = ["Acuracia", "Precisao", "Recall", "F1"]
    a = [mA["acc"], mA["prec"], mA["rec"], mA["f1"]]
    b = [mB["acc"], mB["prec"], mB["rec"], mB["f1"]]
    x = np.arange(len(labels)); wbar = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.6))
    r1 = ax.bar(x - wbar / 2, a, wbar, label="A: EAR (geometrico)", color="#3b82f6")
    r2 = ax.bar(x + wbar / 2, b, wbar, label="B: Blendshape (ML)", color="#f97316")
    ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Valor"); ax.legend()
    ax.set_title("Comparacao dos metodos de deteccao de olho fechado")
    for r in list(r1) + list(r2):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01,
                f"{r.get_height():.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=150)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)


def plot_sweep(ear_ratio, blink, y_true, outbase):
    tA = np.linspace(0.3, 1.0, 71)
    f1A = []
    for t in tA:
        f1A.append(metrics(y_true, (ear_ratio < t).astype(int))["f1"])
    tB = np.linspace(0.0, 1.0, 101)
    f1B = []
    for t in tB:
        f1B.append(metrics(y_true, (blink > t).astype(int))["f1"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(tA, f1A, label="A: EAR (limiar de ear_ratio)", color="#3b82f6")
    ax.plot(tB, f1B, label="B: Blendshape (limiar de eyeBlink)", color="#f97316")
    ax.set_xlabel("Limiar"); ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.set_title("F1 em funcao do limiar")
    fig.tight_layout()
    fig.savefig(outbase + ".png", dpi=150)
    fig.savefig(outbase + ".pdf")
    plt.close(fig)


def write_latex_table(mA, mB, tA, tB, n, path):
    def row(name, m, t):
        tp, fp, fn, tn = m["cm"]
        return (f"{name} & {t:.2f} & {m['acc']:.3f} & {m['prec']:.3f} & "
                f"{m['rec']:.3f} & {m['f1']:.3f} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Comparacao quantitativa dos metodos de deteccao de olho fechado
(%d quadros rotulados).}
\label{tab:results}
\begin{tabular}{lccccc}
\hline
Metodo & Limiar & Acuracia & Precisao & Recall & F1 \\
\hline
%s
%s
\hline
\end{tabular}
\end{table}
""" % (n, row("A: EAR (geometrico)", mA, tA), row("B: Blendshape (ML)", mB, tB))
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    paths = sys.argv[1:]
    if not paths:
        nc = newest_csv()
        if nc is None:
            print("Nenhum CSV em logs/. Rode o dashboard, ligue 'Gravar' e colete dados.")
            return
        paths = [nc]
    print("Lendo:", ", ".join(os.path.basename(p) for p in paths))

    ear_ratio, blink, y_true = load_csvs(paths)
    n = len(y_true)
    if n == 0:
        print("CSV sem quadros validos (com rosto).")
        return
    pos, neg = int(y_true.sum()), int((y_true == 0).sum())
    print(f"Quadros: {n}  |  rotulados sonolento(1): {pos}  |  alerta(0): {neg}")
    if pos == 0 or neg == 0:
        print("AVISO: e preciso ter quadros dos DOIS rotulos (segure 'g' com olhos "
              "fechados em parte da coleta) para metricas significativas.")

    # Limiares operacionais: melhor F1 por varredura (reportado no artigo)
    tA, mA = best_threshold(ear_ratio, y_true, np.linspace(0.3, 1.0, 71),
                            closed_when_below=True)
    tB, mB = best_threshold(blink, y_true, np.linspace(0.0, 1.0, 101),
                            closed_when_below=False)

    print("\n=== METODO A: EAR (geometrico) ===")
    print(f"  limiar ear_ratio < {tA:.2f}")
    print(f"  Acuracia {mA['acc']:.3f}  Precisao {mA['prec']:.3f}  "
          f"Recall {mA['rec']:.3f}  F1 {mA['f1']:.3f}")
    print("=== METODO B: Blendshape eyeBlink (ML) ===")
    print(f"  limiar blink > {tB:.2f}")
    print(f"  Acuracia {mB['acc']:.3f}  Precisao {mB['prec']:.3f}  "
          f"Recall {mB['rec']:.3f}  F1 {mB['f1']:.3f}")

    os.makedirs(FIGS, exist_ok=True)
    plot_confusions(mA, mB, os.path.join(FIGS, "confusion_matrices"))
    plot_bars(mA, mB, os.path.join(FIGS, "metrics_comparison"))
    plot_sweep(ear_ratio, blink, y_true, os.path.join(FIGS, "threshold_sweep"))
    write_latex_table(mA, mB, tA, tB, n, os.path.join(FIGS, "metrics_table.tex"))
    print(f"\nFiguras e tabela salvas em: {FIGS}")


if __name__ == "__main__":
    main()
