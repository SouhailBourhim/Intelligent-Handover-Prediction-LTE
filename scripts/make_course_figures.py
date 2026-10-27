"""
Figures du cours en français (docs/cours/img/).

Usage
-----
    python scripts/make_course_figures.py

Les figures 1–5 ne dépendent que du simulateur (numpy / pandas / matplotlib).
Les figures 6–7 lisent les modèles entraînés (models/) et le split de test
(data/processed/test.csv) ; elles sont ignorées si le pipeline n'a pas tourné.
Les figures 8+ sont extraites des sorties des notebooks exécutés.
"""

from __future__ import annotations

import base64
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.radio_model import (  # noqa: E402
    RADIO, _CQI_TABLE, path_loss_los, path_loss_nlos, update_fast_fading, update_shadow,
)
from src.handover_logic import HO  # noqa: E402

OUT = ROOT / "docs" / "cours" / "img"
INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREEN, RED = "#3b82f6", "#f59e0b", "#10b981", "#ef4444"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {name}")


# ── 1. Les trois échelles de la propagation ───────────────────────────────────

def fig_echelles():
    rng = np.random.default_rng(4)
    d = np.arange(20, 600, 1.0)                     # UE qui s'éloigne à 1 m par pas
    pl = np.array([path_loss_nlos(x) for x in d])
    mean = RADIO.rs_power_dbm - pl
    sh, shadow = 0.0, []
    for _ in d:
        sh = update_shadow(sh, 1.0, False, rng)
        shadow.append(sh)
    shadow = np.array(shadow)
    # évanouissement rapide instantané (un échantillon) vs moyenné sur 1 s
    inst = 10 * np.log10(rng.exponential(1.0, len(d)))
    avg = np.array([update_fast_fading(0, 0, 1.0, False, rng)[2] for _ in d])

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(d, mean + shadow + np.clip(inst, -25, 8), color="#cbd2d9", lw=0.6,
            label="mesure instantanée (évanouissement rapide brut)")
    ax.plot(d, mean + shadow + avg, color=BLUE, lw=1.4,
            label="mesure moyennée sur 1 s (ce que voit le simulateur)")
    ax.plot(d, mean + shadow, color=ORANGE, lw=2, label="affaiblissement + masquage (shadowing)")
    ax.plot(d, mean, color=INK, lw=2.2, ls="--", label="affaiblissement de parcours seul (path loss)")
    ax.set_xlabel("distance à l'eNodeB (m)")
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("Les trois échelles du canal radio : grande échelle, masquage, évanouissement rapide")
    ax.legend(fontsize=8.5, loc="upper right")
    save(fig, "echelles_propagation.png")


# ── 2. Grille de ressources LTE ────────────────────────────────────────────────

def fig_grille():
    n_sc, n_sym = 24, 14                           # 2 RB × 1 sous-trame (2 slots)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.grid(False)
    for k in range(n_sc):
        for l in range(n_sym):
            # CRS port 0 (préfixe cyclique normal) : symboles 0 et 4 de chaque slot,
            # une sous-porteuse sur 6, décalée de 3 entre les deux symboles
            crs = (l % 7 == 0 and k % 6 == 0) or (l % 7 == 4 and k % 6 == 3)
            pdcch = l < 2 and not crs
            c = RED if crs else ("#fde68a" if pdcch else "#eef2f7")
            ax.add_patch(Rectangle((l, k), 1, 1, facecolor=c, edgecolor="white", lw=1))
    ax.add_patch(Rectangle((0, 0), 7, 12, fill=False, edgecolor=INK, lw=2.2))
    ax.text(3.5, 12.4, "1 Resource Block (RB)\n12 sous-porteuses × 7 symboles = 84 RE",
            ha="center", fontsize=9, color=INK)
    ax.annotate("1 Resource Element (RE)\n= 1 sous-porteuse × 1 symbole",
                xy=(9.5, 1.5), xytext=(15.3, 3.0), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.add_patch(Rectangle((9, 1), 1, 1, fill=False, edgecolor=INK, lw=2))
    ax.set_xlim(0, n_sym)
    ax.set_ylim(0, n_sc + 1.5)
    ax.set_xticks(np.arange(n_sym) + 0.5)
    ax.set_xticklabels([str(i % 7) for i in range(n_sym)])
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_xlabel("temps : symboles OFDM (slot 0 | slot 1 = 1 ms, une sous-trame)")
    ax.set_ylabel("fréquence : sous-porteuses (15 kHz)")
    ax.axvline(7, color=INK, lw=1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="signal de référence CRS → mesuré pour le RSRP"),
                       Patch(color="#fde68a", label="contrôle (PDCCH)"),
                       Patch(color="#eef2f7", label="données (PDSCH)")],
              loc="upper right", fontsize=8.5, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Grille de ressources LTE (10 MHz = 50 RB = 600 sous-porteuses)")
    save(fig, "grille_ressources.png")


# ── 3. Événement A3 : décalage, hystérésis, TTT ───────────────────────────────

def fig_a3():
    t = np.arange(0, 30, 0.1)
    srv = -95 - 0.5 * t + 1.2 * np.sin(t * 0.9)
    nb = -112 + 0.4 * t + 1.0 * np.sin(t * 1.3 + 1)
    margin = HO.ho_margin_db
    met = nb > srv + margin
    first = t[np.argmax(met)]
    ttt = 3.0
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(t, srv, color=BLUE, lw=2.2, label="cellule de service (L3)")
    ax.plot(t, nb, color=ORANGE, lw=2.2, label="meilleure voisine (L3)")
    ax.plot(t, srv + margin, color=BLUE, lw=1, ls="--", label=f"service + {margin:.0f} dB (offset/hystérésis)")
    ax.axvspan(first, first + ttt, color=RED, alpha=0.12)
    ax.axvline(first, color=RED, lw=1)
    ax.axvline(first + ttt, color=INK, lw=1.4)
    ax.text(first + 0.2, ax.get_ylim()[1] - 2, "entrée A3", color=RED, fontsize=9)
    ax.text(first + ttt + 0.2, ax.get_ylim()[1] - 2, "TTT écoulé → rapport\nde mesure → handover",
            fontsize=9, va="top")
    ax.text(first + ttt / 2, ax.get_ylim()[0] + 2, "time-to-trigger", ha="center",
            color=RED, fontsize=9)
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("Événement A3 : « la voisine devient meilleure que la cellule de service d'un décalage »")
    ax.legend(loc="lower left", fontsize=8.5)
    ax2.step(t, met.astype(int), where="post", color=RED, lw=1.6)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["non", "oui"])
    ax2.set_ylabel("condition A3")
    ax2.set_xlabel("temps (s)")
    save(fig, "evenement_a3.png")


# ── 4. Filtre L3 sur des données réelles du projet ─────────────────────────────

def fig_filtre_l3():
    path = ROOT / "data" / "raw" / "dataset.csv"
    if not path.exists():
        print("  (filtre_l3.png ignorée : data/raw/dataset.csv absent)")
        return
    df = pd.read_csv(path)
    # a pedestrian window centred on one of its successful handovers
    ho_t = df[(df.ue_id == 3) & (df.handover_event == 1) & (df.handover_failure == 0)
              & (df.timestamp > 600)].timestamp.iloc[0]
    g = df[(df.ue_id == 3) & df.timestamp.between(ho_t - 60, ho_t + 40)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(g.timestamp, g.rsrp_serving, color="#9fb3c8", lw=1, marker=".", ms=3,
            label="RSRP mesuré (couche 1, chaque seconde)")
    ax.plot(g.timestamp, g.l3_rsrp_serving, color=BLUE, lw=2.2,
            label=f"RSRP filtré couche 3 : F(n) = (1−a)·F(n−1) + a·M(n), a = {HO.l3_alpha}")
    ho = g[g.handover_event == 1]
    for ts in ho.timestamp:
        ax.axvline(ts, color=RED, lw=1, ls=":")
    ax.set_xlabel("temps (s)")
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("Filtrage L3 : la décision de handover se fait sur la courbe lissée (pointillés = handovers)")
    ax.legend(fontsize=8.5, loc="lower left")
    save(fig, "filtre_l3.png")


# ── 5. SINR → CQI → efficacité spectrale ──────────────────────────────────────

# Efficacité (bits/symbole) de la table CQI 4 bits de TS 36.213 (Table 7.2.3-1)
CQI_EFF = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
           2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547]
CQI_MOD = ["QPSK"] * 6 + ["16QAM"] * 3 + ["64QAM"] * 6


def fig_cqi():
    th = [t for t, _ in _CQI_TABLE]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (s, e) in enumerate(zip(th, CQI_EFF)):
        c = {"QPSK": BLUE, "16QAM": ORANGE, "64QAM": GREEN}[CQI_MOD[i]]
        nxt = th[i + 1] if i + 1 < len(th) else 26
        ax.plot([s, nxt], [e, e], color=c, lw=3)
        ax.text(s + 0.1, e + 0.12, str(i + 1), fontsize=8, color=c)
    ax2 = ax.twinx()
    ax2.set_ylim(0, 5.9 * 600 * 14 / 1e3 * 0.75)        # ordre de grandeur Mbit/s (10 MHz)
    ax2.set_ylabel("débit brut ≈ (Mbit/s, 10 MHz, 1 couche)")
    ax2.grid(False)
    ax.set_ylim(0, 5.9)
    ax.set_xlim(-8, 26)
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("efficacité (bits / symbole)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, lw=3, label=m)
                       for m, c in [("QPSK", BLUE), ("16QAM", ORANGE), ("64QAM", GREEN)]],
              loc="upper left", fontsize=8.5)
    ax.set_title("Du SINR au CQI (numéros) puis à la modulation et au débit — TS 36.213")
    save(fig, "sinr_cqi_debit.png")


# ── 6. Compromis précision / rappel selon le seuil ────────────────────────────

def _champion_test_probs():
    try:
        from src.serving import champion_metadata, load_model, predict_scaled, feature_columns
        meta = champion_metadata()
        model = load_model(meta["display_name"])
        test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
        p = predict_scaled(model, test, feature_columns()).values
        return meta, test["handover_soon"].values, p
    except Exception as e:                                  # pipeline pas encore exécuté
        print(f"  (figures modèle ignorées : {e})")
        return None


def fig_seuil(res):
    from sklearn.metrics import precision_score, recall_score, f1_score
    meta, y, p = res
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    P = [precision_score(y, p >= t, zero_division=0) for t in ts]
    R = [recall_score(y, p >= t) for t in ts]
    F = [f1_score(y, p >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, P, color=BLUE, lw=2, label="précision")
    ax.plot(ts, R, color=ORANGE, lw=2, label="rappel")
    ax.plot(ts, F, color=RED, lw=2.4, label="F1")
    ax.axvline(meta["threshold"], color=INK, lw=1, ls="--")
    ax.text(meta["threshold"] + 0.01, 0.95, f"seuil choisi sur la validation : {meta['threshold']:.2f}",
            fontsize=9)
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("seuil de décision")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Effet du seuil sur le test — {meta['display_name']}")
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "compromis_seuil.png")


def fig_confusion_expliquee(res):
    meta, y, p = res
    yp = p >= meta["threshold"]
    tn = int(((y == 0) & ~yp).sum()); fp = int(((y == 0) & yp).sum())
    fn = int(((y == 1) & ~yp).sum()); tp = int(((y == 1) & yp).sum())
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.grid(False)
    cells = [((0, 1), tn, "Vrais négatifs (TN)\npas de HO prévu,\npas de HO", "#e0f2fe"),
             ((1, 1), fp, "Faux positifs (FP)\nalarme inutile\n(préparation pour rien)", "#fee2e2"),
             ((0, 0), fn, "Faux négatifs (FN)\nHO non anticipé\n(le cas coûteux)", "#fee2e2"),
             ((1, 0), tp, "Vrais positifs (TP)\nHO bien anticipé", "#dcfce7")]
    for (x, yy), n, txt, c in cells:
        ax.add_patch(Rectangle((x, yy), 1, 1, facecolor=c, edgecolor="white", lw=3))
        ax.text(x + 0.5, yy + 0.62, f"{n}", ha="center", fontsize=18, fontweight="bold")
        ax.text(x + 0.5, yy + 0.3, txt, ha="center", va="center", fontsize=8.5)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["prédit : 0", "prédit : 1 (alarme)"])
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["réel : 1\n(HO dans 3 s)", "réel : 0"])
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    ax.set_title(f"Matrice de confusion — {meta['display_name']} sur le test\n"
                 f"précision = TP/(TP+FP) = {prec:.2f}   rappel = TP/(TP+FN) = {rec:.2f}")
    save(fig, "matrice_confusion_expliquee.png")


# ── 8. Images extraites des notebooks exécutés ────────────────────────────────

NOTEBOOK_IMAGES = {
    # nom du fichier  : (notebook, index de l'image dans l'ordre des sorties)
    "nb06_realisme.png":   ("06_simulation_realism.ipynb", 0),
    "nb09_shap.png":       ("09_shap_explanations.ipynb", 1),
    "nb10_courbes_pr.png": ("10_model_comparison.ipynb", 1),
}


def extract_notebook_images():
    for name, (nb, k) in NOTEBOOK_IMAGES.items():
        path = ROOT / "notebooks" / nb
        if not path.exists():
            continue
        imgs = []
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            for out in cell.get("outputs", []):
                png = out.get("data", {}).get("image/png")
                if png:
                    imgs.append(png if isinstance(png, str) else "".join(png))
        if len(imgs) > k:
            (OUT / name).write_bytes(base64.b64decode(imgs[k]))
            print(f"  {name}  (extrait de {nb})")


def main():
    print("Figures du cours -> docs/cours/img/")
    fig_echelles()
    fig_grille()
    fig_a3()
    fig_filtre_l3()
    fig_cqi()
    res = _champion_test_probs()
    if res:
        fig_seuil(res)
        fig_confusion_expliquee(res)
    extract_notebook_images()


if __name__ == "__main__":
    main()
