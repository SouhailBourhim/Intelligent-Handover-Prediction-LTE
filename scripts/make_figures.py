"""
Render the README / docs figures into docs/img/.

Usage
-----
    python scripts/make_figures.py            # uses data/raw/dataset.csv, simulates it if missing

Every figure is derived from the simulator modules in src/ or from the
simulated dataset, except the model-comparison chart, which reads
reports/evaluation.txt when present and otherwise falls back to the
documented run (the table in README.md).

Only numpy / pandas / matplotlib are required — no model training.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.handover_logic import HO, ho_failure_prob, velocity_ttt  # noqa: E402
from src.radio_model import (  # noqa: E402
    RADIO, _CQI_TABLE, p_los, path_loss_los, path_loss_nlos,
)

RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
EVAL_PATH = ROOT / "reports" / "evaluation.txt"
OUT_DIR = ROOT / "docs" / "img"

BS_POS = [(250.0, 250.0), (750.0, 250.0), (250.0, 750.0), (750.0, 750.0)]

# ── Style ─────────────────────────────────────────────────────────────────────

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
CELL_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]
EVENT_COLORS = {"A3": "#3b82f6", "A4": "#10b981", "A5": "#ef4444"}
POS = "#ef4444"
NEG = "#94a3b8"

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK,
    "ytick.color": INK,
    "legend.frameon": False,
})


def _save(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def load_dataset() -> pd.DataFrame:
    if RAW_PATH.exists():
        return pd.read_csv(RAW_PATH)
    print("  data/raw/dataset.csv not found - running the simulator (seed 42)...")
    import simulate
    df = simulate.add_handover_soon_label(simulate.run_simulation(seed=42))
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    return df


# ── 1. Network map ────────────────────────────────────────────────────────────

def fig_network_map(df: pd.DataFrame):
    """Best-server map (median path loss, no fading) + UE tracks + HO attempts."""
    xs = np.linspace(0, 1000, 401)
    X, Y = np.meshgrid(xs, xs)
    rsrp = []
    for bx, by in BS_POS:
        d = np.hypot(X - bx, Y - by).ravel()
        # Expected path loss: LOS/NLOS mixed by the 3GPP LOS probability
        pl = np.array([p_los(v) * path_loss_los(v) + (1 - p_los(v)) * path_loss_nlos(v)
                       for v in d]).reshape(X.shape)
        rsrp.append(RADIO.tx_power_dbm - pl)
    rsrp = np.stack(rsrp)
    best = rsrp.argmax(axis=0)
    best_rsrp = rsrp.max(axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.grid(False)
    from matplotlib.colors import ListedColormap
    ax.imshow(best, origin="lower", extent=(0, 1000, 0, 1000),
              cmap=ListedColormap(CELL_COLORS), alpha=0.12, interpolation="nearest")
    cs = ax.contour(xs, xs, best_rsrp, levels=[-70, -60],
                    colors=MUTED, linewidths=0.6, alpha=0.7)
    ax.clabel(cs, fmt="%d dBm", fontsize=7)

    # Two vehicles + one pedestrian, first 600 s
    picks = [8, 11, 2]
    for ue in picks:
        g = df[(df.ue_id == ue) & (df.timestamp < 600)]
        kind = "vehicle" if g.ue_speed.iloc[0] > 5 else "pedestrian"
        ax.plot(g.pos_x, g.pos_y, color=INK, lw=0.9 if kind == "vehicle" else 1.6,
                alpha=0.55 if kind == "vehicle" else 0.9,
                ls="-" if kind == "vehicle" else ":")
        ax.annotate(f"UE {ue} ({kind})", (g.pos_x.iloc[0], g.pos_y.iloc[0]),
                    fontsize=8, color=INK, xytext=(4, 4), textcoords="offset points")
        ho = g[g.handover_event == 1]
        for ev, c in EVENT_COLORS.items():
            ok = ho[(ho.event_type == ev) & (ho.handover_failure == 0)]
            bad = ho[(ho.event_type == ev) & (ho.handover_failure == 1)]
            ax.scatter(ok.pos_x, ok.pos_y, s=26, color=c, edgecolor="white", lw=0.6, zorder=4)
            ax.scatter(bad.pos_x, bad.pos_y, s=34, color=c, marker="X", zorder=4)

    for i, (bx, by) in enumerate(BS_POS):
        ax.scatter(bx, by, marker="^", s=220, color=CELL_COLORS[i], edgecolor=INK, lw=1, zorder=5)
        ax.annotate(f"eNB {i}", (bx, by), xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color=INK)

    handles = [
        Line2D([], [], marker="^", ls="", color=MUTED, markeredgecolor=INK, ms=10, label="eNB (46 dBm, 2 GHz)"),
        Line2D([], [], color=INK, lw=1, label="vehicle track (8–20 m/s)"),
        Line2D([], [], color=INK, lw=1, ls=":", label="pedestrian track (0.5–2 m/s)"),
    ] + [
        Line2D([], [], marker="o", ls="", color=c, ms=6, label=f"{ev} handover") for ev, c in EVENT_COLORS.items()
    ] + [Line2D([], [], marker="X", ls="", color=MUTED, ms=7, label="failed handover (RLF)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8.5)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Simulated network: 4 cells, 1 km², best-server areas and UE tracks (first 600 s)")
    _save(fig, "network_map.png")


# ── 2. Radio model ────────────────────────────────────────────────────────────

def fig_radio_model():
    d = np.linspace(10, 800, 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    ax = axes[0]
    ax.plot(d, [path_loss_los(v) for v in d], color=CELL_COLORS[0], lw=2, label="LOS")
    ax.plot(d, [path_loss_nlos(v) for v in d], color=CELL_COLORS[3], lw=2, label="NLOS")
    ax.set_xlabel("distance to eNB (m)")
    ax.set_ylabel("path loss (dB)")
    ax.set_title("3GPP UMa path loss @ 2 GHz")
    ax.legend()

    ax = axes[1]
    ax.plot(d, [p_los(v) for v in d], color=INK, lw=2)
    ax.fill_between(d, [p_los(v) for v in d], color=CELL_COLORS[0], alpha=0.12)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("distance to eNB (m)")
    ax.set_ylabel("P(LOS)")
    ax.set_title("LOS probability (TR 36.873 UMa)")

    ax = axes[2]
    th = [t for t, _ in _CQI_TABLE] + [30]
    cq = [c for _, c in _CQI_TABLE]
    ax.step(th[:-1] + [th[-1]], cq + [cq[-1]], where="post", color=CELL_COLORS[2], lw=2)
    ax.set_xlim(-10, 25)
    ax.set_ylim(0, 16)
    ax.set_xlabel("SINR (dB)")
    ax.set_ylabel("CQI")
    ax.set_title("SINR → CQI (TS 36.213, 15 levels)")
    fig.tight_layout()
    _save(fig, "radio_model.png")


# ── 3. Anatomy of a handover ──────────────────────────────────────────────────

def _pick_example(df: pd.DataFrame) -> tuple[int, int]:
    """Choose a clean, isolated successful A3 handover with a visible TTT window."""
    ho = df[(df.handover_event == 1) & (df.handover_failure == 0) &
            (df.ping_pong == 0) & (df.event_type == "A3")]
    best, best_score = None, -np.inf
    for _, r in ho.iterrows():
        u, t = int(r.ue_id), int(r.timestamp)
        w = df[(df.ue_id == u) & df.timestamp.between(t - 20, t + 13)]
        # isolated: no other HO attempt in the window or just after it
        if len(w) < 34 or w.handover_event.sum() != 1 or w.rlf_flag.sum() > 0:
            continue
        if velocity_ttt(r.ue_speed) < 2:
            continue
        gap = (w.l3_rsrp_neighbor - w.l3_rsrp_serving).values
        pre = gap[:20]
        # reward a steady approach towards the crossing, penalise jitter
        score = np.corrcoef(np.arange(20), pre)[0, 1] - 0.05 * np.abs(np.diff(pre)).mean()
        if score > best_score:
            best, best_score = (u, t), score
    return best


def fig_handover_anatomy(df: pd.DataFrame):
    u, t = _pick_example(df)
    w = df[(df.ue_id == u) & df.timestamp.between(t - 20, t + 10)].copy()
    w["rel"] = w.timestamp - t
    ttt = velocity_ttt(w.ue_speed.iloc[0])
    margin = HO.ho_margin_db

    fig, axes = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 0.9]})

    # (a) L3-filtered RSRP of serving cell vs best neighbour
    ax = axes[0]
    ax.plot(w.rel, w.rsrp_serving, color=CELL_COLORS[0], lw=0.8, alpha=0.35, label="serving RSRP (raw)")
    ax.plot(w.rel, w.l3_rsrp_serving, color=CELL_COLORS[0], lw=2.2, label="serving RSRP (L3-filtered)")
    ax.plot(w.rel, w.l3_rsrp_neighbor, color=CELL_COLORS[1], lw=2.2, label="best neighbour RSRP (L3-filtered)")
    ax.axvline(0, color=INK, lw=1.2)
    ax.annotate("handover executed\n(serving ↔ neighbour swap)", (0, ax.get_ylim()[1]),
                xytext=(6, -26), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title(f"Anatomy of one A3 handover — UE {u}, speed {w.ue_speed.iloc[0]:.1f} m/s, "
                 f"TTT = {ttt} steps")
    ax.legend(loc="lower left", fontsize=8.5)

    # (b) A3 entering condition: neighbour − serving vs margin
    ax = axes[1]
    gap = w.l3_rsrp_neighbor - w.l3_rsrp_serving
    pre = w.rel <= 0
    ax.plot(w.rel[pre], gap[pre], color=INK, lw=2, label="neighbour − serving (L3)")
    ax.plot(w.rel[~pre], gap[~pre], color=INK, lw=1, alpha=0.3)
    ax.axhline(margin, color=POS, ls="--", lw=1.2, label=f"A3 offset (+{margin:.0f} dB)")
    met = pre & (gap > margin)
    ax.fill_between(w.rel, gap.min() - 2, margin, where=met, step="mid",
                    color=POS, alpha=0.12, label="A3 condition met → TTT counting")
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_ylabel("dB")
    ax.legend(loc="lower right", fontsize=8.5)

    # (c) the label
    ax = axes[2]
    ax.grid(False)
    colors = [POS if v == 1 else NEG for v in w.handover_soon]
    ax.bar(w.rel, np.ones(len(w)), width=0.9, color=colors)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks([])
    ax.set_ylabel("label", rotation=0, ha="right", va="center")
    ax.set_xlabel("time relative to handover (s)")
    ax.legend(handles=[Patch(color=POS, label="handover_soon = 1  (successful HO in next 3 s)"),
                       Patch(color=NEG, label="handover_soon = 0")],
              loc="upper left", bbox_to_anchor=(0, -0.55), ncol=2, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "handover_anatomy.png")


# ── 4. Dataset overview ───────────────────────────────────────────────────────

def fig_dataset_overview(df: pd.DataFrame):
    ho = df[df.handover_event == 1]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    ax = axes[0]
    rate = df.handover_soon.mean()
    ax.bar(["0  no HO soon", "1  HO soon"], [1 - rate, rate], color=[NEG, POS])
    for i, v in enumerate([1 - rate, rate]):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Label balance ({len(df):,} rows)")
    ax.set_ylabel("share of rows")

    ax = axes[1]
    ev = ho.event_type.value_counts().reindex(["A3", "A4", "A5"])
    ax.bar(ev.index, ev.values, color=[EVENT_COLORS[e] for e in ev.index])
    for i, v in enumerate(ev.values):
        ax.text(i, v + 20, f"{v / len(ho):.0%}", ha="center", fontweight="bold")
    ax.set_ylim(0, ev.max() * 1.15)
    ax.set_title(f"Trigger event ({len(ho):,} HO attempts)")
    ax.set_ylabel("attempts")

    ax = axes[2]
    succ = ((ho.handover_failure == 0) & (ho.ping_pong == 0)).sum()
    pp = ((ho.handover_failure == 0) & (ho.ping_pong == 1)).sum()
    fail = (ho.handover_failure == 1).sum()
    vals = [succ, pp, fail]
    ax.bar(["success", "ping-pong", "failed → RLF"], vals,
           color=[CELL_COLORS[2], CELL_COLORS[1], CELL_COLORS[3]])
    for i, v in enumerate(vals):
        ax.text(i, v + 20, f"{v / len(ho):.0%}", ha="center", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title("Handover outcome")

    ax = axes[3]
    bins = np.linspace(-40, 60, 60)
    ax.hist(df.loc[df.handover_soon == 0, "rsrp_diff"], bins=bins, density=True,
            color=NEG, alpha=0.7, label="label 0")
    ax.hist(df.loc[df.handover_soon == 1, "rsrp_diff"], bins=bins, density=True,
            color=POS, alpha=0.6, label="label 1")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("rsrp_diff = serving − neighbour (dB)")
    ax.set_yticks([])
    ax.set_title("Strongest single signal")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "dataset_overview.png")


# ── 5. Temporal split ─────────────────────────────────────────────────────────

def fig_temporal_split(df: pd.DataFrame):
    ts = np.sort(df.timestamp.unique())
    n = len(ts)
    cut1, cut2 = ts[int(n * 0.70)], ts[int(n * 0.85)]

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.grid(False)
    for (a, b, name, c) in [(ts[0], cut1, "train 70 %", CELL_COLORS[0]),
                            (cut1, cut2, "val 15 %", CELL_COLORS[1]),
                            (cut2, ts[-1], "test 15 %", CELL_COLORS[2])]:
        ax.axvspan(a, b, color=c, alpha=0.10)
        ax.text((a + b) / 2, df.ue_id.max() + 1.1, name, ha="center", fontweight="bold", color=c)
    ho = df[df.handover_event == 1]
    ax.scatter(ho.timestamp, ho.ue_id, s=3, color=INK, alpha=0.6, lw=0)
    ax.set_ylim(-0.8, df.ue_id.max() + 2)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_yticks(range(0, df.ue_id.max() + 1, 2))
    ax.set_xlabel("simulation time (s)")
    ax.set_ylabel("UE id")
    ax.set_title("Temporal split — every dot is a handover attempt; split is by timestamp, never shuffled")
    _save(fig, "temporal_split.png")


# ── 6. Failure model ──────────────────────────────────────────────────────────

def fig_failure_model():
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    sinr = np.linspace(-25, 20, 200)
    for spd, ls in [(1.5, ":"), (14, "-")]:
        axes[0].plot(sinr, [ho_failure_prob(s, spd, -70, 0) for s in sinr], color=INK, ls=ls,
                     lw=2, label=f"speed {spd} m/s" + (" (vehicle)" if spd > 5 else " (pedestrian)"))
    axes[0].set_xlabel("serving SINR at HO (dB)")
    axes[0].set_ylabel("P(HO failure)")
    axes[0].set_title("SINR  (weight 0.50)")
    axes[0].legend(fontsize=8.5)

    tgt = np.linspace(-110, -55, 200)
    axes[1].plot(tgt, [ho_failure_prob(0, 10, r, 0) for r in tgt], color=CELL_COLORS[0], lw=2)
    axes[1].set_xlabel("target-cell RSRP (dBm)")
    axes[1].set_title("Target RSRP  (weight 0.15)")
    axes[1].invert_xaxis()

    steps = np.arange(0, 40)
    axes[2].plot(steps, [ho_failure_prob(0, 10, -70, s) for s in steps], color=CELL_COLORS[3], lw=2)
    axes[2].set_xlabel("steps already spent below −85 dBm")
    axes[2].set_title("Sustained low RSRP  (weight 0.20)")
    for ax in axes:
        ax.axhline(HO.fail_cap, color=MUTED, ls="--", lw=0.8)
    axes[0].text(-24, HO.fail_cap + 0.015, "cap 0.75", fontsize=8, color=MUTED)
    axes[0].set_ylim(0, 0.8)
    fig.suptitle("Handover failure model — each panel varies one factor, others held at typical values",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, "failure_model.png")


# ── 7. Model comparison ───────────────────────────────────────────────────────

# Documented run (README.md, Phase 4); used when reports/evaluation.txt is absent.
DOCUMENTED = {
    "Logistic Regression": (0.352, 0.869, 0.501, 0.898),
    "Random Forest":       (0.433, 0.736, 0.545, 0.911),
    "XGBoost":             (0.426, 0.778, 0.550, 0.919),
    "LSTM":                (0.342, 0.781, 0.476, 0.881),
    "GRU":                 (0.332, 0.809, 0.471, 0.865),
    "Stacking Ensemble":   (0.561, 0.403, 0.469, 0.919),
}


def _load_metrics() -> dict[str, tuple[float, float, float, float]]:
    if not EVAL_PATH.exists():
        return DOCUMENTED
    out, cur, vals = {}, None, {}
    for ln in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^[=\-\s]*([A-Za-z][A-Za-z _]+?)\s*[=\-]*$", ln.strip())
        key = ln.split(":")[0].strip().lower() if ":" in ln else ""
        if ":" in ln and key.startswith(("precision", "recall", "f1", "roc")):
            try:
                vals[key.split()[0].split("-")[0]] = float(ln.split(":")[-1])
            except ValueError:
                pass
        elif m and ":" not in ln and m.group(1).strip():
            if cur and len(vals) >= 4:
                out[cur] = (vals["precision"], vals["recall"], vals["f1"], vals["roc"])
            cur, vals = m.group(1).strip().replace("_", " ").title(), {}
    if cur and len(vals) >= 4:
        out[cur] = (vals["precision"], vals["recall"], vals["f1"], vals["roc"])
    return out if len(out) >= 3 else DOCUMENTED


def fig_model_comparison():
    metrics = _load_metrics()
    names = list(metrics)
    arr = np.array([metrics[n] for n in names])
    score = 0.6 * arr[:, 2] + 0.4 * arr[:, 3]
    champ = int(np.argmax(score))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1.3]})
    labels = ["Precision", "Recall", "F1", "ROC-AUC"]
    colors = ["#94a3b8", "#64748b", POS, CELL_COLORS[0]]
    x = np.arange(len(names))
    bw = 0.2
    for j, (lab, c) in enumerate(zip(labels, colors)):
        ax.bar(x + (j - 1.5) * bw, arr[:, j], bw, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-set metrics at threshold 0.5")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5)

    order = np.argsort(score)
    ax2.barh([names[i] for i in order], score[order],
             color=[POS if i == champ else NEG for i in order])
    for k, i in enumerate(order):
        ax2.text(score[i] + 0.004, k, f"{score[i]:.3f}", va="center", fontsize=8.5)
    ax2.set_xlim(0.58, 0.72)
    ax2.set_title("Champion score = 0.6·F1 + 0.4·AUC")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def main():
    print("Rendering figures -> docs/img/")
    df = load_dataset()
    fig_network_map(df)
    fig_radio_model()
    fig_handover_anatomy(df)
    fig_dataset_overview(df)
    fig_temporal_split(df)
    fig_failure_model()
    fig_model_comparison()


if __name__ == "__main__":
    main()
