"""Bloco 5: Geração das 4 figuras finais em PDF vetorial no padrão IEEE.

Especificações:
- PDF vetorial, fonte base 8pt, rótulos em português, paleta legível em P&B
  (cores de alto contraste em luminância + marcadores + estilos de linha distintos).
- Estritamente dados validados (sem PELT_L1_gauss).
- Larguras IEEE: Fig. 1 = 7.16 pol (2 colunas); Figs. 2, 3, 4 = 3.5 pol (1 coluna).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cpd")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cpd.detectors import filter_edivisive_by_alpha
from cpd.experiments.calibration import poisson_rate_ci_95

REPO_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_DIR / "results/tables"
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
REPORTS_DIR = REPO_DIR / "reports"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Estilo P&B-safe (luminância monotônica + marcadores + traços distintos)
STYLE_MAP = {
    "CUSUM_perfil":           {"label": "CUSUM",          "color": "#000000", "ls": "-",  "marker": "o"},
    "PELT_L2_perfil":         {"label": "PELT-L2",        "color": "#1b7837", "ls": "--", "marker": "s"},
    "Binseg_L2_perfil":       {"label": "Binseg-L2",      "color": "#2166ac", "ls": "-.", "marker": "^"},
    "EDiv_Means_calibrado":   {"label": "E-Div-Means",    "color": "#d95f02", "ls": ":",  "marker": "D"},
    "EDiv_Medians_calibrado": {"label": "E-Div-Medians",  "color": "#7570b3", "ls": (0, (3, 1, 1, 1)), "marker": "v"},
}

DETS_5 = list(STYLE_MAP.keys())


def ieee_fig1_f1_vs_magnitude(cell_df: pd.DataFrame) -> None:
    sub = cell_df[
        (cell_df["k_tol"] == 5)
        & (cell_df["n"] == 500)
        & (cell_df["detector"].isin(DETS_5))
    ].copy()
    agg = sub.groupby(["detector", "perfil_ruido", "magnitude"], as_index=False)[["tp", "fp", "fn"]].sum()
    agg["p"] = np.where(agg["tp"] + agg["fp"] > 0, agg["tp"] / (agg["tp"] + agg["fp"]), 0.0)
    agg["r"] = np.where(agg["tp"] + agg["fn"] > 0, agg["tp"] / (agg["tp"] + agg["fn"]), 0.0)
    agg["f1"] = np.where(agg["p"] + agg["r"] > 0, 2 * agg["p"] * agg["r"] / (agg["p"] + agg["r"]), 0.0)

    perfis = [("gaussian", "Gaussiano"), ("student_t", "Student-t (ν=3)"), ("ar1", "AR(1) (φ=0,7)")]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharex=True, sharey=True)
    for idx, (pkey, ptitle) in enumerate(perfis):
        ax = axes[idx]
        psub = agg[agg["perfil_ruido"] == pkey]
        for det in DETS_5:
            st = STYLE_MAP[det]
            dsub = psub[psub["detector"] == det].sort_values("magnitude")
            ax.plot(
                dsub["magnitude"], dsub["f1"],
                color=st["color"], linestyle=st["ls"], marker=st["marker"],
                markersize=4, linewidth=1.2, label=st["label"],
            )
        ax.set_title(ptitle)
        ax.set_xlabel("Magnitude (σ)")
        ax.set_xticks([0.5, 1.0, 2.0, 5.0])
        ax.set_ylim(-0.03, 1.05)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel("Micro-F1 (k=5, n=500)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(REPORTS_DIR / "ieee_fig1_f1_vs_magnitude.pdf", format="pdf")
    plt.close(fig)


def ieee_fig2_far_por_detector_perfil(far_df: pd.DataFrame) -> None:
    sub = far_df[far_df["detector"].isin(DETS_5)].copy()
    sub["exp"] = sub["num_series"] * sub["n"]
    exp_df = sub.groupby(["perfil_ruido", "detector"], as_index=False)[["total_alarms", "exp"]].sum()

    cis = [poisson_rate_ci_95(int(r["total_alarms"]), int(r["exp"])) for _, r in exp_df.iterrows()]
    exp_df["far_1k"] = (exp_df["total_alarms"] / exp_df["exp"]) * 1000.0
    exp_df["ci_l"] = [c[0] for c in cis]
    exp_df["ci_h"] = [c[1] for c in cis]

    perfis = [
        ("gaussian", "Gaussiano"),
        ("student_t", "Student-t"),
        ("outliers", "Outliers\n(p=2%, 5σ)"),
        ("bimodal", "Bimodal"),
        ("ar1", "AR(1)\n(φ=0,7)"),
    ]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    y_base = np.arange(len(perfis))
    offsets = np.linspace(-0.28, 0.28, len(DETS_5))

    for d_i, det in enumerate(DETS_5):
        st = STYLE_MAP[det]
        xs, xerr_l, xerr_h, ys = [], [], [], []
        for p_i, (pkey, _) in enumerate(perfis):
            row = exp_df[(exp_df["perfil_ruido"] == pkey) & (exp_df["detector"] == det)].iloc[0]
            val = max(float(row["far_1k"]), 0.01)
            cl = max(float(row["ci_l"]), 0.01)
            ch = max(float(row["ci_h"]), 0.01)
            xs.append(val)
            xerr_l.append(max(0.0, val - cl))
            xerr_h.append(max(0.0, ch - val))
            ys.append(y_base[p_i] + offsets[d_i])
        ax.errorbar(
            xs, ys, xerr=[xerr_l, xerr_h],
            fmt=st["marker"], color=st["color"], markersize=3.8,
            elinewidth=0.9, capsize=1.8, linestyle="none", label=st["label"],
        )

    ax.axvline(0.2, color="#b2182b", linestyle="--", linewidth=0.9, label="Alvo (0,2/1k)")
    ax.set_xscale("log")
    ax.set_xlim(0.008, 30.0)
    ax.set_yticks(y_base)
    ax.set_yticklabels([p[1] for p in perfis])
    ax.set_xlabel("FAR por 1.000 commits (escala log, IC 95%)")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=True, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ieee_fig2_far_por_detector_perfil.pdf", format="pdf")
    plt.close(fig)


def ieee_fig3_prob_atraso(raw_df: pd.DataFrame, manifest: dict) -> None:
    sub = raw_df[
        (raw_df["magnitude"].isin([2.0, 5.0]))
        & (raw_df["ground_truth"].apply(len) == 1)
    ].copy()
    details = manifest["detalhes_por_alvo"]

    delays_by: dict[tuple[float, str], list[float]] = {(m, d): [] for m in (2.0, 5.0) for d in DETS_5}
    for row in sub.to_dict(orient="records"):
        det_name = row["detector"]
        gt0 = int(row["ground_truth"][0])
        mag = float(row["magnitude"])
        perfil = row["perfil_ruido"]
        n_str = str(int(row["n"]))

        if det_name in ("EDiv_Means_base", "EDiv_Medians_base"):
            hist = json.loads(row["parametros"])["division_history"]
            prefix = "EDiv_Means" if det_name == "EDiv_Means_base" else "EDiv_Medians"
            a02 = float(details[perfil][n_str]["0.2"][prefix]["param"])
            target_name = f"{prefix}_calibrado"
            dets = filter_edivisive_by_alpha(hist, a02)
        elif det_name in DETS_5:
            target_name = det_name
            dets = [int(x) for x in row["indices_detectados"]]
        else:
            continue

        after = [d - gt0 for d in dets if d >= gt0]
        delays_by[(mag, target_name)].append(float(min(after)) if after else np.inf)

    d_grid = np.arange(0, 51)
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2), sharex=True, sharey=True)
    for r_i, mag in enumerate((2.0, 5.0)):
        ax = axes[r_i]
        for det in DETS_5:
            st = STYLE_MAP[det]
            arr = np.asarray(delays_by[(mag, det)])
            probs = [float(np.mean(arr <= d)) for d in d_grid]
            ax.plot(d_grid, probs, color=st["color"], linestyle=st["ls"], linewidth=1.2, label=st["label"])
        ax.set_title(f"Magnitude {int(mag)}σ (k_true=1)")
        ax.set_ylabel("P(atraso ≤ d)")
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.25, linewidth=0.5)
    axes[1].set_xlabel("Atraso máximo d (commits após gt)")
    axes[1].legend(loc="lower right", frameon=True, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ieee_fig3_prob_atraso.pdf", format="pdf")
    plt.close(fig)


def ieee_fig4_custo_computacional(cost_df: pd.DataFrame) -> None:
    bench_map = {
        "CUSUM":        STYLE_MAP["CUSUM_perfil"],
        "Binseg_L2":    STYLE_MAP["Binseg_L2_perfil"],
        "PELT_L2":      STYLE_MAP["PELT_L2_perfil"],
        "EDiv_Means":   STYLE_MAP["EDiv_Means_calibrado"],
        "EDiv_Medians": STYLE_MAP["EDiv_Medians_calibrado"],
    }
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for det, st in bench_map.items():
        dsub = cost_df[cost_df["detector"] == det].sort_values("n")
        exp_val = float(dsub["time_exp"].iloc[0])
        y = dsub["time_med_ms"].to_numpy()
        yerr_l = np.maximum(0.0, y - dsub["time_min_ms"].to_numpy())
        yerr_h = np.maximum(0.0, dsub["time_max_ms"].to_numpy() - y)
        ax.errorbar(
            dsub["n"], y, yerr=[yerr_l, yerr_h],
            color=st["color"], linestyle=st["ls"], marker=st["marker"],
            markersize=3.8, linewidth=1.1, capsize=1.8,
            label=f"{st['label']} (b={exp_val:.2f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Tamanho da série (n commits)")
    ax.set_ylabel("Tempo mediano por série (ms)")
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=6.2)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "ieee_fig4_custo_computacional.pdf", format="pdf")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_df = pd.read_csv(TABLES_DIR / "metrics_by_cell.csv")
    far_df = pd.read_csv(TABLES_DIR / "far_by_detector_profile_n.csv")
    cost_df = pd.read_csv(TABLES_DIR / "cost_benchmark.csv")
    raw_df = pd.read_parquet(RAW_PARQUET)
    if "valido" in raw_df.columns:
        raw_df = raw_df[raw_df["valido"].fillna(True)].copy()
    raw_df = raw_df[raw_df["detector"] != "PELT_L1_gauss"].copy()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    ieee_fig1_f1_vs_magnitude(cell_df)
    ieee_fig2_far_por_detector_perfil(far_df)
    ieee_fig3_prob_atraso(raw_df, manifest)
    ieee_fig4_custo_computacional(cost_df)
    print("Figuras IEEE PDF geradas em:", REPORTS_DIR)


if __name__ == "__main__":
    main()
