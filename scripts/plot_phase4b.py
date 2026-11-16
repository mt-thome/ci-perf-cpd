"""Fase 4b (Item E): Geração de figuras em reports/ (300 dpi, rótulos em português).

Uso:
    .venv/bin/python scripts/plot_phase4b.py
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

REPO_DIR = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_DIR / "results/tables"
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
REPORTS_DIR = REPO_DIR / "reports"

PERFIL_LABELS = {
    "gaussian": "Gaussiano",
    "ar1": "AR(1) (φ=0,7)",
    "student_t": "Student-t (ν=3)",
    "outliers": "Outliers (p=2%, 5σ)",
    "bimodal": "Bimodal (w=0,3, 3σ)",
}

DET_ORDER = [
    "CUSUM_perfil",
    "PELT_L2_perfil",
    "PELT_L1_perfil",
    "Binseg_L2_perfil",
    "EDiv_Means_padrao",
    "EDiv_Means_calibrado",
    "EDiv_Medians_padrao",
    "EDiv_Medians_calibrado",
]


def fig1_f1_vs_magnitude(cell_df: pd.DataFrame) -> None:
    sub = cell_df[(cell_df["k_tol"] == 5) & (cell_df["n"].isin([500, 1000])) & (cell_df["detector"].isin(DET_ORDER))]
    agg = (
        sub.groupby(["detector", "perfil_ruido", "n", "magnitude"], as_index=False)[["tp", "fp", "fn"]]
        .sum()
    )
    agg["p"] = np.where(agg["tp"] + agg["fp"] > 0, agg["tp"] / (agg["tp"] + agg["fp"]), 0.0)
    agg["r"] = np.where(agg["tp"] + agg["fn"] > 0, agg["tp"] / (agg["tp"] + agg["fn"]), 0.0)
    agg["f1"] = np.where(agg["p"] + agg["r"] > 0, 2 * agg["p"] * agg["r"] / (agg["p"] + agg["r"]), 0.0)

    perfis = ["gaussian", "ar1", "student_t", "outliers", "bimodal"]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    for row_i, n_val in enumerate([500, 1000]):
        for col_i, perf in enumerate(perfis):
            ax = axes[row_i, col_i]
            p_sub = agg[(agg["n"] == n_val) & (agg["perfil_ruido"] == perf)]
            for det in DET_ORDER:
                d_sub = p_sub[p_sub["detector"] == det].sort_values("magnitude")
                if not d_sub.empty:
                    ax.plot(d_sub["magnitude"], d_sub["f1"], marker="o", linewidth=1.6, markersize=4, label=det)
            ax.set_title(f"{PERFIL_LABELS[perf]} (n={n_val})", fontsize=10)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.3)
            if row_i == 1:
                ax.set_xlabel("Magnitude da mudança (σ)")
            if col_i == 0:
                ax.set_ylabel("Micro-F1 (k=5)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=True)
    fig.suptitle("Micro-F1 (k=5) por Magnitude, Perfil de Ruído e Tamanho da Série", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(REPORTS_DIR / "fig1_f1_vs_magnitude.png", dpi=300)
    plt.close(fig)


def fig2_far_por_detector_perfil(far_df: pd.DataFrame) -> None:
    sub = far_df[far_df["detector"].isin(DET_ORDER)].copy()
    agg = sub.groupby(["perfil_ruido", "detector"], as_index=False)[["total_alarms", "num_series"]].sum()
    # Aproximar exposição média (ou somar exatamente via n)
    exp_df = sub.assign(exp=sub["num_series"] * sub["n"]).groupby(["perfil_ruido", "detector"], as_index=False)[["total_alarms", "exp"]].sum()
    from cpd.experiments.calibration import poisson_rate_ci_95
    cis = [poisson_rate_ci_95(int(r["total_alarms"]), int(r["exp"])) for _, r in exp_df.iterrows()]
    exp_df["far_1k"] = (exp_df["total_alarms"] / exp_df["exp"]) * 1000.0
    exp_df["ci_l"] = [c[0] for c in cis]
    exp_df["ci_h"] = [c[1] for c in cis]

    perfis = ["gaussian", "ar1", "student_t", "outliers", "bimodal"]
    fig, axes = plt.subplots(1, 5, figsize=(21, 5.2), sharey=False)
    for idx, perf in enumerate(perfis):
        ax = axes[idx]
        p_sub = exp_df[exp_df["perfil_ruido"] == perf].set_index("detector").reindex(DET_ORDER)
        y_pos = np.arange(len(DET_ORDER))
        vals = p_sub["far_1k"].to_numpy()
        yerr_low = np.maximum(0.0, vals - p_sub["ci_l"].to_numpy())
        yerr_high = np.maximum(0.0, p_sub["ci_h"].to_numpy() - vals)
        ax.barh(y_pos, vals, xerr=[yerr_low, yerr_high], capsize=3, alpha=0.85, color="#2b6cb0")
        ax.axvline(0.2, color="red", linestyle="--", linewidth=1.1, label="Alvo 0,2/1k")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(DET_ORDER if idx == 0 else [""] * len(DET_ORDER), fontsize=8.5)
        ax.set_title(PERFIL_LABELS[perf], fontsize=10)
        ax.set_xlabel("FAR por 1.000 commits (IC 95%)")
        ax.grid(True, axis="x", alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Taxa de Falsos Alarmes (FAR_1k) em Séries Estacionárias no Grid", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(REPORTS_DIR / "fig2_far_por_detector_perfil.png", dpi=300)
    plt.close(fig)


def fig3_distribuicao_atraso(raw_df: pd.DataFrame, manifest: dict) -> None:
    # Atraso (det - gt) dos acertos em k=10 (para capturar atraso sequencial completo do CUSUM)
    reg = raw_df[raw_df["magnitude"] > 0.0]
    delays_by_det: dict[str, list[int]] = {d: [] for d in DET_ORDER}
    details = manifest["detalhes_por_alvo"]

    for row in reg.to_dict(orient="records"):
        det_name = row["detector"]
        gt = [int(x) for x in row["ground_truth"]]
        if det_name in ("EDiv_Means_base", "EDiv_Medians_base"):
            hist = json.loads(row["parametros"])["division_history"]
            prefix = "EDiv_Means" if det_name == "EDiv_Means_base" else "EDiv_Medians"
            a02 = float(details[row["perfil_ruido"]][str(int(row["n"]))]["0.2"][prefix]["param"])
            variants = [
                (f"{prefix}_padrao", filter_edivisive_by_alpha(hist, 0.05)),
                (f"{prefix}_calibrado", filter_edivisive_by_alpha(hist, a02)),
            ]
        elif det_name in delays_by_det:
            variants = [(det_name, [int(x) for x in row["indices_detectados"]])]
        else:
            continue

        for name, dets in variants:
            pairs = sorted((abs(d - g), gi, di, d - g) for gi, g in enumerate(gt) for di, d in enumerate(dets) if abs(d - g) <= 10)
            ug, ud = set(), set()
            for _, gi, di, diff in pairs:
                if gi not in ug and di not in ud:
                    ug.add(gi); ud.add(di)
                    delays_by_det[name].append(int(diff))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = [delays_by_det[d] for d in DET_ORDER]
    ax.boxplot(data, tick_labels=DET_ORDER, showfliers=False)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Atraso de detecção (commits: detectado − verdadeiro)")
    ax.set_title("Distribuição do Atraso de Detecção (acertos com |atraso| ≤ 10 commits)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig3_distribuicao_atraso.png", dpi=300)
    plt.close(fig)


def fig4_efeito_calibracao(dpnk_df: pd.DataFrame) -> None:
    sub = dpnk_df[dpnk_df["k_tol"] == 5]
    bases = ["CUSUM", "PELT_L2", "PELT_L1", "Binseg_L2"]
    perfis = ["gaussian", "ar1", "student_t", "outliers", "bimodal"]

    # Agregar sobre n (somando tp, fp, fn)
    pairs = []
    for perf in perfis:
        for b in bases:
            for reg in ("gauss", "perfil"):
                dname = f"{b}_{reg}"
                g = sub[(sub["perfil_ruido"] == perf) & (sub["detector"] == dname)]
                tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
                p = tp / (tp + fp) if (tp + fp) else 0.0
                r = tp / (tp + fn) if (tp + fn) else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) else 0.0
                pairs.append({"perfil": perf, "base": b, "regime": reg, "f1": f1})
    pdf = pd.DataFrame(pairs)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharey=True)
    x = np.arange(len(bases))
    w = 0.36
    for i, perf in enumerate(perfis):
        ax = axes[i]
        g_vals = [pdf[(pdf["perfil"] == perf) & (pdf["base"] == b) & (pdf["regime"] == "gauss")]["f1"].iloc[0] for b in bases]
        p_vals = [pdf[(pdf["perfil"] == perf) & (pdf["base"] == b) & (pdf["regime"] == "perfil")]["f1"].iloc[0] for b in bases]
        ax.bar(x - w/2, g_vals, w, label="Calibrado em Gaussiano", color="#a0aec0")
        ax.bar(x + w/2, p_vals, w, label="Calibrado por Perfil", color="#2b6cb0")
        ax.set_xticks(x)
        ax.set_xticklabels(bases, rotation=20, fontsize=8.5)
        ax.set_title(PERFIL_LABELS[perf], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Micro-F1 (k=5)")
            ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Efeito da Calibração por Perfil vs. Calibração Gaussiana (Micro-F1, k=5)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(REPORTS_DIR / "fig4_efeito_calibracao.png", dpi=300)
    plt.close(fig)


def fig5_hazard_level(raw_df: pd.DataFrame, manifest: dict) -> None:
    reg = raw_df[raw_df["magnitude"] > 0.0]
    details = manifest["detalhes_por_alvo"]
    pts_by_det: dict[str, tuple[list[float], list[float]]] = {d: ([], []) for d in DET_ORDER}

    for row in reg.to_dict(orient="records"):
        det_name = row["detector"]
        haz_all = json.loads(row["hazard_levels_est"])
        if not haz_all:
            continue
        if det_name in ("EDiv_Means_base", "EDiv_Medians_base"):
            hist = json.loads(row["parametros"])["division_history"]
            prefix = "EDiv_Means" if det_name == "EDiv_Means_base" else "EDiv_Medians"
            a02 = float(details[row["perfil_ruido"]][str(int(row["n"]))]["0.2"][prefix]["param"])
            for reg_s, av in (("padrao", 0.05), ("calibrado", a02)):
                s_idx = set(filter_edivisive_by_alpha(hist, av))
                for h in haz_all:
                    if int(h["detected_cp"]) in s_idx:
                        pts_by_det[f"{prefix}_{reg_s}"][0].append(float(h["hazard_level_true"]))
                        pts_by_det[f"{prefix}_{reg_s}"][1].append(float(h["hazard_level_est"]))
        elif det_name in pts_by_det:
            for h in haz_all:
                pts_by_det[det_name][0].append(float(h["hazard_level_true"]))
                pts_by_det[det_name][1].append(float(h["hazard_level_est"]))

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), sharex=True, sharey=True)
    rng = np.random.default_rng(42)
    for idx, det in enumerate(DET_ORDER):
        ax = axes[idx // 4, idx % 4]
        xt, ye = np.asarray(pts_by_det[det][0]), np.asarray(pts_by_det[det][1])
        if len(xt) > 2500:
            sel = rng.choice(len(xt), size=2500, replace=False)
            xt_p, ye_p = xt[sel], ye[sel]
        else:
            xt_p, ye_p = xt, ye
        ax.scatter(xt_p, ye_p, s=6, alpha=0.25, color="#2b6cb0")
        ax.plot([-6, 6], [-6, 6], "r--", linewidth=1)
        mae = float(np.mean(np.abs(ye - xt))) if len(xt) else np.nan
        ax.set_title(f"{det}\n(MAE = {mae:.3f})", fontsize=9.5)
        ax.set_xlim(-1.0, 6.5)
        ax.set_ylim(-2.5, 8.0)
        ax.grid(True, alpha=0.3)
        if idx // 4 == 1:
            ax.set_xlabel("Hazard Level Verdadeiro")
        if idx % 4 == 0:
            ax.set_ylabel("Hazard Level Estimado")

    fig.suptitle("Hazard Level Estimado vs. Verdadeiro por Detector", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(REPORTS_DIR / "fig5_hazard_level_est_vs_true.png", dpi=300)
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cell_df = pd.read_csv(TABLES_DIR / "metrics_by_cell.csv")
    dpnk_df = pd.read_csv(TABLES_DIR / "metrics_by_detector_profile_n_k.csv")
    far_df = pd.read_csv(TABLES_DIR / "far_by_detector_profile_n.csv")
    raw_df = pd.read_parquet(RAW_PARQUET)
    if "valido" in raw_df.columns:
        raw_df = raw_df[raw_df["valido"].fillna(True)].copy()
    raw_df = raw_df[raw_df["detector"] != "PELT_L1_gauss"].copy()
    manifest = json.loads(MANIFEST_PATH.read_text())

    fig1_f1_vs_magnitude(cell_df)
    fig2_far_por_detector_perfil(far_df)
    fig3_distribuicao_atraso(raw_df, manifest)
    fig4_efeito_calibracao(dpnk_df)
    fig5_hazard_level(raw_df, manifest)
    print("5 figuras salvas em:", REPORTS_DIR)


if __name__ == "__main__":
    main()
