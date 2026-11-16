"""Bloco 4: Diagnóstico empírico do platô de Q(tau) em EDiv_Medians vs EDiv_Means (5σ).

Isola as séries de student_t (n=500, 5σ, k_true=1) e gaussian (n=1000, 5σ, k_true=1)
em que EDiv_Medians_calibrado apresentou detecção a mais de 10 commits do ponto
verdadeiro, computa Q(argmax), Q(gt) e amplitude max-min em [gt, gt+50] para ambos
os métodos, salva tabela CSV e gera painel visual com 3 séries exemplares.
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
import yaml

from cpd.detectors import EDivisiveMeansDetector, EDivisiveMediansDetector, filter_edivisive_by_alpha
from cpd.experiments.runner import generate_scenario_series

REPO_DIR = Path(__file__).resolve().parent.parent
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
CONFIG_PATH = REPO_DIR / "configs/experiments.yaml"
OUT_CSV = REPO_DIR / "results/tables/plateau_diagnosis_edivisive.csv"
OUT_PDF = REPO_DIR / "reports/fig_diagnostico_plato_edivisive.pdf"
OUT_PNG = REPO_DIR / "reports/fig_diagnostico_plato_edivisive.png"


def main() -> None:
    raw_df = pd.read_parquet(RAW_PARQUET)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    noise_params_map = {p["tipo"]: p["params"] for p in cfg["grid"]["perfis_ruido"]}
    details = manifest["detalhes_por_alvo"]

    sub = raw_df[
        (raw_df["detector"] == "EDiv_Medians_base")
        & (raw_df["magnitude"] == 5.0)
        & (
            ((raw_df["perfil_ruido"] == "student_t") & (raw_df["n"] == 500))
            | ((raw_df["perfil_ruido"] == "gaussian") & (raw_df["n"] == 1000))
        )
    ].copy()
    sub = sub[sub["ground_truth"].apply(len) == 1]

    records = []
    panel_candidates = []

    for _, row in sub.iterrows():
        perfil = row["perfil_ruido"]
        n_val = int(row["n"])
        gt = int(row["ground_truth"][0])
        seed_idx = int(row["semente"])
        cenario_id = row["cenario_id"]

        hist = json.loads(row["parametros"])["division_history"]
        a02 = float(details[perfil][str(n_val)]["0.2"]["EDiv_Medians"]["param"])
        dets = filter_edivisive_by_alpha(hist, a02)

        # Isola séries onde Medians falhou (alguma detecção a mais de 10 commits do gt)
        if not dets or not any(abs(d - gt) > 10 for d in dets):
            continue

        sc_cfg = {
            "cenario_id": cenario_id,
            "n": n_val,
            "perfil_ruido": perfil,
            "noise_params": noise_params_map[perfil],
            "k_true": 1,
            "magnitude_sigma": 5.0,
        }
        s_obj = generate_scenario_series(sc_cfg, seed_idx)
        x = s_obj.values

        dm = EDivisiveMeansDetector(min_size=25)
        dmed = EDivisiveMediansDetector(min_size=25, exact_medians=True)

        qm = dm._calculate_t_stats(dm.compute_distance_matrix(x))
        qmed = dmed._calculate_t_stats(dmed.compute_distance_matrix(x))

        win_end = min(len(x) - 25, gt + 50)
        win_slice_m = qm[gt : win_end + 1] if win_end >= gt else np.array([qm[gt]])
        win_slice_med = qmed[gt : win_end + 1] if win_end >= gt else np.array([qmed[gt]])

        # Apenas índices válidos [25, n-25]
        valid_m = qm[25 : n_val - 25 + 1]
        valid_med = qmed[25 : n_val - 25 + 1]
        amp_m = float(np.max(win_slice_m) - np.min(win_slice_m))
        amp_med = float(np.max(win_slice_med) - np.min(win_slice_med))

        argmax_m = int(np.argmax(qm))
        argmax_med = int(np.argmax(qmed))

        records.append({
            "perfil": perfil,
            "n": n_val,
            "cenario_id": cenario_id,
            "semente": seed_idx,
            "gt": gt,
            "dets_medians": str(dets),
            "means_argmax": argmax_m,
            "means_Q_argmax": float(qm[argmax_m]),
            "means_Q_gt": float(qm[gt]),
            "means_amp_gt_gt50": amp_m,
            "means_rel_amp_pct": 100.0 * amp_m / float(qm[argmax_m]) if qm[argmax_m] > 0 else np.nan,
            "medians_argmax": argmax_med,
            "medians_Q_argmax": float(qmed[argmax_med]),
            "medians_Q_gt": float(qmed[gt]),
            "medians_amp_gt_gt50": amp_med,
            "medians_rel_amp_pct": 100.0 * amp_med / float(qmed[argmax_med]) if qmed[argmax_med] > 0 else np.nan,
        })

        # Candidatos para o painel (priorizar séries onde argmax_med != gt por >10 commits)
        if len(panel_candidates) < 6 and abs(argmax_med - gt) > 10:
            panel_candidates.append((perfil, n_val, seed_idx, gt, argmax_m, argmax_med, qm, qmed))

    # Se menos de 3 com argmax_med != gt > 10, completar com as primeiras falhas
    if len(panel_candidates) < 3:
        for _, row in sub.iterrows():
            if len(panel_candidates) >= 3:
                break
            perfil, n_val, gt, seed_idx = row["perfil_ruido"], int(row["n"]), int(row["ground_truth"][0]), int(row["semente"])
            hist = json.loads(row["parametros"])["division_history"]
            a02 = float(details[perfil][str(n_val)]["0.2"]["EDiv_Medians"]["param"])
            dets = filter_edivisive_by_alpha(hist, a02)
            if not dets or not any(abs(d - gt) > 10 for d in dets):
                continue
            if any(c[0] == perfil and c[2] == seed_idx for c in panel_candidates):
                continue
            sc_cfg = {"cenario_id": row["cenario_id"], "n": n_val, "perfil_ruido": perfil, "noise_params": noise_params_map[perfil], "k_true": 1, "magnitude_sigma": 5.0}
            x = generate_scenario_series(sc_cfg, seed_idx).values
            dm = EDivisiveMeansDetector(min_size=25)
            dmed = EDivisiveMediansDetector(min_size=25, exact_medians=True)
            qm = dm._calculate_t_stats(dm.compute_distance_matrix(x))
            qmed = dmed._calculate_t_stats(dmed.compute_distance_matrix(x))
            panel_candidates.append((perfil, n_val, seed_idx, gt, int(np.argmax(qm)), int(np.argmax(qmed)), qm, qmed))

    df_out = pd.DataFrame(records)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"Total de séries com falha (|det - gt| > 10) em EDiv_Medians: {len(df_out)}")
    summary = df_out.groupby(["perfil", "n"], as_index=False).agg(
        n_falhas=("semente", "count"),
        means_Q_argmax_med=("means_Q_argmax", "median"),
        means_Q_gt_med=("means_Q_gt", "median"),
        means_amp_med=("means_amp_gt_gt50", "median"),
        means_rel_amp_med=("means_rel_amp_pct", "median"),
        medians_Q_argmax_med=("medians_Q_argmax", "median"),
        medians_Q_gt_med=("medians_Q_gt", "median"),
        medians_amp_med=("medians_amp_gt_gt50", "median"),
        medians_rel_amp_med=("medians_rel_amp_pct", "median"),
    )
    print(summary.to_string(index=False))

    # Painel visual com 3 séries (normalizado por Q_max para comparar forma do platô)
    chosen = panel_candidates[:3]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), sharey=True)
    for idx, (perfil, n_val, seed_idx, gt, am_m, am_med, qm, qmed) in enumerate(chosen):
        ax = axes[idx]
        taus = np.arange(25, n_val - 25 + 1)
        qm_norm = qm[25 : n_val - 25 + 1] / np.max(qm)
        qmed_norm = qmed[25 : n_val - 25 + 1] / np.max(qmed)
        ax.plot(taus, qm_norm, color="black", linestyle="-", linewidth=1.1, label="E-Div-Means")
        ax.plot(taus, qmed_norm, color="#555555", linestyle="--", linewidth=1.2, label="E-Div-Medians")
        ax.axvline(gt, color="red", linestyle=":", linewidth=1.0, label=f"gt={gt}")
        ax.axvline(am_med, color="#1f77b4", linestyle="-.", linewidth=0.9, label=f"argmax Med={am_med}")
        ax.set_title(f"{perfil} (n={n_val}, s={seed_idx})")
        ax.set_xlabel("Índice $\\tau$")
        if idx == 0:
            ax.set_ylabel("$Q(\\tau) / Q_{\\max}$")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower center", frameon=True, fontsize=6.5)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf")
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"Salvos: {OUT_CSV}, {OUT_PDF}, {OUT_PNG}")


if __name__ == "__main__":
    main()
