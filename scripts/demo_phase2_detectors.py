"""Executa uma rodada de demonstração comparativa dos detectores da Fase 2.

Testa CUSUM, E-Divisive (Means), E-Divisive (Medians), PELT e Binseg sobre uma série de exemplo
para cada um dos 5 perfis estocásticos de ruído com ground truth conhecido.
Gera uma figura comparativa em reports/fig05_phase2_detectors_demo.png.
"""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cpd.detectors import (
    RobustCUSUMDetector,
    EDivisiveDetector,
    RupturesDetector,
)
from cpd.generators.synthetic import NoiseType, generate_synthetic_series

# Configuração de Estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'figure.dpi': 150,
})

REPORTS_DIR = Path('/home/mtthome/Projetos/ci-perf-cpd/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Calibração prévia dos detectores para FPR <= 1% sob ruído Gaussiano estacionário (sementes 1000..1050)
print("Calibrando detectores em conjunto de calibração independente (FPR <= 1%)...")
calib_series = [
    generate_synthetic_series(n=300, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=s).values
    for s in range(1000, 1050)
]

calib_h = RobustCUSUMDetector.calibrate_h_for_fpr(
    calibration_series_list=calib_series,
    target_fpr=0.01,
    h_candidates=np.linspace(3.0, 10.0, 36),
)

calib_pen_pelt = RupturesDetector.calibrate_penalty_for_fpr(
    calibration_series_list=calib_series,
    algorithm="pelt",
    cost_model="l2",
    target_fpr=0.01,
    pen_candidates=np.logspace(1.0, 3.0, 20),
)

calib_pen_binseg = RupturesDetector.calibrate_penalty_for_fpr(
    calibration_series_list=calib_series,
    algorithm="binseg",
    cost_model="l1",
    target_fpr=0.01,
    pen_candidates=np.logspace(1.0, 3.0, 20),
)

print(f"Limiares calibrados: CUSUM h={calib_h:.2f} | PELT pen={calib_pen_pelt:.1f} | Binseg pen={calib_pen_binseg:.1f}")

# Inicialização dos detectores
detectors = {
    "CUSUM (h calibrado)": RobustCUSUMDetector(h=calib_h, k_slack=0.5, init_window=30, guard_window=15),
    "E-Divisive (Means, Daly 2020)": EDivisiveDetector(alpha=0.05, permutations=50, min_size=25, metric="means", seed=42),
    "E-Divisive (Medians, Twitter)": EDivisiveDetector(alpha=0.05, permutations=50, min_size=25, metric="medians", seed=42),
    "PELT (L2, ruptures)": RupturesDetector(algorithm="pelt", cost_model="l2", pen=calib_pen_pelt, min_size=25),
    "Binseg (L1, ruptures)": RupturesDetector(algorithm="binseg", cost_model="l1", pen=calib_pen_binseg, min_size=25),
}

noise_scenarios = [
    (NoiseType.GAUSSIAN, "Gaussiano N(0, σ²)", 42),
    (NoiseType.AR1, "AR(1) Autocorrelacionado (φ=0.7)", 43),
    (NoiseType.STUDENT_T, "t-Student Cauda Pesada (ν=3)", 44),
    (NoiseType.OUTLIERS, "Outliers Esporádicos (p=0.02, 5σ)", 45),
    (NoiseType.BIMODAL, "Bimodal Heterogêneo (w=0.3, Δ=3σ)", 46),
]

ground_truth_cps = [100, 220]  # Dois change points conhecidos
jump_magnitudes = [2.5, -2.0]   # Saltos em múltiplos de sigma

fig, axes = plt.subplots(len(noise_scenarios), 1, figsize=(15, 18), sharex=True)

print("\n--- RESULTADOS DAS DETECÇÕES POR PERFIL DE RUÍDO ---")

for ax, (ntype, title, seed_val) in zip(axes, noise_scenarios):
    series_obj = generate_synthetic_series(
        n=350,
        base_level=100.0,
        noise_scale=4.0,
        noise_type=ntype,
        change_points=ground_truth_cps,
        magnitudes_std=jump_magnitudes,
        ar1_phi=0.7,
        student_t_df=3.0,
        outlier_prob=0.02,
        bimodal_weight=0.3,
        bimodal_mean_shift_std=3.0,
        seed=seed_val,
    )

    y = series_obj.values
    gt = series_obj.ground_truth_signal

    ax.plot(y, color='lightgray', alpha=0.85, lw=1.2, label='Observações (CI)')
    ax.plot(gt, color='black', lw=2.0, ls='--', label=f'Ground Truth: {ground_truth_cps}')

    for true_cp in ground_truth_cps:
        ax.axvline(true_cp, color='black', alpha=0.4, ls=':')

    print(f"\n[Cenário: {title}] (Ground Truth: {ground_truth_cps})")

    # Cores e marcadores por algoritmo
    colors = {
        "CUSUM (h calibrado)": "red",
        "E-Divisive (Means, Daly 2020)": "blue",
        "E-Divisive (Medians, Twitter)": "purple",
        "PELT (L2, ruptures)": "green",
        "Binseg (L1, ruptures)": "darkorange",
    }
    y_offsets = {
        "CUSUM (h calibrado)": 1.06,
        "E-Divisive (Means, Daly 2020)": 1.03,
        "E-Divisive (Medians, Twitter)": 1.00,
        "PELT (L2, ruptures)": 0.97,
        "Binseg (L1, ruptures)": 0.94,
    }

    y_max = np.max(gt)
    y_min = np.min(gt)
    y_range = max(1.0, y_max - y_min)

    for det_name, detector in detectors.items():
        det_cps = detector.detect(y)
        print(f"  * {det_name:30s}: {det_cps}")
        col = colors[det_name]
        y_pos = y_max + (y_offsets[det_name] - 1.0) * y_range

        for cp in det_cps:
            ax.plot(cp, y_pos, marker='v', markersize=7, color=col)

        # Plot dummy para legenda
        ax.plot([], [], marker='v', color=col, ls='', label=f'{det_name} -> {det_cps}')

    ax.set_title(f'Perfil: {title}', fontweight='bold')
    ax.set_ylabel('Métrica (ms)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.0), fontsize=8, frameon=True)

axes[-1].set_xlabel('Índice do Commit')
plt.tight_layout()
out_fig = REPORTS_DIR / 'fig05_phase2_detectors_demo.png'
plt.savefig(out_fig, bbox_inches='tight')
plt.close()

print(f"\nDemonstração concluída! Gráfico comparativo gerado em: {out_fig}")

