"""Exemplo prático de avaliação de uma série sintética com os três detectores da Fase 2.

Gera uma série com 2 change points (commits 100 e 220), avalia as detecções dos detectores
sob k in {1, 3, 5, 10} usando o matching ótimo bipartido e calcula as métricas com atraso.
"""

import json
import numpy as np

from cpd.detectors import (
    RobustCUSUMDetector,
    EDivisiveDetector,
    RupturesDetector,
)
from cpd.evaluation.metrics import (
    evaluate_single_series,
    aggregate_metrics,
)
from cpd.generators.synthetic import NoiseType, generate_synthetic_series

# 1. Gerar série sintética com ruído gaussiano e dois change points
series_obj = generate_synthetic_series(
    n=350,
    base_level=100.0,
    noise_scale=4.0,
    noise_type=NoiseType.GAUSSIAN,
    change_points=[100, 220],
    magnitudes_std=[2.5, -2.0],
    seed=42,
)

gt = series_obj.change_points
y = series_obj.values

print(f"Ground Truth Real: {gt} (n={len(y)} commits)")

# 2. Executar os três detectores principais
cusum = RobustCUSUMDetector(h=10.0, k_slack=0.5, init_window=30, guard_window=15)
ediv = EDivisiveDetector(alpha=0.05, permutations=50, min_size=25, metric="means", seed=42)
pelt = RupturesDetector(algorithm="pelt", cost_model="l2", pen=297.6, min_size=25)

detectors = {
    "CUSUM (h=10.0)": cusum,
    "E-Divisive (Means)": ediv,
    "PELT (L2)": pelt,
}

raw_detections = {}
for name, det in detectors.items():
    res = det.detect_detailed(y)
    raw_detections[name] = res.change_points
    print(f"-> {name:20s}: detectou commits {res.change_points}")

# 3. Avaliação pós-processamento sob múltiplas tolerâncias k in {1, 3, 5, 10}
tolerances = [1, 3, 5, 10]
print("\n" + "="*85)
print(f"{'Detector':20s} | {'k':2s} | {'TP':2s} | {'FP':2s} | {'FN':2s} | {'Precision':9s} | {'Recall':6s} | {'F1':6s} | {'Matches (GT, Det)':18s} | {'Atraso Médio'}")
print("="*85)

for name, det_cps in raw_detections.items():
    for k in tolerances:
        eval_res = evaluate_single_series(ground_truth=gt, detected=det_cps, k=k)
        p_str = f"{eval_res.precision:.3f}" if not np.isnan(eval_res.precision) else "   NaN   "
        r_str = f"{eval_res.recall:.3f}" if not np.isnan(eval_res.recall) else " NaN  "
        f1_str = f"{eval_res.f1:.3f}" if not np.isnan(eval_res.f1) else " NaN  "
        match_str = str(eval_res.matches)
        delay_str = f"{eval_res.mean_delay:+.1f} commits" if not np.isnan(eval_res.mean_delay) else "N/A"
        print(f"{name:20s} | {k:2d} | {eval_res.tp:2d} | {eval_res.fp:2d} | {eval_res.fn:2d} | {p_str:9s} | {r_str:6s} | {f1_str:6s} | {match_str:18s} | {delay_str}")
    print("-" * 85)

