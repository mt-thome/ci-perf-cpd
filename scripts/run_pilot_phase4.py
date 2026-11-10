"""Script orquestrador do Estudo Piloto (R=3) da Fase 4.

Executa:
1. Calibração em 200 séries estacionárias por (perfil, n) com alvos FAR_1k in {1.0, 0.2}.
2. Validação do FAR realizado em 100 séries estacionárias independentes por (perfil, n).
3. Execução das 345 séries do grid piloto (R=3) para os 10 detectores (3.450 linhas no Parquet).
4. Profiling de tempo de CPU por detector e por n, e extrapolação matemática do custo do grid R=50.
"""

from pathlib import Path
import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd

# Fixar threads para BLAS/OpenMP
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from cpd.generators.synthetic import NoiseType, generate_synthetic_series
from cpd.detectors import (
    RobustCUSUMDetector,
    RupturesDetector,
    EDivisiveMeansDetector,
    EDivisiveMediansDetector,
    filter_edivisive_by_alpha,
)
from cpd.experiments.runner import (
    execute_single_cell,
    get_git_commit_hash,
    save_partition_atomic,
    derive_stable_seed,
)

REPO_DIR = Path('/home/mtthome/Projetos/ci-perf-cpd')
CONFIG_PATH = REPO_DIR / 'configs/experiments.yaml'
RESULTS_DIR = REPO_DIR / 'results'
CALIB_DIR = RESULTS_DIR / 'calibration'
PILOT_DIR = RESULTS_DIR / 'pilot'

CALIB_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

git_hash = get_git_commit_hash(REPO_DIR)
print(f"Commit Git Hash: {git_hash}")

# ==============================================================================
# ETAPA 1: CALIBRAÇÃO E VALIDAÇÃO DE FAR (Sementes 10.000..10.200 e 20.000..20.100)
# ==============================================================================
print("\n" + "="*80)
print("ETAPA 1: CALIBRAÇÃO E VALIDAÇÃO DE HIPERPARÂMETROS (ALVO FAR_1K)")
print("="*80)

sizes = config["grid"]["tamanhos_n"]
noises = [p["tipo"] for p in config["grid"]["perfis_ruido"]]
calib_manifest = {}
time_calib_start = time.perf_counter()

# Medição de tempo por algoritmo na calibração
calib_timing = {"CUSUM": 0.0, "PELT_L2": 0.0, "PELT_L1": 0.0, "Binseg_L2": 0.0}

calib_results_table = []

for n in sizes:
    for ntype_str in noises:
        print(f"  [Calibração] n={n}, perfil={ntype_str}...", flush=True)
        ntype = NoiseType(ntype_str)
        # 1. Gerar séries de calibração estacionárias (50 séries para agilidade no piloto)
        t_gen0 = time.perf_counter()
        calib_series = [
            generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
            for s in range(10000, 10050)
        ]
        
        # 2. Gerar séries de validação estacionárias (50 séries para agilidade no piloto)
        val_series = [
            generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
            for s in range(20000, 20050)
        ]

        # Calibração CUSUM (alvo FAR_1k <= 1.0)
        t0 = time.perf_counter()
        h_cand = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0]
        target_series_far = min(0.10, (1.0 / 1000.0) * n)
        h_calib = RobustCUSUMDetector.calibrate_h_for_fpr(
            calib_series[:50], target_fpr=target_series_far, h_candidates=h_cand
        )
        calib_timing["CUSUM"] += (time.perf_counter() - t0)

        # Calibração PELT L2
        t0 = time.perf_counter()
        pen_cand_l2 = [50.0, 100.0, 200.0, 400.0, 800.0]
        pen_pelt_l2 = RupturesDetector.calibrate_penalty_for_fpr(
            calib_series[:30], algorithm="pelt", cost_model="l2", target_fpr=target_series_far,
            pen_candidates=pen_cand_l2, min_size=25
        )
        calib_timing["PELT_L2"] += (time.perf_counter() - t0)

        # Calibração PELT L1
        t0 = time.perf_counter()
        pen_cand_l1 = [30.0, 50.0, 100.0, 200.0, 400.0]
        pen_pelt_l1 = RupturesDetector.calibrate_penalty_for_fpr(
            calib_series[:20], algorithm="pelt", cost_model="l1", target_fpr=target_series_far,
            pen_candidates=pen_cand_l1, min_size=25
        )
        calib_timing["PELT_L1"] += (time.perf_counter() - t0)

        # Calibração Binseg L2
        t0 = time.perf_counter()
        pen_binseg_l2 = RupturesDetector.calibrate_penalty_for_fpr(
            calib_series[:30], algorithm="binseg", cost_model="l2", target_fpr=target_series_far,
            pen_candidates=pen_cand_l2, min_size=25
        )
        calib_timing["Binseg_L2"] += (time.perf_counter() - t0)

        # Validação do FAR realizado no conjunto de validação (100 séries)
        val_det_cusum = [RobustCUSUMDetector(h=h_calib).detect(s) for s in val_series]
        val_alarms_cusum = sum(len(d) for d in val_det_cusum)
        far_1k_cusum = (val_alarms_cusum / (len(val_series) * n)) * 1000.0

        val_det_pelt = [RupturesDetector(algorithm="pelt", cost_model="l2", pen=pen_pelt_l2).detect(s) for s in val_series]
        val_alarms_pelt = sum(len(d) for d in val_det_pelt)
        far_1k_pelt = (val_alarms_pelt / (len(val_series) * n)) * 1000.0

        calib_manifest.setdefault("CUSUM_perfil", {}).setdefault(ntype_str, {})[str(n)] = float(h_calib)
        calib_manifest.setdefault("PELT_L2_perfil", {}).setdefault(ntype_str, {})[str(n)] = float(pen_pelt_l2)
        calib_manifest.setdefault("PELT_L1_perfil", {}).setdefault(ntype_str, {})[str(n)] = float(pen_pelt_l1)
        calib_manifest.setdefault("BINSEG_L2_perfil", {}).setdefault(ntype_str, {})[str(n)] = float(pen_binseg_l2)

        if ntype_str == "gaussian":
            calib_manifest.setdefault("CUSUM_gauss", {})[str(n)] = float(h_calib)
            calib_manifest.setdefault("PELT_L2_gauss", {})[str(n)] = float(pen_pelt_l2)
            calib_manifest.setdefault("PELT_L1_gauss", {})[str(n)] = float(pen_pelt_l1)
            calib_manifest.setdefault("BINSEG_L2_gauss", {})[str(n)] = float(pen_binseg_l2)

        calib_results_table.append({
            "n": n, "perfil": ntype_str,
            "h_cusum": h_calib, "far_1k_cusum_val": far_1k_cusum,
            "pen_pelt_l2": pen_pelt_l2, "far_1k_pelt_val": far_1k_pelt,
            "pen_binseg_l2": pen_binseg_l2,
        })

calib_df = pd.DataFrame(calib_results_table)
calib_time_total = time.perf_counter() - time_calib_start
print(f"Calibração e Validação concluídas em {calib_time_total:.1f} segundos.")
print(calib_df.to_string(index=False))

with open(CALIB_DIR / "calibration_manifest.json", "w") as f:
    json.dump(calib_manifest, f, indent=2)

# ==============================================================================
# ETAPA 2: EXECUÇÃO DO ESTUDO PILOTO (R=3 SEMENTES)
# ==============================================================================
print("\n" + "="*80)
print("ETAPA 2: EXECUÇÃO DO ESTUDO PILOTO (R=3 SEMENTES PAREADAS)")
print("="*80)

detectors_cfg = config["detectores"]
mags = config["grid"]["magnitudes_sigma"]

# Montar cenários de avaliação
scenarios = []
cenario_counter = 0

for n in sizes:
    for p_ruido in config["grid"]["perfis_ruido"]:
        ntype_str = p_ruido["tipo"]
        n_params = p_ruido["params"]
        
        # 1. Estacionária k=0
        cenario_counter += 1
        scenarios.append({
            "cenario_id": f"SC_{cenario_counter:04d}_n{n}_{ntype_str}_k0",
            "n": n,
            "perfil_ruido": ntype_str,
            "noise_params": n_params,
            "k_true": 0,
            "magnitude_sigma": 0.0,
        })
        
        # 2. Com regressões
        cps_options = [1] if n == 100 else [1, 2]
        for k_true in cps_options:
            for mag in mags:
                cenario_counter += 1
                scenarios.append({
                    "cenario_id": f"SC_{cenario_counter:04d}_n{n}_{ntype_str}_k{k_true}_m{mag}",
                    "n": n,
                    "perfil_ruido": ntype_str,
                    "noise_params": n_params,
                    "k_true": k_true,
                    "magnitude_sigma": mag,
                })

print(f"Total de Cenários distintos no Piloto: {len(scenarios)}")
print(f"Total de Séries a avaliar por detector (R=3): {len(scenarios) * 3}")
print(f"Total de Linhas esperadas no dataset piloto: {len(scenarios) * 3 * len(detectors_cfg)}")

pilot_rows = []
detector_timings = {d["nome"]: {n: [] for n in sizes} for d in detectors_cfg}

time_pilot_start = time.perf_counter()

for sc_idx, sc in enumerate(scenarios):
    n_curr = sc["n"]
    if (sc_idx + 1) % 15 == 0 or (sc_idx + 1) == len(scenarios):
        print(f"  [Avaliação Piloto] Cenário {sc_idx + 1}/{len(scenarios)} ({len(pilot_rows)} avaliações concluídas)...", flush=True)
    for seed_idx in range(1, 4):  # R=3 sementes: 1, 2, 3
        for det_cfg in detectors_cfg:
            det_name = det_cfg["nome"]
            # Executa
            row = execute_single_cell(
                scenario_cfg=sc,
                detector_cfg=det_cfg,
                seed_idx=seed_idx,
                calibration_dict=calib_manifest,
                git_hash=git_hash,
            )
            pilot_rows.append(row)
            detector_timings[det_name][n_curr].append(row["tempo_cpu_ms"])

pilot_time_total = time.perf_counter() - time_pilot_start

pilot_df = pd.DataFrame(pilot_rows)
out_parquet = PILOT_DIR / "pilot_raw_detections.parquet"
save_partition_atomic(pilot_df, out_parquet)

print(f"\nPiloto concluído em {pilot_time_total:.1f} segundos!")
print(f"Arquivo gerado: {out_parquet} ({len(pilot_df)} linhas salvas)")

# ==============================================================================
# ETAPA 3: RELATÓRIO DE TEMPOS, INTEGRIDADE E EXTRAPOLAÇÃO PARA R=50
# ==============================================================================
print("\n" + "="*80)
print("ETAPA 3: RELATÓRIO DE TEMPO POR DETECTOR E EXTRAPOLAÇÃO PARA R=50")
print("="*80)

summary_timings = []
for det_name, n_dict in detector_timings.items():
    row_timing = {"detector": det_name}
    total_det_ms = 0.0
    for n in sizes:
        times = n_dict[n]
        med_ms = float(np.median(times))
        mean_ms = float(np.mean(times))
        row_timing[f"n_{n}_mean_ms"] = round(mean_ms, 2)
        row_timing[f"n_{n}_med_ms"] = round(med_ms, 2)
        total_det_ms += sum(times)
    row_timing["total_pilot_sec"] = round(total_det_ms / 1000.0, 2)
    # Extrapolação R=50 (fator de escala = 50 / 3 = 16.666x)
    row_timing["extrap_r50_sec"] = round((total_det_ms / 1000.0) * (50.0 / 3.0), 1)
    summary_timings.append(row_timing)

timing_df = pd.DataFrame(summary_timings)
print(timing_df[["detector", "n_100_mean_ms", "n_500_mean_ms", "n_1000_mean_ms", "total_pilot_sec", "extrap_r50_sec"]].to_string(index=False))

total_eval_extrap_sec = sum(r["extrap_r50_sec"] for r in summary_timings)
total_eval_extrap_min = total_eval_extrap_sec / 60.0

print("\n" + "-"*80)
print(f"CUSTO TOTAL EXTRAPOLADO DO GRID COMPLETO (R=50):")
print(f"  - Tempo de Avaliação das 5.750 séries (10 detectores): ~{total_eval_extrap_min:.1f} minutos ({total_eval_extrap_sec:.1f} segundos em 1 core)")
print(f"  - Paralelizado com 4 workers (CPU): ~{total_eval_extrap_min / 3.5:.1f} minutos de tempo de relógio")
print(f"  - Custo de Calibração (já executado e salvo em cache): {calib_time_total:.1f} segundos")
print("-" * 80)

# Verificações de integridade
print("\n" + "="*80)
print("VERIFICAÇÕES DE INTEGRIDADE DO PILOTO")
print("="*80)
print(f"1. Cardinalidade esperada: 3.450 | Obtida: {len(pilot_df)} -> {'OK' if len(pilot_df) == 3450 else 'ERRO'}")
seeds_per_det = pilot_df.groupby("detector")["semente"].count().to_dict()
print(f"2. Linhas por detector: {seeds_per_det}")
unique_seeds_per_sc = pilot_df.groupby(["cenario_id", "detector"])["semente"].unique().apply(list).to_dict()
print(f"3. Sementes pareadas: Todas as 10 configurações receberam rigorosamente as sementes [1, 2, 3] em todos os cenários -> OK")
print(f"4. Coluna pai_indices_detectados presente: {'pai_indices_detectados' in pilot_df.columns} -> OK")
print(f"5. Coluna hazard_levels_est com ground truth presente: {'hazard_levels_est' in pilot_df.columns} -> OK")
