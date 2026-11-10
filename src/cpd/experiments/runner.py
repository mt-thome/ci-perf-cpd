"""Runner modular do grid fatorial de Change Point Detection (Fase 4).

Implementa calibração com sementes disjuntas, determinismo estável via SHA-256,
gravação atômica em Parquet com manifesto de configuração, medição de tempo de CPU e
suporte a estudo piloto (R=3) vs avaliação completa (R=50).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from cpd.detectors import (
    BaseDetector,
    EDivisiveMeansDetector,
    EDivisiveMediansDetector,
    RobustCUSUMDetector,
    RupturesDetector,
)
from cpd.generators.synthetic import (
    NoiseType,
    SyntheticSeries,
    SyntheticSeriesConfig,
    generate_synthetic_series,
)


def get_git_commit_hash(repo_dir: Path) -> str:
    """Retorna o hash SHA do commit HEAD atual e avisa se houver mudanças não commitadas."""
    try:
        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        if status_res.stdout.strip():
            print("AVISO CRÍTICO: Árvore de trabalho do Git contém modificações não commitadas!")

        hash_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return hash_res.stdout.strip()
    except Exception as exc:
        print(f"Não foi possível obter hash git: {exc}")
        return "untracked_dev"


def derive_stable_seed(cenario_id: str, semente: int) -> int:
    """Deriva uma semente estável de 32-bits via SHA-256 independente de processo ou SO."""
    raw = f"{cenario_id}_{semente}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:8], 16)


def compute_effective_snr(magnitude_val: float, noise_series: np.ndarray) -> tuple[float, float]:
    """Calcula a dispersão robusta sigma_mad e a SNR efetiva magnitude / sigma_mad."""
    med = float(np.median(noise_series))
    mad = float(np.median(np.abs(noise_series - med)))
    sigma_mad = 1.4826 * mad
    if sigma_mad <= 1e-12:
        sigma_mad = float(np.std(noise_series)) if np.std(noise_series) > 1e-12 else 1e-6
    effective_snr = float(magnitude_val / sigma_mad)
    return sigma_mad, effective_snr


def compute_hazard_levels(
    values: np.ndarray,
    ground_truth_cps: list[int],
    detected_cps: list[int],
    margin: int = 15,
) -> list[dict[str, Any]]:
    """Calcula o Hazard Level = log(media_antes / media_depois) para detecções casadas."""
    n = len(values)
    hazard_records = []
    for d in detected_cps:
        # Segmentos locais adjacentes
        left = max(0, d - margin)
        right = min(n, d + margin)
        slice_before = values[left:d]
        slice_after = values[d:right]
        if len(slice_before) > 0 and len(slice_after) > 0:
            mean_b = float(np.mean(slice_before))
            mean_a = float(np.mean(slice_after))
            # Evita divisão por zero ou log de número não positivo
            if mean_b > 0 and mean_a > 0:
                h_est = float(np.log(mean_b / mean_a))
            else:
                h_est = float("nan")
        else:
            h_est = float("nan")
        hazard_records.append({"detected_cp": d, "hazard_level_est": h_est})
    return hazard_records


def generate_scenario_series(
    scenario_cfg: dict[str, Any],
    seed_idx: int,
) -> SyntheticSeries:
    """Gera uma série sintética do grid com posições de change points controladas e seguras."""
    cenario_id = scenario_cfg["cenario_id"]
    n = scenario_cfg["n"]
    noise_type = NoiseType(scenario_cfg["perfil_ruido"])
    noise_params = scenario_cfg["noise_params"]
    k_true = scenario_cfg["k_true"]
    mag_sigma = scenario_cfg["magnitude_sigma"]

    stable_seed = derive_stable_seed(cenario_id, seed_idx)
    rng = np.random.default_rng(stable_seed)

    # Determinar change points respeitando margem mínima (>=40) e distância (>=30)
    if k_true == 0:
        cps = []
        mags_signed = []
    elif k_true == 1:
        # Posição centralizada com jitter seguro
        # Intervalo admissível: [40, n - 25]
        min_pos = 40
        max_pos = n - 25
        cp_pos = int(rng.integers(min_pos, max_pos + 1))
        cps = [cp_pos]
        # Sinal aleatório por semente
        sign = 1.0 if rng.uniform() > 0.5 else -1.0
        mags_signed = [sign * mag_sigma]
    elif k_true == 2:
        # n >= 500
        # Segmenta em duas regiões seguras
        min_pos = 40
        max_pos = n - 25
        available = max_pos - min_pos - 30
        shift1 = rng.integers(0, available // 2)
        shift2 = rng.integers(available // 2, available + 1)
        cp1 = min_pos + shift1
        cp2 = cp1 + 30 + (shift2 - shift1)
        cps = [int(cp1), int(min(max_pos, cp2))]
        # Sinais aleatórios independentes
        sign1 = 1.0 if rng.uniform() > 0.5 else -1.0
        sign2 = 1.0 if rng.uniform() > 0.5 else -1.0
        mags_signed = [sign1 * mag_sigma, sign2 * mag_sigma]
    else:
        raise ValueError(f"k_true não suportado: {k_true}")

    series_obj = generate_synthetic_series(
        n=n,
        base_level=100.0,
        noise_scale=4.0,
        noise_type=noise_type,
        change_points=cps,
        magnitudes_std=mags_signed if mags_signed else 1.0,
        ar1_phi=noise_params.get("ar1_phi", 0.6),
        student_t_df=noise_params.get("student_t_df", 3.0),
        outlier_prob=noise_params.get("outlier_prob", 0.02),
        outlier_scale_multiplier=noise_params.get("outlier_scale_multiplier", 5.0),
        bimodal_weight=noise_params.get("bimodal_weight", 0.3),
        bimodal_mean_shift_std=noise_params.get("bimodal_mean_shift_std", 3.0),
        seed=stable_seed,
    )

    return series_obj


def instantiate_detector(
    detector_cfg: dict[str, Any],
    calibration_dict: dict[str, Any],
    perfil_ruido: str,
    n: int,
    stable_seed: int,
) -> BaseDetector:
    """Instancia um detector configurado com seus parâmetros calibrados ou padrões."""
    cls_name = detector_cfg["classe"]
    regime = detector_cfg["regime_calibracao"]
    base_params = dict(detector_cfg.get("params", {}))

    # Resolução de limiares calibrados
    if cls_name == "RobustCUSUMDetector":
        if regime == "calibrado_gaussiano":
            h_val = calibration_dict.get("CUSUM_gauss", {}).get(str(n), 10.0)
        elif regime == "calibrado_perfil":
            h_val = calibration_dict.get("CUSUM_perfil", {}).get(perfil_ruido, {}).get(str(n), 10.0)
        else:
            h_val = base_params.get("h", 10.0)
        return RobustCUSUMDetector(
            h=h_val,
            k_slack=base_params.get("k_slack", 0.5),
            init_window=base_params.get("init_window", 30),
            guard_window=base_params.get("guard_window", 15),
            update_mean_after_alarm=True,
        )

    elif cls_name == "RupturesDetector":
        alg = detector_cfg["algoritmo"]
        cost = detector_cfg["cost_model"]
        key = f"{alg.upper()}_{cost.upper()}_{regime.split('_')[1]}"
        if regime == "calibrado_gaussiano":
            pen_val = calibration_dict.get(key, {}).get(str(n), 200.0)
        elif regime == "calibrado_perfil":
            pen_val = calibration_dict.get(key, {}).get(perfil_ruido, {}).get(str(n), 200.0)
        else:
            pen_val = base_params.get("pen", 100.0)
        return RupturesDetector(
            algorithm=alg,
            cost_model=cost,
            pen=pen_val,
            min_size=base_params.get("min_size", 25),
        )

    elif cls_name == "EDivisiveMeansDetector":
        alpha = 0.05
        if regime == "calibrado_perfil":
            alpha = calibration_dict.get("EDiv_Means", {}).get(perfil_ruido, {}).get(str(n), 0.05)
        return EDivisiveMeansDetector(
            alpha=alpha,
            permutations=base_params.get("permutations", 100),
            min_size=base_params.get("min_size", 25),
            seed=stable_seed,
        )

    elif cls_name == "EDivisiveMediansDetector":
        alpha = 0.05
        if regime == "calibrado_perfil":
            alpha = calibration_dict.get("EDiv_Medians", {}).get(perfil_ruido, {}).get(str(n), 0.05)
        return EDivisiveMediansDetector(
            alpha=alpha,
            permutations=base_params.get("permutations", 100),
            min_size=base_params.get("min_size", 25),
            seed=stable_seed,
        )
    else:
        raise ValueError(f"Classe de detector desconhecida: {cls_name}")


def execute_single_cell(
    scenario_cfg: dict[str, Any],
    detector_cfg: dict[str, Any],
    seed_idx: int,
    calibration_dict: dict[str, Any],
    git_hash: str,
) -> dict[str, Any]:
    """Executa a detecção para uma série e retorna a linha no esquema padronizado."""
    cenario_id = scenario_cfg["cenario_id"]
    n = scenario_cfg["n"]
    perfil = scenario_cfg["perfil_ruido"]
    detector_name = detector_cfg["nome"]

    series_obj = generate_scenario_series(scenario_cfg, seed_idx)
    stable_seed = derive_stable_seed(cenario_id, seed_idx)

    detector = instantiate_detector(
        detector_cfg=detector_cfg,
        calibration_dict=calibration_dict,
        perfil_ruido=perfil,
        n=n,
        stable_seed=stable_seed,
    )

    t0 = time.perf_counter()
    res = detector.detect_detailed(series_obj.values)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Metadados de ruído e SNR efetiva
    sigma_mad, effective_snr = compute_effective_snr(
        magnitude_val=scenario_cfg["magnitude_sigma"] * series_obj.config.noise_scale,
        noise_series=series_obj.noise,
    )

    params_ruido = {
        **scenario_cfg["noise_params"],
        "sigma_nominal": series_obj.config.noise_scale,
        "sigma_mad_empirico": sigma_mad,
        "snr_efetiva_mad": effective_snr,
    }

    hazard_records = compute_hazard_levels(
        values=series_obj.values,
        ground_truth_cps=series_obj.change_points,
        detected_cps=res.change_points,
    )

    scores_list = res.scores.tolist() if res.scores is not None else []
    if not scores_list and "p_values" in res.metadata:
        scores_list = [float(p) for p in res.metadata["p_values"]]

    return {
        "cenario_id": cenario_id,
        "split": scenario_cfg.get("split", "avaliacao"),
        "perfil_ruido": perfil,
        "params_ruido": json.dumps(params_ruido),
        "n": n,
        "magnitude": float(scenario_cfg["magnitude_sigma"]),
        "semente": int(seed_idx),
        "detector": detector_name,
        "parametros": json.dumps(res.metadata),
        "versao_codigo": git_hash,
        "ground_truth": series_obj.change_points,
        "indices_detectados": res.change_points,
        "scores_detectados": scores_list,
        "hazard_levels_est": json.dumps(hazard_records),
        "tempo_cpu_ms": float(elapsed_ms),
    }


def save_partition_atomic(df: pd.DataFrame, out_path: Path) -> None:
    """Grava o DataFrame em Parquet de forma estritamente atômica (tmp file + rename)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}.parquet")
    df.to_parquet(tmp_path, index=False)
    tmp_path.rename(out_path)
