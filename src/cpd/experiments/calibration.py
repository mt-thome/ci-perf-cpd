"""Calibração e validação oficial de hiperparâmetros com controle de FAR (Fase 4).

Executa calibração justa usando exatamente o mesmo conjunto de séries estacionárias
para todos os detectores (CUSUM, PELT_L2, PELT_L1, Binseg_L2, EDiv_Means, EDiv_Medians),
registrando FAR_1k, FAR_series e flag `saturado` em calibração e validação.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cpd.detectors import (
    EDivisiveMeansDetector,
    EDivisiveMediansDetector,
    RobustCUSUMDetector,
    RupturesDetector,
    filter_edivisive_by_alpha,
)
from cpd.detectors.edivisive import warmup_edivisive_jit
from cpd.experiments.runner import compute_config_hash
from cpd.generators.synthetic import NoiseType, generate_synthetic_series


def _init_calib_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    warmup_edivisive_jit()


def _compute_far_metrics(alarms_per_series: list[int], n: int) -> tuple[float, float]:
    """Retorna (far_1k, far_series) a partir da contagem de alarmes por série estacionária."""
    num_series = len(alarms_per_series)
    if num_series == 0 or n <= 0:
        return 0.0, 0.0
    total_alarms = sum(alarms_per_series)
    series_with_alarm = sum(1 for a in alarms_per_series if a > 0)
    far_1k = (total_alarms / (num_series * n)) * 1000.0
    far_series = series_with_alarm / num_series
    return float(far_1k), float(far_series)


def _calibrate_ediv_for_profile(
    calib_histories: list[list[dict[str, Any]]],
    val_histories: list[list[dict[str, Any]]],
    n: int,
    alpha_candidates: list[float],
    far_1k_target: float,
) -> dict[str, Any]:
    """Seleciona o maior alpha em `alpha_candidates` cujo FAR_1k em calibração <= alvo."""
    alphas_desc = sorted(alpha_candidates, reverse=True)
    chosen_alpha = min(alphas_desc)
    saturated = True
    chosen_cal_far_1k = 0.0
    chosen_cal_far_series = 0.0

    for a in alphas_desc:
        alarms = [len(filter_edivisive_by_alpha(h, target_alpha=a)) for h in calib_histories]
        f1k, fser = _compute_far_metrics(alarms, n)
        if f1k <= far_1k_target:
            chosen_alpha = float(a)
            chosen_cal_far_1k = f1k
            chosen_cal_far_series = fser
            saturated = False
            break
        chosen_cal_far_1k = f1k
        chosen_cal_far_series = fser

    val_alarms = [len(filter_edivisive_by_alpha(h, target_alpha=chosen_alpha)) for h in val_histories]
    val_f1k, val_fser = _compute_far_metrics(val_alarms, n)

    return {
        "param": float(chosen_alpha),
        "far_1k_calib": chosen_cal_far_1k,
        "far_series_calib": chosen_cal_far_series,
        "far_1k_val": val_f1k,
        "far_series_val": val_fser,
        "saturado": bool(saturated),
    }


def _calibrate_ruptures_for_profile(
    calib_fitted: list[tuple[Any, int]],
    val_series: list[np.ndarray],
    algorithm: str,
    cost_model: str,
    pen_candidates: list[float],
    far_1k_target: float,
    min_size: int = 25,
) -> dict[str, Any]:
    """Busca monótona decrescente em penalidade para FAR_1k <= alvo."""
    pen_desc = sorted((float(p) for p in pen_candidates), reverse=True)
    chosen_pen = pen_desc[0]
    saturated = True
    chosen_cal_far_1k = 0.0
    chosen_cal_far_series = 0.0

    for pen_val in pen_desc:
        alarms = []
        for model, n in calib_fitted:
            raw_bkps = model.predict(pen=pen_val)
            cps = [cp for cp in raw_bkps if 0 < cp < n]
            alarms.append(len(cps))
        f1k, fser = _compute_far_metrics(alarms, calib_fitted[0][1])
        if f1k <= far_1k_target:
            chosen_pen = pen_val
            chosen_cal_far_1k = f1k
            chosen_cal_far_series = fser
            saturated = False
        else:
            if saturated:
                chosen_cal_far_1k = f1k
                chosen_cal_far_series = fser
            break

    val_det = RupturesDetector(algorithm=algorithm, cost_model=cost_model, pen=chosen_pen, min_size=min_size)
    val_alarms = [len(val_det.detect(s)) for s in val_series]
    val_f1k, val_fser = _compute_far_metrics(val_alarms, len(val_series[0]))

    return {
        "param": float(chosen_pen),
        "far_1k_calib": chosen_cal_far_1k,
        "far_series_calib": chosen_cal_far_series,
        "far_1k_val": val_f1k,
        "far_series_val": val_fser,
        "saturado": bool(saturated),
    }


def _calibrate_cusum_for_profile(
    calib_series: list[np.ndarray],
    val_series: list[np.ndarray],
    h_candidates: list[float],
    far_1k_target: float,
) -> dict[str, Any]:
    """Varre h em ordem crescente até FAR_1k <= alvo."""
    n = len(calib_series[0])
    h_asc = sorted(float(h) for h in h_candidates)
    chosen_h = h_asc[-1]
    saturated = True
    chosen_cal_far_1k = 0.0
    chosen_cal_far_series = 0.0

    for h_val in h_asc:
        det = RobustCUSUMDetector(h=h_val)
        alarms = [len(det.detect(s)) for s in calib_series]
        f1k, fser = _compute_far_metrics(alarms, n)
        chosen_cal_far_1k = f1k
        chosen_cal_far_series = fser
        if f1k <= far_1k_target:
            chosen_h = h_val
            saturated = False
            break

    val_det = RobustCUSUMDetector(h=chosen_h)
    val_alarms = [len(val_det.detect(s)) for s in val_series]
    val_f1k, val_fser = _compute_far_metrics(val_alarms, n)

    return {
        "param": float(chosen_h),
        "far_1k_calib": chosen_cal_far_1k,
        "far_series_calib": chosen_cal_far_series,
        "far_1k_val": val_f1k,
        "far_series_val": val_fser,
        "saturado": bool(saturated),
    }


def _calibrate_single_pair(task: dict[str, Any]) -> dict[str, Any]:
    """Calibra e valida todos os 6 detectores para um par (n, perfil_ruido)."""
    import ruptures as rpt

    t_start = time.perf_counter()
    n = int(task["n"])
    ntype_str = task["perfil"]
    n_calib = int(task["n_calib"])
    n_val = int(task["n_val"])
    calib_seed_start = int(task["calib_seed_start"])
    val_seed_start = int(task["val_seed_start"])
    targets_far_1k = [float(x) for x in task["targets_far_1k"]]
    ntype = NoiseType(ntype_str)

    calib_series = [
        generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        for s in range(calib_seed_start, calib_seed_start + n_calib)
    ]
    val_series = [
        generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        for s in range(val_seed_start, val_seed_start + n_val)
    ]

    # Pré-ajuste ruptures (fit uma única vez por série)
    fitted_pelt_l2 = [(rpt.Pelt(model="l2", min_size=25).fit(s), n) for s in calib_series]
    fitted_pelt_l1 = [(rpt.Pelt(model="l1", min_size=25).fit(s), n) for s in calib_series]
    fitted_binseg_l2 = [(rpt.Binseg(model="l2", min_size=25).fit(s), n) for s in calib_series]

    # E-Divisive executado UMA única vez com alpha=0.20 (permissivo) por série
    calib_ediv_means_hist = [
        EDivisiveMeansDetector(alpha=0.20, permutations=100, min_size=25, seed=s).detect_detailed(s_arr).metadata["division_history"]
        for s, s_arr in zip(range(calib_seed_start, calib_seed_start + n_calib), calib_series)
    ]
    val_ediv_means_hist = [
        EDivisiveMeansDetector(alpha=0.20, permutations=100, min_size=25, seed=s).detect_detailed(s_arr).metadata["division_history"]
        for s, s_arr in zip(range(val_seed_start, val_seed_start + n_val), val_series)
    ]
    calib_ediv_medians_hist = [
        EDivisiveMediansDetector(alpha=0.20, permutations=100, min_size=25, seed=s, exact_medians=True).detect_detailed(s_arr).metadata["division_history"]
        for s, s_arr in zip(range(calib_seed_start, calib_seed_start + n_calib), calib_series)
    ]
    val_ediv_medians_hist = [
        EDivisiveMediansDetector(alpha=0.20, permutations=100, min_size=25, seed=s, exact_medians=True).detect_detailed(s_arr).metadata["division_history"]
        for s, s_arr in zip(range(val_seed_start, val_seed_start + n_val), val_series)
    ]

    h_cand = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]
    pen_cand_l2 = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
    pen_cand_l1 = [30.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    alpha_cand = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]

    per_target: dict[str, dict[str, Any]] = {}
    for tgt in targets_far_1k:
        tgt_key = f"{tgt:.1f}"
        per_target[tgt_key] = {
            "CUSUM": _calibrate_cusum_for_profile(calib_series, val_series, h_cand, tgt),
            "PELT_L2": _calibrate_ruptures_for_profile(fitted_pelt_l2, val_series, "pelt", "l2", pen_cand_l2, tgt),
            "PELT_L1": _calibrate_ruptures_for_profile(fitted_pelt_l1, val_series, "pelt", "l1", pen_cand_l1, tgt),
            "BINSEG_L2": _calibrate_ruptures_for_profile(fitted_binseg_l2, val_series, "binseg", "l2", pen_cand_l2, tgt),
            "EDiv_Means": _calibrate_ediv_for_profile(calib_ediv_means_hist, val_ediv_means_hist, n, alpha_cand, tgt),
            "EDiv_Medians": _calibrate_ediv_for_profile(calib_ediv_medians_hist, val_ediv_medians_hist, n, alpha_cand, tgt),
        }

    elapsed_s = time.perf_counter() - t_start
    return {
        "n": n,
        "perfil": ntype_str,
        "per_target": per_target,
        "elapsed_s": elapsed_s,
    }


def run_calibration_suite(
    config: dict[str, Any],
    config_path: Path,
    out_manifest_path: Path,
    git_hash: str,
    sizes: list[int],
    n_calib: int,
    n_val: int,
    max_workers: int,
    primary_far_1k: float = 0.2,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Executa calibração e validação em paralelo (maiores `n` primeiro) e salva manifesto."""
    calib_cfg = config["calibracao"]
    calib_seed_start = int(calib_cfg["faixa_sementes_calibracao"][0])
    val_seed_start = int(calib_cfg["faixa_sementes_validacao"][0])
    targets_far_1k = sorted(set([float(x) for x in calib_cfg["alvos_far_1k"]] + [primary_far_1k]))
    noises = [p["tipo"] for p in config["grid"]["perfis_ruido"]]
    is_provisional = (n_calib < 200) or (n_val < 100) or (set(sizes) != set(config["grid"]["tamanhos_n"]))

    # Ordena do maior n para o menor para evitar tail stragglers
    tasks = [
        {
            "n": n,
            "perfil": p,
            "n_calib": n_calib,
            "n_val": n_val,
            "calib_seed_start": calib_seed_start,
            "val_seed_start": val_seed_start,
            "targets_far_1k": targets_far_1k,
        }
        for n in sorted(sizes, reverse=True)
        for p in noises
    ]

    manifest: dict[str, Any] = {
        "_meta": {
            "provisorio": is_provisional,
            "alvo_principal_far_1k": primary_far_1k,
            "alvos_far_1k": targets_far_1k,
            "num_series_calibracao": n_calib,
            "num_series_validacao": n_val,
            "faixa_sementes_calibracao": [calib_seed_start, calib_seed_start + n_calib],
            "faixa_sementes_validacao": [val_seed_start, val_seed_start + n_val],
            "tamanhos_n": sorted(sizes),
            "config_sha256": compute_config_hash(config_path),
            "git_hash": git_hash,
        },
        "detalhes_por_alvo": {},
    }

    primary_key = f"{primary_far_1k:.1f}"
    total_tasks = len(tasks)
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_calib_worker) as executor:
        futures = {executor.submit(_calibrate_single_pair, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            n = res["n"]
            perfil = res["perfil"]
            manifest["detalhes_por_alvo"].setdefault(perfil, {})[str(n)] = res["per_target"]

            # Popula chaves planas do alvo principal para consumo direto pelo runner
            prim = res["per_target"][primary_key]
            for det_key in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2"):
                manifest.setdefault(f"{det_key}_perfil", {}).setdefault(perfil, {})[str(n)] = prim[det_key]["param"]
                if perfil == "gaussian":
                    manifest.setdefault(f"{det_key}_gauss", {})[str(n)] = prim[det_key]["param"]
            manifest.setdefault("EDiv_Means", {}).setdefault(perfil, {})[str(n)] = prim["EDiv_Means"]["param"]
            manifest.setdefault("EDiv_Medians", {}).setdefault(perfil, {})[str(n)] = prim["EDiv_Medians"]["param"]

            if log_fn:
                sat_flags = [
                    f"{k}(sat={prim[k]['saturado']},val_far1k={prim[k]['far_1k_val']:.2f})"
                    for k in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2", "EDiv_Means", "EDiv_Medians")
                ]
                log_fn(
                    f"[Calibração {done}/{total_tasks}] n={n} perfil={perfil} "
                    f"tempo={res['elapsed_s']:.1f}s | {' '.join(sat_flags)}"
                )

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_manifest_path.with_suffix(f".tmp_{os.getpid()}.json")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.rename(out_manifest_path)
    return manifest
