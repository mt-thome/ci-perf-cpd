"""Calibração e validação oficial de hiperparâmetros com controle de FAR (Fase 4).

Implementa:
- Tamanhos por n: n=100 (2000 calib / 1000 val), n in {500,1000} (200 calib / 100 val)
- Mesmo conjunto de séries para todos os detectores
- Saturação bidirecional: saturado_inferior e saturado_superior
- IC de Poisson de 95% e critério de validação (ci_lower <= alvo, sem recalibrar)
- Paralelismo por sub-lotes de séries com agregação determinística ordenada por semente
- jump=1 explícito em todas as chamadas ao ruptures
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import chi2

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


def poisson_rate_ci_95(k_alarms: int, exposure_commits: int) -> tuple[float, float]:
    """Calcula o IC exato de Poisson de 95% para a taxa por 1.000 commits (FAR_1k)."""
    if exposure_commits <= 0:
        return 0.0, 0.0
    lower_cnt = 0.0 if k_alarms == 0 else 0.5 * float(chi2.ppf(0.025, 2 * k_alarms))
    upper_cnt = 0.5 * float(chi2.ppf(0.975, 2 * (k_alarms + 1)))
    scale = 1000.0 / float(exposure_commits)
    return lower_cnt * scale, upper_cnt * scale


def _compute_far_summary(alarms_per_series: list[int], n: int, target_far_1k: float) -> dict[str, Any]:
    num_series = len(alarms_per_series)
    exposure = num_series * n
    total_alarms = int(sum(alarms_per_series))
    series_with_alarm = int(sum(1 for a in alarms_per_series if a > 0))
    far_1k = (total_alarms / exposure) * 1000.0 if exposure > 0 else 0.0
    far_series = series_with_alarm / num_series if num_series > 0 else 0.0
    ci_low, ci_high = poisson_rate_ci_95(total_alarms, exposure)
    expected_alarms = (target_far_1k / 1000.0) * exposure
    return {
        "total_alarms": total_alarms,
        "expected_alarms": float(expected_alarms),
        "far_1k": float(far_1k),
        "far_series": float(far_series),
        "ci95_far_1k": [float(ci_low), float(ci_high)],
    }


def _chunk_list(items: list[int], num_chunks: int) -> list[list[int]]:
    k = max(1, min(num_chunks, len(items)))
    return [items[i::k] for i in range(k)]


def _worker_ediv_cusum_binseg(payload: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Processa um sub-lote de sementes para EDiv_Means, EDiv_Medians, CUSUM e Binseg_L2."""
    import ruptures as rpt

    n = int(payload["n"])
    ntype = NoiseType(payload["perfil"])
    seeds: list[int] = payload["seeds"]
    h_cand: list[float] = payload["h_cand"]
    pen_cand_l2: list[float] = payload["pen_cand_l2"]
    run_binseg: bool = payload["run_binseg"]

    out: list[tuple[int, dict[str, Any]]] = []
    for s in seeds:
        arr = generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        ediv_means_h = EDivisiveMeansDetector(alpha=0.20, permutations=100, min_size=25, seed=s).detect_detailed(arr).metadata["division_history"]
        ediv_med_h = EDivisiveMediansDetector(alpha=0.20, permutations=100, min_size=25, seed=s, exact_medians=True).detect_detailed(arr).metadata["division_history"]
        cusum_alarms = {float(h): len(RobustCUSUMDetector(h=float(h)).detect(arr)) for h in h_cand}
        binseg_alarms: dict[float, int] = {}
        if run_binseg:
            bm = rpt.Binseg(model="l2", min_size=25, jump=1).fit(arr)
            for pen in pen_cand_l2:
                bkps = bm.predict(pen=float(pen))
                binseg_alarms[float(pen)] = len([cp for cp in bkps if 0 < cp < n])

        out.append((s, {
            "ediv_means_hist": ediv_means_h,
            "ediv_medians_hist": ediv_med_h,
            "cusum_alarms": cusum_alarms,
            "binseg_alarms": binseg_alarms,
        }))
    return out


def _worker_pelt_chunk(payload: dict[str, Any]) -> list[tuple[int, int]]:
    """Processa um sub-lote de sementes para PELT com jump=1 e uma penalidade fixa."""
    import ruptures as rpt

    n = int(payload["n"])
    ntype = NoiseType(payload["perfil"])
    cost_model: str = payload["cost_model"]
    pen: float = float(payload["pen"])
    seeds: list[int] = payload["seeds"]

    out: list[tuple[int, int]] = []
    for s in seeds:
        arr = generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        bkps = rpt.Pelt(model=cost_model, min_size=25, jump=1).fit(arr).predict(pen=pen)
        out.append((s, len([cp for cp in bkps if 0 < cp < n])))
    return out


def _worker_val_ruptures_chunk(payload: dict[str, Any]) -> list[tuple[int, dict[str, int]]]:
    """Valida CUSUM, PELT_L2, PELT_L1 e BINSEG_L2 (jump=1) em sub-lote de validação."""
    n = int(payload["n"])
    ntype = NoiseType(payload["perfil"])
    seeds: list[int] = payload["seeds"]
    params_by_target: dict[str, dict[str, float]] = payload["params_by_target"]

    # Cache local de detectores instanciados para evitar reconstrução
    out: list[tuple[int, dict[str, int]]] = []
    for s in seeds:
        arr = generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        res_s: dict[str, int] = {}
        # Coleta parâmetros únicos por (detector, valor) para evitar rodar o mesmo pen/h duas vezes
        det_cache: dict[tuple[str, float], int] = {}
        for tgt_key, pmap in params_by_target.items():
            for det_key in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2"):
                pval = float(pmap[det_key])
                ck = (det_key, pval)
                if ck not in det_cache:
                    if det_key == "CUSUM":
                        cnt = len(RobustCUSUMDetector(h=pval).detect(arr))
                    elif det_key == "PELT_L2":
                        cnt = len(RupturesDetector("pelt", "l2", pen=pval, min_size=25, jump=1).detect(arr))
                    elif det_key == "PELT_L1":
                        cnt = len(RupturesDetector("pelt", "l1", pen=pval, min_size=25, jump=1).detect(arr))
                    else:
                        cnt = len(RupturesDetector("binseg", "l2", pen=pval, min_size=25, jump=1).detect(arr))
                    det_cache[ck] = cnt
                res_s[f"{tgt_key}:{det_key}"] = det_cache[ck]
        out.append((s, res_s))
    return out


def _run_mapped(
    executor: ProcessPoolExecutor | None,
    fn: Callable[[dict[str, Any]], list[Any]],
    payloads: list[dict[str, Any]],
) -> list[Any]:
    if executor is None or len(payloads) <= 1:
        merged = []
        for p in payloads:
            merged.extend(fn(p))
        return merged
    merged = []
    for res in executor.map(fn, payloads):
        merged.extend(res)
    return merged


def _select_param_and_flags(
    alarms_by_param: dict[float, list[int]],
    params_permissive_to_strict: list[float],
    n: int,
    target_far_1k: float,
) -> tuple[float, dict[str, Any], bool, bool]:
    """Seleciona o parâmetro mais permissivo que atende FAR_1k <= alvo e calcula saturação."""
    strictest = params_permissive_to_strict[-1]
    permissivest = params_permissive_to_strict[0]

    # Saturação superior: mesmo no parâmetro mais permissivo, FAR_1k < alvo
    perm_summary = _compute_far_summary(alarms_by_param[permissivest], n, target_far_1k)
    saturado_superior = perm_summary["far_1k"] < target_far_1k

    chosen_param = strictest
    chosen_summary = _compute_far_summary(alarms_by_param[strictest], n, target_far_1k)
    saturado_inferior = True

    # Varre do mais permissivo para o mais restritivo (ou inverso) buscando o mais permissivo <= alvo
    for p in params_permissive_to_strict:
        summ = _compute_far_summary(alarms_by_param[p], n, target_far_1k)
        if summ["far_1k"] <= target_far_1k:
            chosen_param = p
            chosen_summary = summ
            saturado_inferior = False
            break

    return chosen_param, chosen_summary, saturado_inferior, saturado_superior


def calibrate_pair(
    n: int,
    perfil: str,
    n_calib: int,
    n_val: int,
    calib_seed_start: int,
    val_seed_start: int,
    targets_far_1k: list[float],
    executor: ProcessPoolExecutor | None,
    num_chunks: int,
) -> dict[str, Any]:
    """Calibra e valida todos os 6 detectores em sub-lotes paralelos com agregação determinística."""
    t_start = time.perf_counter()
    calib_seeds = list(range(calib_seed_start, calib_seed_start + n_calib))
    val_seeds = list(range(val_seed_start, val_seed_start + n_val))
    calib_chunks = _chunk_list(calib_seeds, num_chunks)
    val_chunks = _chunk_list(val_seeds, num_chunks)

    h_cand = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]  # permissivo -> restritivo
    pen_cand_l2 = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]  # permissivo -> restritivo
    pen_cand_l1 = [30.0, 50.0, 100.0, 200.0, 400.0, 800.0]    # permissivo -> restritivo
    alpha_cand = [0.20, 0.15, 0.10, 0.05, 0.02, 0.01]         # permissivo -> restritivo

    # 1. EDiv + CUSUM + Binseg na calibração
    calib_base_res = _run_mapped(
        executor,
        _worker_ediv_cusum_binseg,
        [
            {
                "n": n, "perfil": perfil, "seeds": ch,
                "h_cand": h_cand, "pen_cand_l2": pen_cand_l2, "run_binseg": True,
            }
            for ch in calib_chunks
        ],
    )
    calib_base_map = dict(sorted(calib_base_res, key=lambda x: x[0]))

    # 2. EDiv na validação
    val_ediv_res = _run_mapped(
        executor,
        _worker_ediv_cusum_binseg,
        [
            {
                "n": n, "perfil": perfil, "seeds": ch,
                "h_cand": [], "pen_cand_l2": [], "run_binseg": False,
            }
            for ch in val_chunks
        ],
    )
    val_ediv_map = dict(sorted(val_ediv_res, key=lambda x: x[0]))

    # 3. PELT_L2 e PELT_L1 na calibração com busca monótona decrescente (do maior pen para o menor)
    max_target = max(targets_far_1k)
    pelt_l2_alarms: dict[float, list[int]] = {}
    for pen in sorted(pen_cand_l2, reverse=True):
        res_p = _run_mapped(
            executor,
            _worker_pelt_chunk,
            [{"n": n, "perfil": perfil, "cost_model": "l2", "pen": pen, "seeds": ch} for ch in calib_chunks],
        )
        res_sorted = [cnt for _, cnt in sorted(res_p, key=lambda x: x[0])]
        pelt_l2_alarms[pen] = res_sorted
        if (sum(res_sorted) / (n_calib * n)) * 1000.0 > max_target:
            # Penalidades menores terão FAR ainda maior; preenche placeholder para manter chaves
            break
    for pen in pen_cand_l2:
        if pen not in pelt_l2_alarms:
            pelt_l2_alarms[pen] = [n] * n_calib  # estritamente acima do alvo

    pelt_l1_alarms: dict[float, list[int]] = {}
    for pen in sorted(pen_cand_l1, reverse=True):
        res_p = _run_mapped(
            executor,
            _worker_pelt_chunk,
            [{"n": n, "perfil": perfil, "cost_model": "l1", "pen": pen, "seeds": ch} for ch in calib_chunks],
        )
        res_sorted = [cnt for _, cnt in sorted(res_p, key=lambda x: x[0])]
        pelt_l1_alarms[pen] = res_sorted
        if (sum(res_sorted) / (n_calib * n)) * 1000.0 > max_target:
            break
    for pen in pen_cand_l1:
        if pen not in pelt_l1_alarms:
            pelt_l1_alarms[pen] = [n] * n_calib

    # Montar dicionários por parâmetro em ordem determinística de sementes
    cusum_alarms = {h: [calib_base_map[s]["cusum_alarms"][h] for s in calib_seeds] for h in h_cand}
    binseg_alarms = {pen: [calib_base_map[s]["binseg_alarms"][pen] for s in calib_seeds] for pen in pen_cand_l2}
    ediv_means_alarms = {
        a: [len(filter_edivisive_by_alpha(calib_base_map[s]["ediv_means_hist"], a)) for s in calib_seeds]
        for a in alpha_cand
    }
    ediv_medians_alarms = {
        a: [len(filter_edivisive_by_alpha(calib_base_map[s]["ediv_medians_hist"], a)) for s in calib_seeds]
        for a in alpha_cand
    }

    per_target: dict[str, dict[str, Any]] = {}
    params_by_target_for_val: dict[str, dict[str, float]] = {}

    for tgt in targets_far_1k:
        tgt_key = f"{tgt:.1f}"
        c_p, c_sum, c_sinf, c_ssup = _select_param_and_flags(cusum_alarms, h_cand, n, tgt)
        pl2_p, pl2_sum, pl2_sinf, pl2_ssup = _select_param_and_flags(pelt_l2_alarms, pen_cand_l2, n, tgt)
        pl1_p, pl1_sum, pl1_sinf, pl1_ssup = _select_param_and_flags(pelt_l1_alarms, pen_cand_l1, n, tgt)
        bl2_p, bl2_sum, bl2_sinf, bl2_ssup = _select_param_and_flags(binseg_alarms, pen_cand_l2, n, tgt)
        em_p, em_sum, em_sinf, em_ssup = _select_param_and_flags(ediv_means_alarms, alpha_cand, n, tgt)
        emd_p, emd_sum, emd_sinf, emd_ssup = _select_param_and_flags(ediv_medians_alarms, alpha_cand, n, tgt)

        params_by_target_for_val[tgt_key] = {
            "CUSUM": c_p, "PELT_L2": pl2_p, "PELT_L1": pl1_p, "BINSEG_L2": bl2_p
        }
        per_target[tgt_key] = {
            "CUSUM": {"param": c_p, "calib": c_sum, "saturado_inferior": c_sinf, "saturado_superior": c_ssup},
            "PELT_L2": {"param": pl2_p, "calib": pl2_sum, "saturado_inferior": pl2_sinf, "saturado_superior": pl2_ssup},
            "PELT_L1": {"param": pl1_p, "calib": pl1_sum, "saturado_inferior": pl1_sinf, "saturado_superior": pl1_ssup},
            "BINSEG_L2": {"param": bl2_p, "calib": bl2_sum, "saturado_inferior": bl2_sinf, "saturado_superior": bl2_ssup},
            "EDiv_Means": {"param": em_p, "calib": em_sum, "saturado_inferior": em_sinf, "saturado_superior": em_ssup},
            "EDiv_Medians": {"param": emd_p, "calib": emd_sum, "saturado_inferior": emd_sinf, "saturado_superior": emd_ssup},
        }

    # 4. Validação independente (sem recalibrar!)
    val_rup_res = _run_mapped(
        executor,
        _worker_val_ruptures_chunk,
        [{"n": n, "perfil": perfil, "seeds": ch, "params_by_target": params_by_target_for_val} for ch in val_chunks],
    )
    val_rup_map = dict(sorted(val_rup_res, key=lambda x: x[0]))

    for tgt in targets_far_1k:
        tgt_key = f"{tgt:.1f}"
        for det_key in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2"):
            alarms = [val_rup_map[s][f"{tgt_key}:{det_key}"] for s in val_seeds]
            vs = _compute_far_summary(alarms, n, tgt)
            per_target[tgt_key][det_key]["val"] = vs
            per_target[tgt_key][det_key]["validacao_falhou"] = bool(vs["ci95_far_1k"][0] > tgt)

        for det_key, hist_field in (("EDiv_Means", "ediv_means_hist"), ("EDiv_Medians", "ediv_medians_hist")):
            chosen_a = per_target[tgt_key][det_key]["param"]
            alarms = [len(filter_edivisive_by_alpha(val_ediv_map[s][hist_field], chosen_a)) for s in val_seeds]
            vs = _compute_far_summary(alarms, n, tgt)
            per_target[tgt_key][det_key]["val"] = vs
            per_target[tgt_key][det_key]["validacao_falhou"] = bool(vs["ci95_far_1k"][0] > tgt)

    return {
        "n": n,
        "perfil": perfil,
        "n_calib": n_calib,
        "n_val": n_val,
        "per_target": per_target,
        "elapsed_s": time.perf_counter() - t_start,
    }


def calibration_sizes_for_n(n: int, override_sizes: dict[int, tuple[int, int]] | None = None) -> tuple[int, int]:
    """Retorna (n_calib, n_val) conforme ADR-014 (ou override em smoke)."""
    if override_sizes and n in override_sizes:
        return override_sizes[n]
    if n == 100:
        return 2000, 1000
    return 200, 100


def run_calibration_suite(
    config: dict[str, Any],
    config_path: Path,
    out_manifest_path: Path,
    git_hash: str,
    sizes: list[int],
    max_workers: int,
    primary_far_1k: float = 0.2,
    override_sizes: dict[int, tuple[int, int]] | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Executa calibração e validação oficiais (maiores `n` primeiro) e salva manifesto."""
    calib_cfg = config["calibracao"]
    calib_seed_start = int(calib_cfg["faixa_sementes_calibracao"][0])
    val_seed_start = int(calib_cfg["faixa_sementes_validacao"][0])
    targets_far_1k = sorted(set([float(x) for x in calib_cfg["alvos_far_1k"]] + [primary_far_1k]))
    noises = [p["tipo"] for p in config["grid"]["perfis_ruido"]]

    is_provisional = (override_sizes is not None) or (set(sizes) != set(config["grid"]["tamanhos_n"]))

    manifest: dict[str, Any] = {
        "_meta": {
            "provisorio": is_provisional,
            "alvo_principal_far_1k": primary_far_1k,
            "alvos_far_1k": targets_far_1k,
            "tamanhos_por_n": {
                str(n): {
                    "num_series_calibracao": calibration_sizes_for_n(n, override_sizes)[0],
                    "num_series_validacao": calibration_sizes_for_n(n, override_sizes)[1],
                    "faixa_sementes_calibracao": [calib_seed_start, calib_seed_start + calibration_sizes_for_n(n, override_sizes)[0]],
                    "faixa_sementes_validacao": [val_seed_start, val_seed_start + calibration_sizes_for_n(n, override_sizes)[1]],
                }
                for n in sorted(sizes)
            },
            "tamanhos_n": sorted(sizes),
            "jump_ruptures": 1,
            "config_sha256": compute_config_hash(config_path),
            "git_hash": git_hash,
        },
        "detalhes_por_alvo": {},
    }

    primary_key = f"{primary_far_1k:.1f}"
    pairs = [(n, p) for n in sorted(sizes, reverse=True) for p in noises]
    total_pairs = len(pairs)

    use_exec = max_workers > 1
    ctx = ProcessPoolExecutor(max_workers=max_workers, initializer=_init_calib_worker) if use_exec else None
    try:
        if not use_exec:
            warmup_edivisive_jit()
        for idx, (n, perfil) in enumerate(pairs, start=1):
            n_calib, n_val = calibration_sizes_for_n(n, override_sizes)
            res = calibrate_pair(
                n=n,
                perfil=perfil,
                n_calib=n_calib,
                n_val=n_val,
                calib_seed_start=calib_seed_start,
                val_seed_start=val_seed_start,
                targets_far_1k=targets_far_1k,
                executor=ctx,
                num_chunks=max(1, max_workers * 2),
            )
            manifest["detalhes_por_alvo"].setdefault(perfil, {})[str(n)] = res["per_target"]

            prim = res["per_target"][primary_key]
            for det_key in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2"):
                manifest.setdefault(f"{det_key}_perfil", {}).setdefault(perfil, {})[str(n)] = prim[det_key]["param"]
                if perfil == "gaussian":
                    manifest.setdefault(f"{det_key}_gauss", {})[str(n)] = prim[det_key]["param"]
            manifest.setdefault("EDiv_Means", {}).setdefault(perfil, {})[str(n)] = prim["EDiv_Means"]["param"]
            manifest.setdefault("EDiv_Medians", {}).setdefault(perfil, {})[str(n)] = prim["EDiv_Medians"]["param"]

            if log_fn:
                flags_summary = [
                    f"{k}(p={prim[k]['param']},val={prim[k]['val']['far_1k']:.2f}"
                    f"{',SINF' if prim[k]['saturado_inferior'] else ''}"
                    f"{',SSUP' if prim[k]['saturado_superior'] else ''}"
                    f"{',VFAL' if prim[k]['validacao_falhou'] else ''})"
                    for k in ("CUSUM", "PELT_L2", "PELT_L1", "BINSEG_L2", "EDiv_Means", "EDiv_Medians")
                ]
                log_fn(
                    f"[Calibração {idx}/{total_pairs}] n={n} perfil={perfil} ({n_calib}c/{n_val}v) "
                    f"tempo={res['elapsed_s']:.1f}s | {' '.join(flags_summary)}"
                )
    finally:
        if ctx is not None:
            ctx.shutdown(wait=True)

    out_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_manifest_path.with_suffix(f".tmp_{os.getpid()}.json")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_path.rename(out_manifest_path)
    return manifest
