"""Bloco 3 (ADR-019): Reexecução idempotente de Binseg_L2_gauss e PELT_L2_gauss (R=50).

Executa apenas Binseg_L2_gauss e PELT_L2_gauss com 10 workers, salva checkpoint
idempotente por cenário, atualiza results/raw_detections.parquet (marcando
PELT_L1_gauss como valido=False) e grava logs em logs/rerun_gauss.log.

Uso:
    .venv/bin/python scripts/rerun_gauss.py
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cpd.experiments.runner import (
    execute_series_for_detectors,
    get_git_commit_hash,
    save_partition_atomic,
    validate_calibration_manifest,
)

REPO_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_DIR / "configs/experiments.yaml"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
CHECKPOINT_PARQUET = REPO_DIR / "results/rerun_gauss_checkpoint.parquet"
LOG_PATH = REPO_DIR / "logs/rerun_gauss.log"

TARGET_DETECTORS = {"Binseg_L2_gauss", "PELT_L2_gauss"}


def _build_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    sizes = sorted([int(s) for s in config["grid"]["tamanhos_n"]], reverse=True)
    mags = config["grid"]["magnitudes_sigma"]
    scenarios: list[dict[str, Any]] = []
    counter = 0
    for n in sizes:
        for p_ruido in config["grid"]["perfis_ruido"]:
            ntype_str = p_ruido["tipo"]
            n_params = p_ruido["params"]
            counter += 1
            scenarios.append({
                "cenario_id": f"SC_{counter:04d}_n{n}_{ntype_str}_k0",
                "n": int(n),
                "perfil_ruido": ntype_str,
                "noise_params": n_params,
                "k_true": 0,
                "magnitude_sigma": 0.0,
            })
            cps_options = [1] if int(n) == 100 else [1, 2]
            for k_true in cps_options:
                for mag in mags:
                    counter += 1
                    scenarios.append({
                        "cenario_id": f"SC_{counter:04d}_n{n}_{ntype_str}_k{k_true}_m{mag}",
                        "n": int(n),
                        "perfil_ruido": ntype_str,
                        "noise_params": n_params,
                        "k_true": int(k_true),
                        "magnitude_sigma": float(mag),
                    })
    return scenarios


def _run_scenario_worker(
    scenario_cfg: dict[str, Any],
    detectors_cfg: list[dict[str, Any]],
    r_reps: int,
    calib_manifest: dict[str, Any],
    git_hash: str,
) -> tuple[str, list[dict[str, Any]]]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"

    rows: list[dict[str, Any]] = []
    for seed_idx in range(1, r_reps + 1):
        rows.extend(
            execute_series_for_detectors(
                scenario_cfg=scenario_cfg,
                detectors_cfg=detectors_cfg,
                seed_idx=seed_idx,
                calibration_dict=calib_manifest,
                git_hash=git_hash,
            )
        )
    return scenario_cfg["cenario_id"], rows


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("rerun_gauss")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_calibration_manifest(manifest, CONFIG_PATH, allow_provisional=False)

    git_hash = get_git_commit_hash(REPO_DIR)
    r_reps = int(config["grid"]["repeticoes_avaliacao"])
    detectors_cfg = [d for d in config["detectores"] if d["nome"] in TARGET_DETECTORS]
    assert len(detectors_cfg) == 2, f"Esperados 2 detectores, obtidos {[d['nome'] for d in detectors_cfg]}"

    scenarios = _build_scenarios(config)
    ref_df = pd.read_parquet(RAW_PARQUET, columns=["detector", "cenario_id"])
    ref_cids = set(ref_df[ref_df["detector"] == "CUSUM_perfil"]["cenario_id"].unique().tolist())
    assert {sc["cenario_id"] for sc in scenarios} == ref_cids, "cenario_id mismatch contra CUSUM_perfil!"
    logger.info("Iniciando reexecução idempotente | git_hash=%s | R=%d | cenários=%d | workers=10", git_hash, r_reps, len(scenarios))

    done_ids: set[str] = set()
    all_new_rows: list[dict[str, Any]] = []
    if CHECKPOINT_PARQUET.exists():
        ckpt_df = pd.read_parquet(CHECKPOINT_PARQUET)
        counts = ckpt_df.groupby("cenario_id").size()
        done_ids = set(counts[counts == r_reps * len(detectors_cfg)].index.tolist())
        if done_ids:
            ckpt_df = ckpt_df[ckpt_df["cenario_id"].isin(done_ids)]
            all_new_rows = ckpt_df.to_dict(orient="records")
        logger.info("Checkpoint carregado: %d/%d cenários já concluídos.", len(done_ids), len(scenarios))

    pending = [sc for sc in scenarios if sc["cenario_id"] not in done_ids]
    t_start = time.perf_counter()

    if pending:
        with ProcessPoolExecutor(max_workers=10) as pool:
            futures = {
                pool.submit(_run_scenario_worker, sc, detectors_cfg, r_reps, manifest, git_hash): sc["cenario_id"]
                for sc in pending
            }
            completed_since_ckpt = 0
            for fut in as_completed(futures):
                cid, rows = fut.result()
                all_new_rows.extend(rows)
                done_ids.add(cid)
                completed_since_ckpt += 1
                if completed_since_ckpt % 10 == 0 or len(done_ids) == len(scenarios):
                    save_partition_atomic(pd.DataFrame(all_new_rows), CHECKPOINT_PARQUET)
                    elapsed = time.perf_counter() - t_start
                    logger.info("Progresso: %d/%d cenários (%d linhas) em %.1f s", len(done_ids), len(scenarios), len(all_new_rows), elapsed)

    new_df = pd.DataFrame(all_new_rows)
    assert len(new_df) == len(scenarios) * r_reps * len(detectors_cfg), f"Esperado 11500 linhas, obtido {len(new_df)}"
    new_df["valido"] = True

    logger.info("Mesclando 11.500 novas linhas em %s...", RAW_PARQUET)
    old_df = pd.read_parquet(RAW_PARQUET)
    if "valido" not in old_df.columns:
        old_df["valido"] = True
    kept_df = old_df[~old_df["detector"].isin(TARGET_DETECTORS)].copy()
    kept_df.loc[kept_df["detector"] == "PELT_L1_gauss", "valido"] = False

    merged_df = pd.concat([kept_df, new_df[kept_df.columns]], ignore_index=True)
    assert len(merged_df) == 57500, f"Esperado 57500 linhas totais, obtido {len(merged_df)}"
    save_partition_atomic(merged_df, RAW_PARQUET)

    if CHECKPOINT_PARQUET.exists():
        CHECKPOINT_PARQUET.unlink()

    logger.info("Parquet atualizado com sucesso! Novo versao_codigo=%s em %d linhas.", git_hash, len(new_df))


if __name__ == "__main__":
    main()
