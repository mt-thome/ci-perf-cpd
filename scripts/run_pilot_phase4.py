"""Script orquestrador da Fase 4: Teste de Fumaça, Calibração Oficial e Piloto (R=3).

Modos:
  --mode smoke      Etapa (a): calibração provisória (30/15), n in {100,500}, R=2
  --mode calibrate  Etapa (b): calibração oficial (200/100), n in {100,500,1000}
  --mode pilot      Etapa (c): piloto R=3 usando a calibração oficial
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

# Fixar threads antes de importar NumPy/Numba
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

from cpd.detectors.edivisive import warmup_edivisive_jit
from cpd.experiments.calibration import run_calibration_suite
from cpd.experiments.runner import (
    execute_series_for_detectors,
    get_git_commit_hash,
    save_partition_atomic,
    validate_calibration_manifest,
)

REPO_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_DIR / "configs/experiments.yaml"
RESULTS_DIR = REPO_DIR / "results"
CALIB_DIR = RESULTS_DIR / "calibration"
PILOT_DIR = RESULTS_DIR / "pilot"
SMOKE_DIR = RESULTS_DIR / "smoke"
LOG_DIR = REPO_DIR / "logs"


class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = FlushingStreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = FlushingFileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def _init_eval_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    warmup_edivisive_jit()


def _eval_worker_task(payload: dict[str, Any]) -> tuple[str, int, float, list[dict[str, Any]]]:
    t0 = time.perf_counter()
    sc = payload["scenario"]
    seed_idx = payload["seed_idx"]
    rows = execute_series_for_detectors(
        scenario_cfg=sc,
        detectors_cfg=payload["detectors_cfg"],
        seed_idx=seed_idx,
        calibration_dict=payload["calib_manifest"],
        git_hash=payload["git_hash"],
    )
    elapsed_s = time.perf_counter() - t0
    return sc["cenario_id"], seed_idx, elapsed_s, rows


def build_scenarios(config: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    mags = config["grid"]["magnitudes_sigma"]
    scenarios: list[dict[str, Any]] = []
    counter = 0
    # Ordena do maior n para o menor para executar tarefas longas primeiro
    for n in sorted(sizes, reverse=True):
        for p_ruido in config["grid"]["perfis_ruido"]:
            ntype_str = p_ruido["tipo"]
            n_params = p_ruido["params"]
            counter += 1
            scenarios.append({
                "cenario_id": f"SC_{counter:04d}_n{n}_{ntype_str}_k0",
                "n": n,
                "perfil_ruido": ntype_str,
                "noise_params": n_params,
                "k_true": 0,
                "magnitude_sigma": 0.0,
            })
            cps_options = [1] if n == 100 else [1, 2]
            for k_true in cps_options:
                for mag in mags:
                    counter += 1
                    scenarios.append({
                        "cenario_id": f"SC_{counter:04d}_n{n}_{ntype_str}_k{k_true}_m{mag}",
                        "n": n,
                        "perfil_ruido": ntype_str,
                        "noise_params": n_params,
                        "k_true": k_true,
                        "magnitude_sigma": mag,
                    })
    return scenarios


def run_evaluation_grid(
    scenarios: list[dict[str, Any]],
    detectors_cfg: list[dict[str, Any]],
    r_seeds: int,
    calib_manifest: dict[str, Any],
    git_hash: str,
    out_dir: Path,
    out_filename: str,
    max_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    resumed_rows: list[dict[str, Any]] = []
    for sc in scenarios:
        for seed_idx in range(1, r_seeds + 1):
            part_path = parts_dir / f"{sc['cenario_id']}_s{seed_idx}.parquet"
            if part_path.exists():
                resumed_rows.extend(pd.read_parquet(part_path).to_dict(orient="records"))
            else:
                tasks.append({
                    "scenario": sc,
                    "seed_idx": seed_idx,
                    "detectors_cfg": detectors_cfg,
                    "calib_manifest": calib_manifest,
                    "git_hash": git_hash,
                })

    total_series = len(scenarios) * r_seeds
    done_series = total_series - len(tasks)
    logger.info(
        f"Avaliação: {total_series} séries ({total_series * len(detectors_cfg)} linhas) | "
        f"retomadas={done_series} | pendentes={len(tasks)} | workers={max_workers}"
    )

    all_rows = list(resumed_rows)
    if tasks:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_eval_worker) as executor:
            futures = {executor.submit(_eval_worker_task, t): t for t in tasks}
            for fut in as_completed(futures):
                sc_id, seed_idx, elapsed_s, rows = fut.result()
                part_path = parts_dir / f"{sc_id}_s{seed_idx}.parquet"
                save_partition_atomic(pd.DataFrame(rows), part_path)
                all_rows.extend(rows)
                done_series += 1
                logger.info(
                    f"[{done_series}/{total_series}] ({100.0 * done_series / total_series:5.1f}%) "
                    f"cenario={sc_id} seed={seed_idx} ({len(rows)} detectores) tempo={elapsed_s:.2f}s"
                )

    df = pd.DataFrame(all_rows)
    out_parquet = out_dir / out_filename
    save_partition_atomic(df, out_parquet)
    logger.info(f"Parquet consolidado salvo atomicamente em: {out_parquet} ({len(df)} linhas)")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador Fase 4 (Smoke / Calibração / Piloto)")
    parser.add_argument("--mode", choices=["smoke", "calibrate", "pilot"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    git_hash = get_git_commit_hash(REPO_DIR)

    if args.mode == "smoke":
        logger = setup_logger(LOG_DIR / "smoke.log")
        logger.info(f"=== ETAPA (a): TESTE DE FUMAÇA | workers={args.workers} | git={git_hash} ===")
        smoke_sizes = [100, 500]
        smoke_calib_path = SMOKE_DIR / "calibration_smoke_provisional.json"
        calib_manifest = run_calibration_suite(
            config=config,
            config_path=CONFIG_PATH,
            out_manifest_path=smoke_calib_path,
            git_hash=git_hash,
            sizes=smoke_sizes,
            n_calib=30,
            n_val=15,
            max_workers=args.workers,
            primary_far_1k=0.2,
            log_fn=logger.info,
        )
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=True)
        scenarios = build_scenarios(config, smoke_sizes)
        df = run_evaluation_grid(
            scenarios=scenarios,
            detectors_cfg=config["detectores"],
            r_seeds=2,
            calib_manifest=calib_manifest,
            git_hash=git_hash,
            out_dir=SMOKE_DIR,
            out_filename="smoke_raw_detections.parquet",
            max_workers=args.workers,
            logger=logger,
        )
        expected_rows = len(scenarios) * 2 * len(config["detectores"])
        logger.info(f"VERIFICAÇÃO SMOKE: esperado={expected_rows} | obtido={len(df)}")

    elif args.mode == "calibrate":
        logger = setup_logger(LOG_DIR / "calibration.log")
        logger.info(f"=== ETAPA (b): CALIBRAÇÃO OFICIAL (200/100) | workers={args.workers} | git={git_hash} ===")
        calib_cfg = config["calibracao"]
        calib_path = CALIB_DIR / "calibration_manifest.json"
        calib_manifest = run_calibration_suite(
            config=config,
            config_path=CONFIG_PATH,
            out_manifest_path=calib_path,
            git_hash=git_hash,
            sizes=config["grid"]["tamanhos_n"],
            n_calib=int(calib_cfg["num_series_calibracao"]),
            n_val=int(calib_cfg["num_series_validacao"]),
            max_workers=args.workers,
            primary_far_1k=0.2,
            log_fn=logger.info,
        )
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=False)
        logger.info(f"Calibração oficial salva e validada em: {calib_path}")

    elif args.mode == "pilot":
        logger = setup_logger(LOG_DIR / "pilot.log")
        logger.info(f"=== ETAPA (c): PILOTO R=3 | workers={args.workers} | git={git_hash} ===")
        calib_path = CALIB_DIR / "calibration_manifest.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"Manifesto oficial não encontrado: {calib_path}. Rode --mode calibrate antes.")
        with open(calib_path) as f:
            calib_manifest = json.load(f)
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=False)
        scenarios = build_scenarios(config, config["grid"]["tamanhos_n"])
        df = run_evaluation_grid(
            scenarios=scenarios,
            detectors_cfg=config["detectores"],
            r_seeds=int(config["grid"]["repeticoes_piloto"]),
            calib_manifest=calib_manifest,
            git_hash=git_hash,
            out_dir=PILOT_DIR,
            out_filename="pilot_raw_detections.parquet",
            max_workers=args.workers,
            logger=logger,
        )
        expected_rows = len(scenarios) * int(config["grid"]["repeticoes_piloto"]) * len(config["detectores"])
        logger.info(f"VERIFICAÇÃO PILOTO: esperado={expected_rows} | obtido={len(df)}")


if __name__ == "__main__":
    main()
