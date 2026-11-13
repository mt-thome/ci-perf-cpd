"""Orquestrador completo da Fase 4: Smoke, Calibração Oficial, Piloto (R=3) e Grid (R=50).

Modos:
  --mode smoke      Teste de fumaça provisório
  --mode calibrate  Etapa 1: Calibração oficial (2000/1000 em n=100; 200/100 em n=500,1000)
  --mode pilot      Etapa 2: Piloto R=3 com critérios de parada automáticos
  --mode grid       Etapa 3: Rodagem final R=50 com ETA, retomada e relatório de integridade
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

from cpd.detectors.edivisive import warmup_edivisive_jit
from cpd.evaluation.metrics import evaluate_single_series
from cpd.experiments.calibration import poisson_rate_ci_95, run_calibration_suite
from cpd.experiments.runner import (
    compute_config_hash,
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
GRID_DIR = RESULTS_DIR / "raw_detections"
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
    fh = FlushingFileHandler(log_file, mode="a", encoding="utf-8")
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
    return sc["cenario_id"], seed_idx, time.perf_counter() - t0, rows


def build_scenarios(config: dict[str, Any], sizes: list[int]) -> list[dict[str, Any]]:
    mags = config["grid"]["magnitudes_sigma"]
    scenarios: list[dict[str, Any]] = []
    counter = 0
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
    parts_dir: Path,
    out_parquet: Path,
    max_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
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
    error_counts: Counter[str] = Counter()
    error_tracebacks: dict[str, str] = {}
    max_allowed_errors = max(1, int(0.01 * total_series))
    t_eval_start = time.perf_counter()
    newly_done = 0

    if tasks:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_eval_worker) as executor:
            futures = {executor.submit(_eval_worker_task, t): t for t in tasks}
            for fut in as_completed(futures):
                t_info = futures[fut]
                try:
                    sc_id, seed_idx, elapsed_s, rows = fut.result()
                except Exception as exc:
                    exc_type = type(exc).__name__
                    error_counts[exc_type] += 1
                    error_tracebacks.setdefault(exc_type, traceback.format_exc())
                    logger.error(f"ERRO em {t_info['scenario']['cenario_id']} s={t_info['seed_idx']}: {exc}")
                    if error_counts[exc_type] > max_allowed_errors:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise RuntimeError(
                            f"CRITÉRIO DE PARADA: exceção recorrente {exc_type} ({error_counts[exc_type]} > {max_allowed_errors}).\n"
                            f"Traceback:\n{error_tracebacks[exc_type]}"
                        )
                    continue

                part_path = parts_dir / f"{sc_id}_s{seed_idx}.parquet"
                save_partition_atomic(pd.DataFrame(rows), part_path)
                all_rows.extend(rows)
                done_series += 1
                newly_done += 1
                wall_elapsed = time.perf_counter() - t_eval_start
                rate = newly_done / max(wall_elapsed, 1e-6)
                eta_min = ((len(tasks) - newly_done) / max(rate, 1e-9)) / 60.0
                logger.info(
                    f"[{done_series}/{total_series}] ({100.0 * done_series / total_series:5.1f}%) "
                    f"cenario={sc_id} seed={seed_idx} tempo={elapsed_s:.2f}s ETA={eta_min:.1f}m"
                )

    df = pd.DataFrame(all_rows)
    save_partition_atomic(df, out_parquet)
    logger.info(f"Parquet consolidado salvo atomicamente em: {out_parquet} ({len(df)} linhas)")
    return df


def generate_calibration_report(manifest: dict[str, Any], out_path: Path, elapsed_s: float) -> tuple[bool, str]:
    """Gera calibration_report.md e verifica critério de parada (>30% células com flags no alvo 0.2)."""
    prim_key = f"{manifest['_meta']['alvo_principal_far_1k']:.1f}"
    details = manifest["detalhes_por_alvo"]
    lines = [
        "# Relatório de Calibração e Validação Oficial (Fase 4)",
        "",
        f"- **Tempo total**: {elapsed_s / 60.0:.2f} min ({elapsed_s:.1f} s)",
        f"- **Provisório**: `{manifest['_meta']['provisorio']}`",
        f"- **Config SHA-256**: `{manifest['_meta']['config_sha256']}`",
        f"- **Git Hash**: `{manifest['_meta']['git_hash']}`",
        "",
    ]

    total_cells = 0
    flagged_cells = 0
    for tgt_key in [f"{t:.1f}" for t in manifest["_meta"]["alvos_far_1k"]]:
        lines.append(f"## Alvo `FAR_1k <= {tgt_key}`")
        lines.append("")
        lines.append("| Perfil | n | Detector | Param | Calib FAR_1k (IC 95%) | Calib Alarmes (Obs/Esp) | Val FAR_1k (IC 95%) | Val Alarmes (Obs/Esp) | Flags |")
        lines.append("|---|---:|---|---:|---|---|---|---|---|")
        for perfil, n_dict in sorted(details.items()):
            for n_str, tgt_dict in sorted(n_dict.items(), key=lambda x: int(x[0])):
                for det_key, rec in tgt_dict[tgt_key].items():
                    c = rec["calib"]
                    v = rec["val"]
                    flags = []
                    if rec["saturado_inferior"]:
                        flags.append("saturado_inferior")
                    if rec["saturado_superior"]:
                        flags.append("saturado_superior")
                    if rec["validacao_falhou"]:
                        flags.append("validacao_falhou")
                    flag_str = ", ".join(flags) if flags else "OK"
                    if tgt_key == prim_key:
                        total_cells += 1
                        if flags:
                            flagged_cells += 1
                    lines.append(
                        f"| {perfil} | {n_str} | {det_key} | {rec['param']} | "
                        f"{c['far_1k']:.3f} [{c['ci95_far_1k'][0]:.3f}, {c['ci95_far_1k'][1]:.3f}] | "
                        f"{c['total_alarms']} / {c['expected_alarms']:.1f} | "
                        f"{v['far_1k']:.3f} [{v['ci95_far_1k'][0]:.3f}, {v['ci95_far_1k'][1]:.3f}] | "
                        f"{v['total_alarms']} / {v['expected_alarms']:.1f} | {flag_str} |"
                    )
        lines.append("")

    frac_flagged = flagged_cells / max(total_cells, 1)
    stop_triggered = frac_flagged > 0.30
    summary = (
        f"Células com flags no alvo primário ({prim_key}): {flagged_cells}/{total_cells} ({100*frac_flagged:.1f}%)"
    )
    lines.append(f"**Resumo do critério de parada (>30% flags no alvo primário)**: {summary} -> {'PARAR' if stop_triggered else 'APROVADO'}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return stop_triggered, summary


def verify_and_report_grid(
    df: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
    expected_rows: int,
    expected_seeds: list[int],
    out_report_path: Path,
) -> tuple[bool, list[str]]:
    """Executa verificações automáticas pós-execução e salva relatório."""
    failures: list[str] = []
    if len(df) != expected_rows:
        failures.append(f"Cardinalidade divergente: esperado={expected_rows}, obtido={len(df)}")

    dups = int(df.duplicated(subset=["cenario_id", "semente", "detector"]).sum())
    if dups > 0:
        failures.append(f"Linhas duplicadas encontradas: {dups}")

    seed_lists = df.groupby(["cenario_id", "detector"])["semente"].apply(lambda s: sorted(s.tolist()))
    if not (seed_lists == expected_seeds).all():
        failures.append("Sementes não pareadas em 100% dos pares (cenario_id, detector)")

    if not (df["versao_codigo"] == manifest["_meta"]["git_hash"]).all():
        failures.append("versao_codigo no Parquet diverge do git_hash do manifesto de calibração")

    if compute_config_hash(CONFIG_PATH) != manifest["_meta"]["config_sha256"]:
        failures.append("Hash de configuração atual diverge do manifesto de calibração")

    # Tabela FAR estacionário no grid vs validação
    prim_key = f"{manifest['_meta']['alvo_principal_far_1k']:.1f}"
    stat = df[df["magnitude"] == 0.0]
    lines = [
        "# Relatório de Integridade do Grid (Fase 4)",
        "",
        f"- **Linhas**: {len(df)} (esperado: {expected_rows})",
        f"- **Duplicatas**: {dups}",
        f"- **Sementes pareadas**: {'OK' if 'Sementes' not in str(failures) else 'FALHA'}",
        f"- **Git Hash**: `{manifest['_meta']['git_hash']}`",
        f"- **Config SHA-256**: `{manifest['_meta']['config_sha256']}`",
        "",
        "## FAR em Séries Estacionárias (Grid vs. Validação)",
        "",
        "| Perfil | n | Detector | Grid FAR_series | Grid FAR_1k (IC 95%) | Val FAR_1k (IC 95%) |",
        "|---|---:|---|---:|---|---|",
    ]

    det_to_calkey = {
        "CUSUM_gauss": "CUSUM", "CUSUM_perfil": "CUSUM",
        "PELT_L2_gauss": "PELT_L2", "PELT_L2_perfil": "PELT_L2",
        "PELT_L1_gauss": "PELT_L1", "PELT_L1_perfil": "PELT_L1",
        "Binseg_L2_gauss": "BINSEG_L2", "Binseg_L2_perfil": "BINSEG_L2",
        "EDiv_Means_base": "EDiv_Means", "EDiv_Medians_base": "EDiv_Medians",
    }
    for (perfil, n_val, det), grp in stat.groupby(["perfil_ruido", "n", "detector"]):
        alarms = [len(x) for x in grp["indices_detectados"]]
        total_al = sum(alarms)
        exp_commits = len(grp) * int(n_val)
        f1k = (total_al / exp_commits) * 1000.0
        fser = sum(1 for a in alarms if a > 0) / len(grp)
        ci_l, ci_h = poisson_rate_ci_95(total_al, exp_commits)
        cal_k = det_to_calkey.get(det)
        val_rec = manifest["detalhes_por_alvo"][perfil][str(int(n_val))][prim_key][cal_k]["val"]
        lines.append(
            f"| {perfil} | {int(n_val)} | {det} | {fser:.3f} | "
            f"{f1k:.3f} [{ci_l:.3f}, {ci_h:.3f}] | "
            f"{val_rec['far_1k']:.3f} [{val_rec['ci95_far_1k'][0]:.3f}, {val_rec['ci95_far_1k'][1]:.3f}] |"
        )

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(failures) == 0, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestrador Fase 4")
    parser.add_argument("--mode", choices=["smoke", "calibrate", "pilot", "grid"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    git_hash = get_git_commit_hash(REPO_DIR)

    if args.mode == "smoke":
        logger = setup_logger(LOG_DIR / "smoke.log")
        smoke_sizes = [100, 500]
        calib_manifest = run_calibration_suite(
            config=config, config_path=CONFIG_PATH,
            out_manifest_path=SMOKE_DIR / "calibration_smoke_provisional.json",
            git_hash=git_hash, sizes=smoke_sizes, max_workers=args.workers,
            primary_far_1k=0.2, override_sizes={100: (30, 15), 500: (30, 15)}, log_fn=logger.info,
        )
        scenarios = build_scenarios(config, smoke_sizes)
        run_evaluation_grid(
            scenarios, config["detectores"], 2, calib_manifest, git_hash,
            SMOKE_DIR / "parts", SMOKE_DIR / "smoke_raw_detections.parquet", args.workers, logger,
        )

    elif args.mode == "calibrate":
        logger = setup_logger(LOG_DIR / "calibration.log")
        logger.info(f"=== ETAPA 1: CALIBRAÇÃO OFICIAL | workers={args.workers} | git={git_hash} ===")
        t0 = time.perf_counter()
        calib_path = CALIB_DIR / "calibration_manifest.json"
        calib_manifest = run_calibration_suite(
            config=config, config_path=CONFIG_PATH, out_manifest_path=calib_path,
            git_hash=git_hash, sizes=config["grid"]["tamanhos_n"],
            max_workers=args.workers, primary_far_1k=0.2, log_fn=logger.info,
        )
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=False)
        elapsed_s = time.perf_counter() - t0
        stop, summary = generate_calibration_report(calib_manifest, CALIB_DIR / "calibration_report.md", elapsed_s)
        logger.info(summary)
        if stop:
            logger.error("CRITÉRIO DE PARADA DISPARADO na Etapa 1 (>30% células com flags no alvo 0.2).")
            sys.exit(2)

    elif args.mode == "pilot":
        logger = setup_logger(LOG_DIR / "pilot.log")
        logger.info(f"=== ETAPA 2: PILOTO R=3 | workers={args.workers} | git={git_hash} ===")

        # 1. Verificar pytest e git limpo
        status_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip()
        if status_out:
            logger.error(f"CRITÉRIO DE PARADA: árvore git suja:\n{status_out}")
            sys.exit(2)

        pt = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, capture_output=True, text=True)
        logger.info(f"Pytest: exit={pt.returncode} | {pt.stdout.strip()}")
        if pt.returncode != 0:
            logger.error(f"CRITÉRIO DE PARADA: pytest falhou:\n{pt.stdout}\n{pt.stderr}")
            sys.exit(2)

        calib_path = CALIB_DIR / "calibration_manifest.json"
        with open(calib_path) as f:
            calib_manifest = json.load(f)
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=False)
        if calib_manifest["_meta"]["git_hash"] != git_hash:
            logger.error("CRITÉRIO DE PARADA: git_hash atual difere do manifesto de calibração.")
            sys.exit(2)

        t0 = time.perf_counter()
        scenarios = build_scenarios(config, config["grid"]["tamanhos_n"])
        r_pilot = int(config["grid"]["repeticoes_piloto"])
        df = run_evaluation_grid(
            scenarios, config["detectores"], r_pilot, calib_manifest, git_hash,
            PILOT_DIR / "parts", PILOT_DIR / "pilot_raw_detections.parquet", args.workers, logger,
        )
        pilot_wall_s = time.perf_counter() - t0
        proj_grid_h = (pilot_wall_s * (50.0 / r_pilot)) / 3600.0
        logger.info(f"Piloto concluído em {pilot_wall_s/60:.2f} min | Projeção R=50 ({args.workers} workers): {proj_grid_h:.2f} h")

        expected_rows = len(scenarios) * r_pilot * len(config["detectores"])
        ok, failures = verify_and_report_grid(
            df, calib_manifest, config, expected_rows, list(range(1, r_pilot + 1)),
            PILOT_DIR / "pilot_integrity_report.md",
        )
        if not ok:
            logger.error(f"CRITÉRIO DE PARADA (integridade): {failures}")
            sys.exit(2)

        # Verificar F1(k=5) >= 0.90 em gaussian 5sigma n>=500 para detectores não saturados
        prim_key = "0.2"
        sub = df[(df["perfil_ruido"] == "gaussian") & (df["n"] >= 500) & (df["magnitude"] == 5.0)]
        det_to_cal = {
            "CUSUM_gauss": "CUSUM", "CUSUM_perfil": "CUSUM",
            "PELT_L2_gauss": "PELT_L2", "PELT_L2_perfil": "PELT_L2",
            "PELT_L1_gauss": "PELT_L1", "PELT_L1_perfil": "PELT_L1",
            "Binseg_L2_gauss": "BINSEG_L2", "Binseg_L2_perfil": "BINSEG_L2",
            "EDiv_Means_base": "EDiv_Means", "EDiv_Medians_base": "EDiv_Medians",
        }
        for det, grp in sub.groupby("detector"):
            cal_k = det_to_cal[det]
            sat_any = any(
                calib_manifest["detalhes_por_alvo"]["gaussian"][str(nv)][prim_key][cal_k]["saturado_inferior"]
                or calib_manifest["detalhes_por_alvo"]["gaussian"][str(nv)][prim_key][cal_k]["saturado_superior"]
                for nv in ("500", "1000")
            )
            tps = fps = fns = 0
            for _, r in grp.iterrows():
                m = evaluate_single_series(list(r["ground_truth"]), list(r["indices_detectados"]), k=5)
                tps += m.tp; fps += m.fp; fns += m.fn
            p = tps / (tps + fps) if (tps + fps) else 0.0
            rc = tps / (tps + fns) if (tps + fns) else 0.0
            f1 = 2 * p * rc / (p + rc) if (p + rc) else 0.0
            logger.info(f"Sanidade Piloto (gauss 5sigma n>=500) {det}: F1(k=5)={f1:.3f} (sat={sat_any})")
            if not sat_any and f1 < 0.90:
                logger.error(f"CRITÉRIO DE PARADA: {det} não saturado com F1(k=5)={f1:.3f} < 0.90")
                sys.exit(2)

        if proj_grid_h > 8.0:
            logger.error(f"CRITÉRIO DE PARADA: projeção do grid ({proj_grid_h:.2f} h) > 8.0 h")
            sys.exit(2)

    elif args.mode == "grid":
        logger = setup_logger(LOG_DIR / "grid.log")
        logger.info(f"=== ETAPA 3: GRID COMPLETO R=50 | workers={args.workers} | git={git_hash} ===")
        calib_path = CALIB_DIR / "calibration_manifest.json"
        with open(calib_path) as f:
            calib_manifest = json.load(f)
        validate_calibration_manifest(calib_manifest, CONFIG_PATH, allow_provisional=False)
        if calib_manifest["_meta"]["git_hash"] != git_hash:
            logger.error("CRITÉRIO DE PARADA: git_hash atual diverge do manifesto de calibração.")
            sys.exit(2)

        scenarios = build_scenarios(config, config["grid"]["tamanhos_n"])
        r_eval = int(config["grid"]["repeticoes_avaliacao"])
        df = run_evaluation_grid(
            scenarios, config["detectores"], r_eval, calib_manifest, git_hash,
            GRID_DIR / "parts", RESULTS_DIR / "raw_detections.parquet", args.workers, logger,
        )
        expected_rows = len(scenarios) * r_eval * len(config["detectores"])
        ok, failures = verify_and_report_grid(
            df, calib_manifest, config, expected_rows, list(range(1, r_eval + 1)),
            RESULTS_DIR / "grid_integrity_report.md",
        )
        if not ok:
            logger.error(f"CRITÉRIO DE PARADA (integridade do grid): {failures}")
            sys.exit(2)


if __name__ == "__main__":
    main()
