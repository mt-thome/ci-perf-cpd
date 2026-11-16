"""Fase 5: Benchmark de custo computacional (tempo e memória) em subprocessos isolados.

Cada medição é executada em um subprocesso novo com 1 thread estrita
(OMP/OPENBLAS/MKL/NUMBA_NUM_THREADS=1), aquecimento prévio do Numba JIT
excluído da medição, tempo via time.perf_counter() e pico de memória via
resource.getrusage(RUSAGE_SELF).ru_maxrss.

Uso:
    .venv/bin/python scripts/benchmark_cost.py
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
OUT_CSV = REPO_DIR / "results/tables/cost_benchmark.csv"
OUT_EXP_CSV = REPO_DIR / "results/tables/cost_benchmark_exponents.csv"


def _run_worker(detector_name: str, n: int, seed: int, param_val: float) -> None:
    """Executa uma única medição isolada dentro de um subprocesso."""
    from cpd.detectors import (
        EDivisiveMeansDetector,
        EDivisiveMediansDetector,
        RobustCUSUMDetector,
        RupturesDetector,
    )
    from cpd.generators.synthetic import NoiseType, generate_synthetic_series

    # Aquecimento explícito do JIT (Numba) fora da janela de medição
    if detector_name == "EDiv_Medians":
        warm_s = generate_synthetic_series(
            n=60, base_level=100.0, noise_scale=4.0,
            noise_type=NoiseType.GAUSSIAN, change_points=[30],
            magnitudes_std=2.0, seed=999,
        ).values
        EDivisiveMediansDetector(alpha=0.20, permutations=5, min_size=25, seed=1, exact_medians=True).detect(warm_s)

    series = generate_synthetic_series(
        n=n,
        base_level=100.0,
        noise_scale=4.0,
        noise_type=NoiseType.GAUSSIAN,
        change_points=[int(0.5 * n)],
        magnitudes_std=2.0,
        seed=seed,
    ).values

    if detector_name == "CUSUM":
        det = RobustCUSUMDetector(h=param_val, k_slack=0.5, init_window=30, guard_window=15)
    elif detector_name == "Binseg_L2":
        det = RupturesDetector(algorithm="binseg", cost_model="l2", pen=param_val, min_size=25, jump=1)
    elif detector_name == "PELT_L2":
        det = RupturesDetector(algorithm="pelt", cost_model="l2", pen=param_val, min_size=25, jump=1)
    elif detector_name == "EDiv_Means":
        det = EDivisiveMeansDetector(alpha=0.20, permutations=100, min_size=25, seed=seed)
    elif detector_name == "EDiv_Medians":
        det = EDivisiveMediansDetector(alpha=0.20, permutations=100, min_size=25, seed=seed, exact_medians=True)
    else:
        raise ValueError(f"Detector desconhecido: {detector_name}")

    t0 = time.perf_counter()
    det.detect(series)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print(json.dumps({"elapsed_ms": elapsed_ms, "max_rss_mb": max_rss_mb}))


def _manifest_param(manifest: dict, det: str, n: int) -> float:
    n_key = "100" if n <= 100 else ("500" if n <= 500 else "1000")
    if det == "CUSUM":
        return float(manifest["CUSUM_gauss"][n_key])
    if det == "Binseg_L2":
        return float(manifest["BINSEG_L2_gauss"][n_key])
    if det == "PELT_L2":
        return float(manifest["PELT_L2_gauss"][n_key])
    return 0.20


def _fit_loglog(ns: list[int], vals: list[float]) -> tuple[float, float, float, float]:
    """Ajuste log-log com IC de 95% para o expoente (usando distribuição t)."""
    x = np.log10(np.asarray(ns, dtype=float))
    y = np.log10(np.asarray(vals, dtype=float))
    res = stats.linregress(x, y)
    df = len(x) - 2
    t_crit = float(stats.t.ppf(0.975, df)) if df > 0 else float("nan")
    ci_low = float(res.slope - t_crit * res.stderr) if df > 0 else float("nan")
    ci_high = float(res.slope + t_crit * res.stderr) if df > 0 else float("nan")
    return float(res.slope), ci_low, ci_high, float(res.rvalue ** 2)


def run_benchmark() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    detectors = ["CUSUM", "Binseg_L2", "PELT_L2", "EDiv_Means", "EDiv_Medians"]
    sizes = [100, 200, 500, 1000, 2000]
    seeds = [50001, 50002, 50003]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMBA_NUM_THREADS"] = "1"

    raw_records = []
    for det in detectors:
        for n in sizes:
            param_val = _manifest_param(manifest, det, n)
            times_ms = []
            mems_mb = []
            status = "OK"
            for seed in seeds:
                cmd = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--detector", det,
                    "--n", str(n),
                    "--seed", str(seed),
                    "--param", str(param_val),
                ]
                try:
                    proc = subprocess.run(
                        cmd, env=env, capture_output=True, text=True, timeout=600, check=True
                    )
                    out = json.loads(proc.stdout.strip().splitlines()[-1])
                    times_ms.append(float(out["elapsed_ms"]))
                    mems_mb.append(float(out["max_rss_mb"]))
                    raw_records.append({
                        "detector": det, "n": n, "seed": seed, "param": param_val,
                        "elapsed_ms": float(out["elapsed_ms"]), "max_rss_mb": float(out["max_rss_mb"]),
                        "status": "OK",
                    })
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    raw_records.append({
                        "detector": det, "n": n, "seed": seed, "param": param_val,
                        "elapsed_ms": np.nan, "max_rss_mb": np.nan, "status": "timeout",
                    })
                    break
            if status == "timeout":
                print(f"[{det} n={n}] TIMEOUT (>600s)", flush=True)
            else:
                print(
                    f"[{det:12s} n={n:4d}] med={np.median(times_ms):9.3f} ms "
                    f"[{min(times_ms):.3f}, {max(times_ms):.3f}] | rss_med={np.median(mems_mb):.1f} MB",
                    flush=True,
                )

    raw_df = pd.DataFrame(raw_records)

    # Tabela resumo por (detector, n)
    summary_rows = []
    for (det, n), grp in raw_df.groupby(["detector", "n"], sort=False):
        ok = grp[grp["status"] == "OK"]
        if len(ok) == 0 or (grp["status"] == "timeout").any():
            summary_rows.append({
                "detector": det, "n": int(n), "param": float(grp["param"].iloc[0]),
                "reps": len(grp), "status": "timeout",
                "time_med_ms": np.nan, "time_min_ms": np.nan, "time_max_ms": np.nan,
                "rss_med_mb": np.nan, "rss_min_mb": np.nan, "rss_max_mb": np.nan,
            })
        else:
            t_vals = ok["elapsed_ms"].to_numpy()
            m_vals = ok["max_rss_mb"].to_numpy()
            summary_rows.append({
                "detector": det, "n": int(n), "param": float(ok["param"].iloc[0]),
                "reps": len(ok), "status": "OK",
                "time_med_ms": float(np.median(t_vals)),
                "time_min_ms": float(np.min(t_vals)),
                "time_max_ms": float(np.max(t_vals)),
                "rss_med_mb": float(np.median(m_vals)),
                "rss_min_mb": float(np.min(m_vals)),
                "rss_max_mb": float(np.max(m_vals)),
            })

    sum_df = pd.DataFrame(summary_rows)

    # Ajuste log-log para n >= 200
    exp_rows = []
    for det in detectors:
        sub = raw_df[(raw_df["detector"] == det) & (raw_df["n"] >= 200) & (raw_df["status"] == "OK")]
        n_range = f"{int(sub['n'].min())}..{int(sub['n'].max())}" if len(sub) else "N/A"
        if len(sub) >= 3 and sub["n"].nunique() >= 2:
            t_exp, t_l, t_h, t_r2 = _fit_loglog(sub["n"].tolist(), sub["elapsed_ms"].tolist())
            m_exp, m_l, m_h, m_r2 = _fit_loglog(sub["n"].tolist(), sub["max_rss_mb"].tolist())
        else:
            t_exp = t_l = t_h = t_r2 = m_exp = m_l = m_h = m_r2 = float("nan")
        exp_rows.append({
            "detector": det,
            "n_range": n_range,
            "n_points": len(sub),
            "time_exp": t_exp, "time_ci95_low": t_l, "time_ci95_high": t_h, "time_r2": t_r2,
            "mem_exp": m_exp, "mem_ci95_low": m_l, "mem_ci95_high": m_h, "mem_r2": m_r2,
        })
    exp_df = pd.DataFrame(exp_rows)

    # Anexar expoentes ao sum_df para exportação unificada
    merged_df = sum_df.merge(exp_df, on="detector", how="left")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUT_CSV, index=False)
    exp_df.to_csv(OUT_EXP_CSV, index=False)

    print("\n=== TABELA DE CUSTO COMPUTACIONAL (results/tables/cost_benchmark.csv) ===")
    print(sum_df.to_string(index=False))
    print("\n=== EXPOENTES LOG-LOG (n >= 200) ===")
    print(exp_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--detector", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--param", type=float)
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.detector, args.n, args.seed, args.param)
    else:
        run_benchmark()
