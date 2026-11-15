"""Estudo de equivalência ampliado do E-Divisive (200 séries estratificadas do grid).

Compara execução direta com alpha=0.05 (padrao) e alpha' calibrado (alvo 0.2)
contra a derivação via filter_edivisive_by_alpha(division_history, alpha) do Parquet.
Salva results/tables/edivisive_equivalence.csv.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yaml

from cpd.detectors import (
    EDivisiveMeansDetector,
    EDivisiveMediansDetector,
    filter_edivisive_by_alpha,
)
from cpd.detectors.edivisive import warmup_edivisive_jit
from cpd.evaluation.metrics import evaluate_single_series
from cpd.experiments.runner import derive_stable_seed, generate_scenario_series
from scripts.run_pilot_phase4 import build_scenarios

RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
CONFIG_PATH = REPO_DIR / "configs/experiments.yaml"
OUT_CSV = REPO_DIR / "results/tables/edivisive_equivalence.csv"


def _init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    warmup_edivisive_jit()


def _worker_task(payload: dict[str, Any]) -> dict[str, Any]:
    sc = payload["scenario"]
    seed_idx = int(payload["seed_idx"])
    alpha_cal_means = float(payload["alpha_cal_means"])
    alpha_cal_med = float(payload["alpha_cal_med"])

    series_obj = generate_scenario_series(sc, seed_idx)
    stable_seed = derive_stable_seed(sc["cenario_id"], seed_idx)

    out: dict[str, Any] = {
        "cenario_id": sc["cenario_id"],
        "semente": seed_idx,
        "perfil_ruido": sc["perfil_ruido"],
        "n": int(sc["n"]),
        "ground_truth": list(series_obj.change_points),
        "direct": {},
    }

    for key, cls, a_cal in (
        ("EDiv_Means", EDivisiveMeansDetector, alpha_cal_means),
        ("EDiv_Medians", EDivisiveMediansDetector, alpha_cal_med),
    ):
        kw = {"permutations": 100, "min_size": 25, "seed": stable_seed}
        if cls is EDivisiveMediansDetector:
            kw["exact_medians"] = True
        d_pad = cls(alpha=0.05, **kw).detect(series_obj.values)
        if abs(a_cal - 0.05) < 1e-12:
            d_cal = d_pad
        else:
            d_cal = cls(alpha=a_cal, **kw).detect(series_obj.values)
        out["direct"][f"{key}_padrao"] = d_pad
        out["direct"][f"{key}_calibrado"] = d_cal
    return out


def _paired_diff_bootstrap_ci(
    gt_list: list[list[int]],
    direct_list: list[list[int]],
    derived_list: list[list[int]],
    k_tol: int = 5,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    n = len(gt_list)
    tp_d = np.zeros(n); fp_d = np.zeros(n); fn_d = np.zeros(n)
    tp_v = np.zeros(n); fp_v = np.zeros(n); fn_v = np.zeros(n)
    for i in range(n):
        md = evaluate_single_series(gt_list[i], direct_list[i], k=k_tol)
        mv = evaluate_single_series(gt_list[i], derived_list[i], k=k_tol)
        tp_d[i], fp_d[i], fn_d[i] = md.tp, md.fp, md.fn
        tp_v[i], fp_v[i], fn_v[i] = mv.tp, mv.fp, mv.fn

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    pd0, rd0, fd0 = _prf(tp_d.sum(), fp_d.sum(), fn_d.sum())
    pv0, rv0, fv0 = _prf(tp_v.sum(), fp_v.sum(), fn_v.sum())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    dp_arr = np.zeros(n_boot); dr_arr = np.zeros(n_boot); df_arr = np.zeros(n_boot)
    for b in range(n_boot):
        ii = idx[b]
        pd_b, rd_b, fd_b = _prf(tp_d[ii].sum(), fp_d[ii].sum(), fn_d[ii].sum())
        pv_b, rv_b, fv_b = _prf(tp_v[ii].sum(), fp_v[ii].sum(), fn_v[ii].sum())
        dp_arr[b] = pd_b - pv_b
        dr_arr[b] = rd_b - rv_b
        df_arr[b] = fd_b - fv_b

    return {
        "diff_precision": pd0 - pv0,
        "diff_precision_ci_low": float(np.percentile(dp_arr, 2.5)),
        "diff_precision_ci_high": float(np.percentile(dp_arr, 97.5)),
        "diff_recall": rd0 - rv0,
        "diff_recall_ci_low": float(np.percentile(dr_arr, 2.5)),
        "diff_recall_ci_high": float(np.percentile(dr_arr, 97.5)),
        "diff_f1": fd0 - fv0,
        "diff_f1_ci_low": float(np.percentile(df_arr, 2.5)),
        "diff_f1_ci_high": float(np.percentile(df_arr, 97.5)),
    }


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    scenarios = {s["cenario_id"]: s for s in build_scenarios(config, config["grid"]["tamanhos_n"])}

    df = pd.read_parquet(
        RAW_PARQUET,
        columns=["cenario_id", "semente", "perfil_ruido", "n", "magnitude", "detector", "parametros"],
    )
    ediv_df = df[df["detector"].isin(["EDiv_Means_base", "EDiv_Medians_base"])].copy()

    # Amostra estratificada de 200 séries por (perfil_ruido, n) -> 15 estratos (13 ou 14 por estrato)
    unique_series = ediv_df[["cenario_id", "semente", "perfil_ruido", "n", "magnitude"]].drop_duplicates()
    rng = np.random.default_rng(2026)
    sampled_parts = []
    strata = sorted(unique_series.groupby(["perfil_ruido", "n"]).groups.keys())
    base_k, rem = divmod(200, len(strata))
    for i, (perfil, n_val) in enumerate(strata):
        take = base_k + (1 if i < rem else 0)
        grp = unique_series[(unique_series["perfil_ruido"] == perfil) & (unique_series["n"] == n_val)]
        # Garantir presença de estacionárias e com regressão
        stat = grp[grp["magnitude"] == 0.0]
        reg = grp[grp["magnitude"] > 0.0]
        n_stat = min(len(stat), max(2, take // 5))
        n_reg = take - n_stat
        s_stat = stat.sample(n=n_stat, random_state=int(rng.integers(0, 10**6)))
        s_reg = reg.sample(n=n_reg, random_state=int(rng.integers(0, 10**6)))
        sampled_parts.extend([s_stat, s_reg])

    sampled = pd.concat(sampled_parts, ignore_index=True)
    assert len(sampled) == 200

    # Indexar division_history do Parquet
    hist_lookup: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    merged = ediv_df.merge(sampled[["cenario_id", "semente"]], on=["cenario_id", "semente"])
    for _, r in merged.iterrows():
        hist_lookup[(r["cenario_id"], int(r["semente"]), r["detector"])] = json.loads(r["parametros"])["division_history"]

    payloads = []
    for _, r in sampled.iterrows():
        perfil, n_str = r["perfil_ruido"], str(int(r["n"]))
        payloads.append({
            "scenario": scenarios[r["cenario_id"]],
            "seed_idx": int(r["semente"]),
            "alpha_cal_means": float(manifest["EDiv_Means"][perfil][n_str]),
            "alpha_cal_med": float(manifest["EDiv_Medians"][perfil][n_str]),
        })

    workers = max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        results = list(ex.map(_worker_task, payloads))

    rows = []
    for (perfil, n_val), _ in sampled.groupby(["perfil_ruido", "n"]):
        sub_res = [x for x in results if x["perfil_ruido"] == perfil and x["n"] == int(n_val)]
        for det_base, det_short, cal_key in (
            ("EDiv_Means_base", "EDiv_Means", "EDiv_Means"),
            ("EDiv_Medians_base", "EDiv_Medians", "EDiv_Medians"),
        ):
            a_cal = float(manifest[cal_key][perfil][str(int(n_val))])
            for regime, alpha_val in (("padrao", 0.05), ("calibrado", a_cal)):
                reg_name = f"{det_short}_{regime}"
                gts, dirs, ders = [], [], []
                exact_matches = 0
                for x in sub_res:
                    d_list = x["direct"][reg_name]
                    hist = hist_lookup[(x["cenario_id"], x["semente"], det_base)]
                    v_list = filter_edivisive_by_alpha(hist, alpha_val)
                    if d_list == v_list:
                        exact_matches += 1
                    gts.append(x["ground_truth"])
                    dirs.append(d_list)
                    ders.append(v_list)
                ci = _paired_diff_bootstrap_ci(gts, dirs, ders, k_tol=5)
                rows.append({
                    "perfil_ruido": perfil,
                    "n": int(n_val),
                    "detector": reg_name,
                    "alpha": alpha_val,
                    "n_series": len(sub_res),
                    "exact_matches": exact_matches,
                    "frac_identical": exact_matches / len(sub_res),
                    **ci,
                })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Salvo {OUT_CSV} ({len(out_df)} linhas)")
    print(f"Total identicos: {out_df['exact_matches'].sum()}/{out_df['n_series'].sum()} ({100*out_df['exact_matches'].sum()/out_df['n_series'].sum():.2f}%)")


if __name__ == "__main__":
    main()
