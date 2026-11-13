"""Gera tabelas agregadas (micro-F1, Precision, Recall com IC bootstrap e FAR) do grid final.

Uso:
    .venv/bin/python scripts/summarize_grid.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from cpd.evaluation.metrics import evaluate_single_series
from cpd.experiments.calibration import poisson_rate_ci_95

REPO_DIR = Path(__file__).resolve().parent.parent
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
TABLES_DIR = REPO_DIR / "results/tables"


def _bootstrap_micro_metrics(
    tp_arr: np.ndarray,
    fp_arr: np.ndarray,
    fn_arr: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    n = len(tp_arr)
    tp_s, fp_s, fn_s = float(tp_arr.sum()), float(fp_arr.sum()), float(fn_arr.sum())
    prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
    rec = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    btp = tp_arr[idx].sum(axis=1).astype(np.float64)
    bfp = fp_arr[idx].sum(axis=1).astype(np.float64)
    bfn = fn_arr[idx].sum(axis=1).astype(np.float64)

    bp = np.divide(btp, btp + bfp, out=np.zeros_like(btp), where=(btp + bfp) > 0)
    br = np.divide(btp, btp + bfn, out=np.zeros_like(btp), where=(btp + bfn) > 0)
    bf1 = np.divide(2 * bp * br, bp + br, out=np.zeros_like(bp), where=(bp + br) > 0)

    return {
        "precision": prec,
        "precision_ci_low": float(np.percentile(bp, 2.5)),
        "precision_ci_high": float(np.percentile(bp, 97.5)),
        "recall": rec,
        "recall_ci_low": float(np.percentile(br, 2.5)),
        "recall_ci_high": float(np.percentile(br, 97.5)),
        "f1": f1,
        "f1_ci_low": float(np.percentile(bf1, 2.5)),
        "f1_ci_high": float(np.percentile(bf1, 97.5)),
    }


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(RAW_PARQUET)

    # 1. Tabelas de F1 / Precision / Recall (séries com regressão: magnitude > 0)
    reg = df[df["magnitude"] > 0.0].copy()
    tolerances = (1, 3, 5, 10)
    eval_records = []
    for _, row in reg.iterrows():
        gt = list(row["ground_truth"])
        det = list(row["indices_detectados"])
        for k in tolerances:
            res = evaluate_single_series(gt, det, k=k)
            eval_records.append({
                "detector": row["detector"],
                "perfil_ruido": row["perfil_ruido"],
                "n": int(row["n"]),
                "magnitude": float(row["magnitude"]),
                "k_tol": k,
                "tp": res.tp,
                "fp": res.fp,
                "fn": res.fn,
            })
    eval_df = pd.DataFrame(eval_records)

    summary_rows = []
    for (det, perfil, n_val, k_tol), grp in eval_df.groupby(["detector", "perfil_ruido", "n", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        summary_rows.append({
            "detector": det,
            "perfil_ruido": perfil,
            "n": int(n_val),
            "k_tol": int(k_tol),
            "num_series": len(grp),
            "tp": int(grp["tp"].sum()),
            "fp": int(grp["fp"].sum()),
            "fn": int(grp["fn"].sum()),
            **boot,
        })
    pd.DataFrame(summary_rows).to_csv(TABLES_DIR / "metrics_by_detector_profile_n_k.csv", index=False)

    overall_rows = []
    for (det, k_tol), grp in eval_df.groupby(["detector", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        overall_rows.append({
            "detector": det,
            "k_tol": int(k_tol),
            "num_series": len(grp),
            "tp": int(grp["tp"].sum()),
            "fp": int(grp["fp"].sum()),
            "fn": int(grp["fn"].sum()),
            **boot,
        })
    pd.DataFrame(overall_rows).to_csv(TABLES_DIR / "metrics_overall_by_detector_k.csv", index=False)

    # 2. Tabelas FAR (séries estacionárias: magnitude == 0)
    stat = df[df["magnitude"] == 0.0]
    far_rows = []
    for (det, perfil, n_val), grp in stat.groupby(["detector", "perfil_ruido", "n"]):
        alarms = [len(x) for x in grp["indices_detectados"]]
        total_al = int(sum(alarms))
        exp_commits = len(grp) * int(n_val)
        f1k = (total_al / exp_commits) * 1000.0
        fser = sum(1 for a in alarms if a > 0) / len(grp)
        ci_l, ci_h = poisson_rate_ci_95(total_al, exp_commits)
        far_rows.append({
            "detector": det,
            "perfil_ruido": perfil,
            "n": int(n_val),
            "num_series": len(grp),
            "total_alarms": total_al,
            "far_series": float(fser),
            "far_1k": float(f1k),
            "far_1k_ci95_low": float(ci_l),
            "far_1k_ci95_high": float(ci_h),
        })
    pd.DataFrame(far_rows).to_csv(TABLES_DIR / "far_by_detector_profile_n.csv", index=False)


if __name__ == "__main__":
    main()
