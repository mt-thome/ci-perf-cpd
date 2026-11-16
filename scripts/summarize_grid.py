"""Fase 4b: Derivação do E-Divisive, tabelas finais (Item C) e auditoria de anomalias (Item D).

Uso:
    .venv/bin/python scripts/summarize_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from cpd.detectors import filter_edivisive_by_alpha
from cpd.evaluation.metrics import evaluate_single_series
from cpd.experiments.calibration import poisson_rate_ci_95

REPO_DIR = Path(__file__).resolve().parent.parent
RAW_PARQUET = REPO_DIR / "results/raw_detections.parquet"
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
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


def _match_with_delays(gt: list[int], det: list[int], low_tol: int, high_tol: int) -> tuple[int, int, int, list[int]]:
    """Matching 1-para-1 guloso na janela [gt + low_tol, gt + high_tol] ordenado por |det - gt|."""
    pairs: list[tuple[int, int, int]] = []
    for gi, g in enumerate(gt):
        for di, d in enumerate(det):
            diff = d - g
            if low_tol <= diff <= high_tol:
                pairs.append((abs(diff), gi, di))
    pairs.sort()
    used_g: set[int] = set()
    used_d: set[int] = set()
    delays: list[int] = []
    for _, gi, di in pairs:
        if gi not in used_g and di not in used_d:
            used_g.add(gi)
            used_d.add(di)
            delays.append(int(det[di] - gt[gi]))
    tp = len(delays)
    fp = len(det) - tp
    fn = len(gt) - tp
    return tp, fp, fn, delays


def _delay_stats(delays: list[int]) -> dict[str, float]:
    if not delays:
        return {"delay_n": 0, "delay_mean": np.nan, "delay_median": np.nan, "delay_q25": np.nan, "delay_q75": np.nan}
    arr = np.asarray(delays, dtype=np.float64)
    return {
        "delay_n": int(len(arr)),
        "delay_mean": float(arr.mean()),
        "delay_median": float(np.median(arr)),
        "delay_q25": float(np.percentile(arr, 25)),
        "delay_q75": float(np.percentile(arr, 75)),
    }


def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj.tolist()


def _wilcoxon_pair(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    diff = x - y
    nz = diff[diff != 0.0]
    n_nz = len(nz)
    if n_nz == 0:
        return {"n_pairs": len(x), "n_nonzero": 0, "mean_diff": 0.0, "stat": 0.0, "p_value": 1.0, "rank_biserial": 0.0}
    res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    # Rank-biserial correlation: (W+ - W-) / (n_nz*(n_nz+1)/2)
    ranks = pd.Series(np.abs(nz)).rank(method="average").to_numpy()
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    total_w = w_plus + w_minus
    r_rb = (w_plus - w_minus) / total_w if total_w > 0 else 0.0
    return {
        "n_pairs": int(len(x)),
        "n_nonzero": int(n_nz),
        "mean_diff": float(diff.mean()),
        "stat": float(res.statistic),
        "p_value": float(res.pvalue),
        "rank_biserial": float(r_rb),
    }


def _build_derived_dataframe(df: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    """Exclui linhas inválidas (PELT_L1_gauss), exclui EDiv_*_base e deriva padrao/calibrado."""
    if "valido" in df.columns:
        df = df[df["valido"].fillna(True)].copy()
    df = df[df["detector"] != "PELT_L1_gauss"].copy()

    non_ediv = df[~df["detector"].isin(["EDiv_Means_base", "EDiv_Medians_base"])].copy()
    non_ediv["derivado"] = False

    ediv_base = df[df["detector"].isin(["EDiv_Means_base", "EDiv_Medians_base"])]
    details = manifest["detalhes_por_alvo"]

    derived_rows: list[dict[str, Any]] = []
    for row in ediv_base.to_dict(orient="records"):
        hist = json.loads(row["parametros"])["division_history"]
        haz_all = json.loads(row["hazard_levels_est"])
        perfil = row["perfil_ruido"]
        n_str = str(int(row["n"]))
        base_det = row["detector"]
        if base_det == "EDiv_Means_base":
            prefix, cal_k = "EDiv_Means", "EDiv_Means"
        else:
            prefix, cal_k = "EDiv_Medians", "EDiv_Medians"

        a_02 = float(details[perfil][n_str]["0.2"][cal_k]["param"])
        a_10 = float(details[perfil][n_str]["1.0"][cal_k]["param"])

        variants = [
            (f"{prefix}_padrao", 0.05),
            (f"{prefix}_calibrado", a_02),
            (f"{prefix}_calibrado_far1", a_10),
        ]
        for new_name, alpha_val in variants:
            idxs = filter_edivisive_by_alpha(hist, alpha_val)
            idx_set = set(idxs)
            haz_sub = [h for h in haz_all if int(h["detected_cp"]) in idx_set]
            new_row = dict(row)
            new_row["detector"] = new_name
            new_row["derivado"] = True
            new_row["indices_detectados"] = idxs
            new_row["hazard_levels_est"] = json.dumps(haz_sub)
            derived_rows.append(new_row)

    out = pd.concat([non_ediv, pd.DataFrame(derived_rows)], ignore_index=True)
    out["k_true"] = out["ground_truth"].apply(len)
    return out


def _audit_anomalies(df: pd.DataFrame, far_df: pd.DataFrame, cell_df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Item D: Lista objetiva de anomalias (sem interpretação)."""
    anomalies: list[dict[str, Any]] = []

    # 1. Detectores que retornam vazio [] em todas as séries de alguma célula (com regressão)
    reg = df[df["magnitude"] > 0.0]
    for (det, perf, n_val, mag, kt), grp in reg.groupby(["detector", "perfil_ruido", "n", "magnitude", "k_true"]):
        if all(len(x) == 0 for x in grp["indices_detectados"]):
            anomalies.append({
                "categoria": "sempre_vazio",
                "detector": det, "perfil_ruido": perf, "n": int(n_val), "magnitude": float(mag), "k_true": int(kt),
                "detalhe": f"0 detecções em todas as {len(grp)} séries da célula",
            })

    # 2. Células com F1(k=5) muito baixo (< 0.50) em gaussian 5sigma
    g5 = cell_df[(cell_df["perfil_ruido"] == "gaussian") & (cell_df["magnitude"] == 5.0) & (cell_df["k_tol"] == 5) & (cell_df["f1"] < 0.50)]
    for _, r in g5.iterrows():
        anomalies.append({
            "categoria": "f1_baixo_gaussiano_5sigma",
            "detector": r["detector"], "perfil_ruido": "gaussian", "n": int(r["n"]), "magnitude": 5.0, "k_true": int(r["k_true"]),
            "detalhe": f"F1(k=5)={r['f1']:.3f} (P={r['precision']:.3f}, R={r['recall']:.3f})",
        })

    # 3. Índices detectados fora de [0, n-1] ou duplicados
    oob_cnt = 0
    dup_cnt = 0
    for row in df.to_dict(orient="records"):
        idxs = [int(x) for x in row["indices_detectados"]]
        n_val = int(row["n"])
        if any(x < 0 or x >= n_val for x in idxs):
            oob_cnt += 1
        if len(idxs) != len(set(idxs)):
            dup_cnt += 1
    anomalies.append({
        "categoria": "indices_invalidos_ou_duplicados",
        "detector": "TODOS", "perfil_ruido": "TODOS", "n": 0, "magnitude": np.nan, "k_true": -1,
        "detalhe": f"fora_de_intervalo={oob_cnt}, duplicados_na_serie={dup_cnt}",
    })

    # 4. Diferenças grandes entre FAR do grid e FAR da validação (ICs 95% disjuntos ou |diff| > 0.5)
    # Para regimes _gauss em perfil != gaussian, o grid avalia o perfil X enquanto a validação é no gaussiano (esperado, não anomalia).
    for _, r in far_df.iterrows():
        if pd.isna(r["val_far_1k"]):
            continue
        if str(r["detector"]).endswith("_gauss") and r["perfil_ruido"] != "gaussian":
            continue
        diff = abs(float(r["far_1k"]) - float(r["val_far_1k"]))
        disjoint = (float(r["far_1k_ci95_low"]) > float(r["val_far_1k_ci95_high"])) or (float(r["val_far_1k_ci95_low"]) > float(r["far_1k_ci95_high"]))
        if disjoint or diff > 0.5:
            anomalies.append({
                "categoria": "divergencia_far_grid_vs_validacao",
                "detector": r["detector"], "perfil_ruido": r["perfil_ruido"], "n": int(r["n"]), "magnitude": 0.0, "k_true": 0,
                "detalhe": f"grid_far1k={r['far_1k']:.3f} vs val_far1k={r['val_far_1k']:.3f} (disjoint_ci={disjoint})",
            })

    # 5. Células com saturação ou falha de validação no manifesto (alvo 0.2 e 1.0)
    for tgt_key in [f"{t:.1f}" for t in manifest["_meta"]["alvos_far_1k"]]:
        for perfil, n_dict in manifest["detalhes_por_alvo"].items():
            for n_str, t_dict in n_dict.items():
                for det_k, rec in t_dict[tgt_key].items():
                    flags = [f for f in ("saturado_inferior", "saturado_superior", "validacao_falhou") if rec.get(f)]
                    if flags:
                        anomalies.append({
                            "categoria": f"flag_calibracao_alvo_{tgt_key}",
                            "detector": det_k, "perfil_ruido": perfil, "n": int(n_str), "magnitude": 0.0, "k_true": 0,
                            "detalhe": ", ".join(flags) + f" (param={rec['param']}, val_far1k={rec['val']['far_1k']:.3f})",
                        })

    # 6. Falsos alarmes concentrados por decil de posição nas séries estacionárias
    stat = df[df["magnitude"] == 0.0]
    for det, grp in stat.groupby("detector"):
        dec_counts = np.zeros(10, dtype=int)
        for row in grp.to_dict(orient="records"):
            n_val = int(row["n"])
            for cp in row["indices_detectados"]:
                d = min(9, int((int(cp) / n_val) * 10))
                dec_counts[d] += 1
        tot = int(dec_counts.sum())
        if tot >= 20:
            max_share = float(dec_counts.max() / tot)
            if max_share > 0.25:  # >2.5x esperado uniforme (0.10)
                top_dec = int(np.argmax(dec_counts))
                anomalies.append({
                    "categoria": "concentracao_espacial_falsos_alarmes",
                    "detector": det, "perfil_ruido": "TODOS", "n": 0, "magnitude": 0.0, "k_true": 0,
                    "detalhe": f"decil_{top_dec} concentra {dec_counts[top_dec]}/{tot} ({100*max_share:.1f}%)",
                })

    anom_df = pd.DataFrame(anomalies)
    anom_df.to_csv(TABLES_DIR / "anomalies_phase4b.csv", index=False)

    md_lines = ["# Auditoria de Anomalias (Fase 4b — Sem Interpretação)", ""]
    for cat, grp in anom_df.groupby("categoria", sort=False):
        md_lines.append(f"## {cat} ({len(grp)})")
        md_lines.append("")
        for _, r in grp.iterrows():
            md_lines.append(f"- `{r['detector']}` | `{r['perfil_ruido']}` | n={r['n']} | m={r['magnitude']} | k_true={r['k_true']}: {r['detalhe']}")
        md_lines.append("")
    (TABLES_DIR / "anomalies_phase4b.md").write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_parquet(RAW_PARQUET)
    manifest = json.loads(MANIFEST_PATH.read_text())

    df = _build_derived_dataframe(raw_df, manifest)
    det_deriv_map = df.groupby("detector")["derivado"].first().to_dict()

    # 1. Avaliação por série para k in {1, 3, 5, 10} (simétrico) e [-1, +k] (assimétrico)
    reg = df[df["magnitude"] > 0.0].copy()
    sym_records: list[dict[str, Any]] = []
    asym_records: list[dict[str, Any]] = []
    haz_records: list[dict[str, Any]] = []

    for row in reg.to_dict(orient="records"):
        gt = [int(x) for x in row["ground_truth"]]
        det = [int(x) for x in row["indices_detectados"]]
        base_meta = {
            "cenario_id": row["cenario_id"],
            "semente": int(row["semente"]),
            "detector": row["detector"],
            "derivado": bool(row["derivado"]),
            "perfil_ruido": row["perfil_ruido"],
            "n": int(row["n"]),
            "magnitude": float(row["magnitude"]),
            "k_true": int(row["k_true"]),
        }
        for k in (1, 3, 5, 10):
            tp, fp, fn, delays = _match_with_delays(gt, det, -k, k)
            sym_records.append({**base_meta, "k_tol": k, "tp": tp, "fp": fp, "fn": fn, "delays": delays})
        for k in (3, 5, 10):
            tp, fp, fn, delays = _match_with_delays(gt, det, -1, k)
            asym_records.append({**base_meta, "k_tol": k, "tp": tp, "fp": fp, "fn": fn, "delays": delays})

        for h in json.loads(row["hazard_levels_est"]):
            haz_records.append({
                "detector": row["detector"],
                "derivado": bool(row["derivado"]),
                "perfil_ruido": row["perfil_ruido"],
                "n": int(row["n"]),
                "magnitude": float(row["magnitude"]),
                "err": float(h["hazard_level_est"]) - float(h["hazard_level_true"]),
                "abs_err": float(h["hazard_level_abs_err"]),
            })

    sym_df = pd.DataFrame(sym_records)
    asym_df = pd.DataFrame(asym_records)

    # 1a. metrics_by_cell.csv
    cell_rows = []
    for keys, grp in sym_df.groupby(["detector", "perfil_ruido", "n", "magnitude", "k_true", "k_tol"]):
        det_name, perf, nv, mag, kt, ktol = keys
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        all_delays = [d for dl in grp["delays"] for d in dl]
        cell_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "perfil_ruido": perf, "n": int(nv), "magnitude": float(mag), "k_true": int(kt), "k_tol": int(ktol),
            "num_series": len(grp),
            "tp": int(grp["tp"].sum()), "fp": int(grp["fp"].sum()), "fn": int(grp["fn"].sum()),
            **boot, **_delay_stats(all_delays),
        })
    cell_df = pd.DataFrame(cell_rows)
    cell_df.to_csv(TABLES_DIR / "metrics_by_cell.csv", index=False)

    # 1b. metrics_by_detector_profile_n_k.csv e metrics_overall_by_detector_k.csv
    dpnk_rows = []
    for (det_name, perf, nv, ktol), grp in sym_df.groupby(["detector", "perfil_ruido", "n", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        all_delays = [d for dl in grp["delays"] for d in dl]
        dpnk_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "perfil_ruido": perf, "n": int(nv), "k_tol": int(ktol),
            "num_series": len(grp),
            "tp": int(grp["tp"].sum()), "fp": int(grp["fp"].sum()), "fn": int(grp["fn"].sum()),
            **boot, **_delay_stats(all_delays),
        })
    pd.DataFrame(dpnk_rows).to_csv(TABLES_DIR / "metrics_by_detector_profile_n_k.csv", index=False)

    overall_rows = []
    for (det_name, ktol), grp in sym_df.groupby(["detector", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        all_delays = [d for dl in grp["delays"] for d in dl]
        overall_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "k_tol": int(ktol), "num_series": len(grp),
            "tp": int(grp["tp"].sum()), "fp": int(grp["fp"].sum()), "fn": int(grp["fn"].sum()),
            **boot, **_delay_stats(all_delays),
        })
    pd.DataFrame(overall_rows).to_csv(TABLES_DIR / "metrics_overall_by_detector_k.csv", index=False)

    # 1c. Janela assimétrica [-1, +k]
    asym_rows = []
    for (det_name, perf, nv, mag, kt, ktol), grp in asym_df.groupby(["detector", "perfil_ruido", "n", "magnitude", "k_true", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        asym_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "perfil_ruido": perf, "n": int(nv), "magnitude": float(mag), "k_true": int(kt), "k_tol": int(ktol),
            "num_series": len(grp),
            "tp": int(grp["tp"].sum()), "fp": int(grp["fp"].sum()), "fn": int(grp["fn"].sum()),
            **boot,
        })
    pd.DataFrame(asym_rows).to_csv(TABLES_DIR / "metrics_asymmetric_window.csv", index=False)

    asym_ov_rows = []
    for (det_name, ktol), grp in asym_df.groupby(["detector", "k_tol"]):
        boot = _bootstrap_micro_metrics(grp["tp"].to_numpy(), grp["fp"].to_numpy(), grp["fn"].to_numpy())
        asym_ov_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "k_tol": int(ktol), "num_series": len(grp), **boot,
        })
    pd.DataFrame(asym_ov_rows).to_csv(TABLES_DIR / "metrics_asymmetric_overall.csv", index=False)

    # 2. FAR nas séries estacionárias (magnitude == 0) + calibração/validação
    det_to_cal_tgt = {
        "CUSUM_gauss": ("CUSUM", "0.2", "gaussian"), "CUSUM_perfil": ("CUSUM", "0.2", None),
        "PELT_L2_gauss": ("PELT_L2", "0.2", "gaussian"), "PELT_L2_perfil": ("PELT_L2", "0.2", None),
        "PELT_L1_gauss": ("PELT_L1", "0.2", "gaussian"), "PELT_L1_perfil": ("PELT_L1", "0.2", None),
        "Binseg_L2_gauss": ("BINSEG_L2", "0.2", "gaussian"), "Binseg_L2_perfil": ("BINSEG_L2", "0.2", None),
        "EDiv_Means_calibrado": ("EDiv_Means", "0.2", None), "EDiv_Means_calibrado_far1": ("EDiv_Means", "1.0", None),
        "EDiv_Medians_calibrado": ("EDiv_Medians", "0.2", None), "EDiv_Medians_calibrado_far1": ("EDiv_Medians", "1.0", None),
    }
    stat = df[df["magnitude"] == 0.0]
    far_rows = []
    for (det_name, perf, nv), grp in stat.groupby(["detector", "perfil_ruido", "n"]):
        alarms = [len(x) for x in grp["indices_detectados"]]
        total_al = int(sum(alarms))
        exp_commits = len(grp) * int(nv)
        f1k = (total_al / exp_commits) * 1000.0
        fser = sum(1 for a in alarms if a > 0) / len(grp)
        ci_l, ci_h = poisson_rate_ci_95(total_al, exp_commits)

        cal_info = det_to_cal_tgt.get(det_name)
        val_f1k = val_l = val_h = val_fser = cal_f1k = cal_fser = np.nan
        flags_str = ""
        if cal_info is not None:
            cal_k, tgt_k, forced_perf = cal_info
            lookup_perf = forced_perf or perf
            rec = manifest["detalhes_por_alvo"][lookup_perf][str(int(nv))][tgt_k][cal_k]
            val_f1k = float(rec["val"]["far_1k"])
            val_l, val_h = float(rec["val"]["ci95_far_1k"][0]), float(rec["val"]["ci95_far_1k"][1])
            val_fser = float(rec["val"]["far_series"])
            cal_f1k = float(rec["calib"]["far_1k"])
            cal_fser = float(rec["calib"]["far_series"])
            flags_str = ",".join([f for f in ("saturado_inferior", "saturado_superior", "validacao_falhou") if rec.get(f)])

        far_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "perfil_ruido": perf, "n": int(nv), "num_series": len(grp),
            "total_alarms": total_al, "far_series": float(fser),
            "far_1k": float(f1k), "far_1k_ci95_low": float(ci_l), "far_1k_ci95_high": float(ci_h),
            "calib_far_1k": cal_f1k, "calib_far_series": cal_fser,
            "val_far_1k": val_f1k, "val_far_1k_ci95_low": val_l, "val_far_1k_ci95_high": val_h,
            "val_far_series": val_fser, "calib_flags": flags_str,
        })
    far_df = pd.DataFrame(far_rows)
    far_df.to_csv(TABLES_DIR / "far_by_detector_profile_n.csv", index=False)

    # 3. Erro de estimativa do Hazard Level
    haz_df = pd.DataFrame(haz_records)
    haz_rows = []
    for (det_name, perf, nv, mag), grp in haz_df.groupby(["detector", "perfil_ruido", "n", "magnitude"]):
        haz_rows.append({
            "detector": det_name, "derivado": det_deriv_map[det_name],
            "perfil_ruido": perf, "n": int(nv), "magnitude": float(mag),
            "n_detections": len(grp),
            "mae": float(grp["abs_err"].mean()),
            "median_abs_err": float(grp["abs_err"].median()),
            "rmse": float(np.sqrt((grp["err"] ** 2).mean())),
            "bias": float(grp["err"].mean()),
        })
    pd.DataFrame(haz_rows).to_csv(TABLES_DIR / "hazard_level_error.csv", index=False)

    # 4. Comparações pareadas por semente (Wilcoxon + rank-biserial + Holm por família) em F1(k=5)
    sym5 = sym_df[sym_df["k_tol"] == 5].copy()
    denom = 2 * sym5["tp"] + sym5["fp"] + sym5["fn"]
    sym5["f1_series"] = np.where(denom > 0, 2.0 * sym5["tp"] / denom, 0.0)
    piv = sym5.pivot(index=["perfil_ruido", "cenario_id", "semente"], columns="detector", values="f1_series")

    profiles = sorted(sym5["perfil_ruido"].unique().tolist())
    wil_rows: list[dict[str, Any]] = []

    # Família 1: Means vs Medians em cada perfil
    for perf in profiles:
        sub = piv.loc[perf]
        for reg_name in ("padrao", "calibrado"):
            d1, d2 = f"EDiv_Means_{reg_name}", f"EDiv_Medians_{reg_name}"
            st = _wilcoxon_pair(sub[d1].to_numpy(), sub[d2].to_numpy())
            wil_rows.append({"familia": "means_vs_medians", "perfil_ruido": perf, "detector_a": d1, "detector_b": d2, **st})

    # Família 2: calibrado-por-perfil vs calibrado-em-gaussiano (sem PELT_L1_gauss, descartada por ADR-019)
    for perf in profiles:
        sub = piv.loc[perf]
        for base in ("CUSUM", "PELT_L2", "Binseg_L2"):
            d1, d2 = f"{base}_perfil", f"{base}_gauss"
            st = _wilcoxon_pair(sub[d1].to_numpy(), sub[d2].to_numpy())
            wil_rows.append({"familia": "perfil_vs_gauss", "perfil_ruido": perf, "detector_a": d1, "detector_b": d2, **st})

    # Família 3: CUSUM vs PELT_L2 vs Binseg_L2 vs E-Div-Means (padrao e calibrado) por perfil
    fam3_dets = ["CUSUM_perfil", "PELT_L2_perfil", "Binseg_L2_perfil", "EDiv_Means_padrao", "EDiv_Means_calibrado"]
    for perf in profiles:
        sub = piv.loc[perf]
        for i in range(len(fam3_dets)):
            for j in range(i + 1, len(fam3_dets)):
                d1, d2 = fam3_dets[i], fam3_dets[j]
                st = _wilcoxon_pair(sub[d1].to_numpy(), sub[d2].to_numpy())
                wil_rows.append({"familia": "comparacao_detectores", "perfil_ruido": perf, "detector_a": d1, "detector_b": d2, **st})

    wil_df = pd.DataFrame(wil_rows)
    wil_df["p_value_holm"] = 1.0
    for fam, idx_grp in wil_df.groupby("familia").groups.items():
        pvals = wil_df.loc[idx_grp, "p_value"].tolist()
        wil_df.loc[idx_grp, "p_value_holm"] = _holm_adjust(pvals)
    wil_df.to_csv(TABLES_DIR / "paired_comparisons_wilcoxon.csv", index=False)

    # 5. Auditoria de anomalias (Item D)
    _audit_anomalies(df, far_df, cell_df, manifest)
    print("Fase 4b (tabelas e auditoria) concluída em:", TABLES_DIR)


if __name__ == "__main__":
    main()
