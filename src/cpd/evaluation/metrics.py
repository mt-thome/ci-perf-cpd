"""Módulo de avaliação, matching temporal ótimo e métricas de CPD para CI.

Implementa matching ótimo bipartido com custo finito (Hungarian/Munkres via SciPy),
cálculo de atraso de detecção, métricas por série e agregadas (Micro e Macro),
análise de séries estacionárias (FAR), curvas Precision-Recall, Bootstrap e teste pareado de Wilcoxon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


@dataclass(frozen=True)
class SeriesEvaluationResult:
    """Resultado da avaliação de uma única série temporal para um dado k de tolerância.

    Parameters
    ----------
    tp : int
        True Positives (pontos verdadeiros casados com uma detecção).
    fp : int
        False Positives (detecções que não casaram com nenhum ground truth).
    fn : int
        False Negatives (pontos verdadeiros não detectados).
    precision : float
        Precisão TP / (TP + FP). NaN se TP + FP == 0.
    recall : float
        Recall TP / (TP + FN). NaN se TP + FN == 0.
    f1 : float
        F1-score 2*P*R / (P + R). NaN se ambos forem NaN, ou 0.0 se P+R == 0.
    delays : list[int]
        Vetor com o atraso temporal (índice_detectado - índice_verdadeiro) para cada match.
    matches : list[tuple[int, int]]
        Lista de pares casados (índice_ground_truth, índice_detectado).
    mean_delay : float
        Média dos atrasos (NaN se sem matches).
    median_delay : float
        Mediana dos atrasos (NaN se sem matches).
    """

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    delays: list[int] = field(default_factory=list)
    matches: list[tuple[int, int]] = field(default_factory=list)
    mean_delay: float = float("nan")
    median_delay: float = float("nan")


def match_optimal_bipartite(
    ground_truth: Sequence[int],
    detected: Sequence[int],
    k: int | None = None,
    k_before: int | None = None,
    k_after: int | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Realiza emparelhamento 1-para-1 ótimo minimizando a distância temporal global.

    Utiliza uma penalidade M finita na matriz de custo para assegurar que:
    1. A cardinalidade dos matches (True Positives) seja maximizada antes de minimizar a distância;
    2. scipy.optimize.linear_sum_assignment execute estavelmente sem valores infinitos.

    Parameters
    ----------
    ground_truth : Sequence[int]
        Lista de índices verdadeiros dos change points.
    detected : Sequence[int]
        Lista de índices detectados pelo algoritmo.
    k : int | None
        Tolerância simétrica [-k, +k].
    k_before : int | None
        Tolerância para detecções antes do commit verdadeiro [g - k_before, g + k_after].
    k_after : int | None
        Tolerância para detecções depois do commit verdadeiro.

    Returns
    -------
    matches : list[tuple[int, int]]
        Pares casados (índice_gt, índice_det).
    delays : list[int]
        Atrasos temporais correspondentes (det - gt).
    """
    gt_list = sorted(list(ground_truth))
    det_list = sorted(list(detected))

    n_gt = len(gt_list)
    n_det = len(det_list)

    if n_gt == 0 or n_det == 0:
        return [], []

    # Configuração da janela temporal
    if k is not None:
        kb = int(k)
        ka = int(k)
    else:
        kb = int(k_before) if k_before is not None else 0
        ka = int(k_after) if k_after is not None else 0

    k_max = max(kb, ka)
    # M finito estritamente superior ao custo de qualquer combinação admissível
    m_penalty = float(max(n_gt, n_det) * (k_max + 1) + 1000.0)

    # Construção da matriz de custo retangular |G| x |D|
    cost_matrix = np.full((n_gt, n_det), m_penalty, dtype=np.float64)

    for i, g in enumerate(gt_list):
        for j, d in enumerate(det_list):
            delay = d - g
            # Condição de aceitação na janela temporal: -kb <= delay <= ka
            if -kb <= delay <= ka:
                cost_matrix[i, j] = abs(delay)

    # Resolução via algoritmo húngaro
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches: list[tuple[int, int]] = []
    delays: list[int] = []

    for r, c in zip(row_ind, col_ind):
        # Apenas pares com custo finito estrito menor que M são matches válidos
        if cost_matrix[r, c] < m_penalty:
            g_idx = gt_list[r]
            d_idx = det_list[c]
            matches.append((g_idx, d_idx))
            delays.append(d_idx - g_idx)

    # Ordena matches cronologicamente pelo ground truth
    matches = sorted(matches, key=lambda x: x[0])
    delays = [d - g for g, d in matches]

    return matches, delays


def match_greedy_auxiliary(
    ground_truth: Sequence[int],
    detected: Sequence[int],
    k: int = 3,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Matching guloso auxiliar mantido estritamente para comparação com o ótimo."""
    gt_list = sorted(list(ground_truth))
    det_list = sorted(list(detected))

    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[tuple[int, int]] = []
    delays: list[int] = []

    # Ordena todos os pares potenciais pela distância temporal
    pairs = []
    for i, g in enumerate(gt_list):
        for j, d in enumerate(det_list):
            dist = abs(d - g)
            if dist <= k:
                pairs.append((dist, i, j, g, d))

    pairs.sort(key=lambda x: x[0])

    for dist, i, j, g, d in pairs:
        if i not in used_gt and j not in used_det:
            used_gt.add(i)
            used_det.add(j)
            matches.append((g, d))
            delays.append(d - g)

    matches.sort(key=lambda x: x[0])
    delays = [d - g for g, d in matches]
    return matches, delays


def evaluate_single_series(
    ground_truth: Sequence[int],
    detected: Sequence[int],
    k: int | None = 3,
    k_before: int | None = None,
    k_after: int | None = None,
) -> SeriesEvaluationResult:
    """Avalia métricas TP, FP, FN, Precision, Recall, F1 e atrasos para uma série."""
    matches, delays = match_optimal_bipartite(
        ground_truth=ground_truth,
        detected=detected,
        k=k,
        k_before=k_before,
        k_after=k_after,
    )

    n_gt = len(ground_truth)
    n_det = len(detected)

    tp = len(matches)
    fp = n_det - tp
    fn = n_gt - tp

    # Tratamento formal de casos degenerados
    if n_det == 0 and n_gt == 0:
        precision = float("nan")
        recall = float("nan")
        f1 = float("nan")
    elif n_det == 0 and n_gt > 0:
        precision = float("nan")
        recall = 0.0
        f1 = 0.0
    elif n_det > 0 and n_gt == 0:
        precision = 0.0
        recall = float("nan")
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if np.isnan(precision) or np.isnan(recall) or (precision + recall == 0.0):
            f1 = 0.0
        else:
            f1 = (2.0 * precision * recall) / (precision + recall)

    mean_d = float(np.mean(delays)) if len(delays) > 0 else float("nan")
    med_d = float(np.median(delays)) if len(delays) > 0 else float("nan")

    return SeriesEvaluationResult(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        delays=delays,
        matches=matches,
        mean_delay=mean_d,
        median_delay=med_d,
    )


def evaluate_stationary_series(
    detected_list: Sequence[Sequence[int]],
    series_lengths: Sequence[int],
) -> dict[str, float]:
    """Calcula taxas de falso alarme em séries sem regressão (k_true = 0).

    Returns
    -------
    dict com:
        - "far_series": proporção de séries com pelo menos 1 alarme espúrio.
        - "far_1k_commits": número de alarmes falsos normalizado por 1000 commits.
        - "total_false_alarms": contagem absoluta de detecções espúrias.
        - "num_series": número total de séries avaliadas.
    """
    r = len(detected_list)
    if r == 0:
        raise ValueError("Lista de detecções vazia.")

    series_with_alarms = sum(1 for det in detected_list if len(det) > 0)
    total_alarms = sum(len(det) for det in detected_list)
    total_commits = sum(series_lengths)

    far_series = series_with_alarms / r
    far_1k = (total_alarms / total_commits) * 1000.0 if total_commits > 0 else 0.0

    return {
        "far_series": float(far_series),
        "far_1k_commits": float(far_1k),
        "total_false_alarms": int(total_alarms),
        "total_commits": int(total_commits),
        "num_series": int(r),
    }


def aggregate_metrics(
    eval_results: Sequence[SeriesEvaluationResult],
) -> dict[str, Any]:
    """Calcula a agregação Micro (primária) e Macro (secundária com contagem de NaNs).

    Parameters
    ----------
    eval_results : Sequence[SeriesEvaluationResult]
        Lista de avaliações individuais por série.
    """
    if len(eval_results) == 0:
        raise ValueError("Lista de resultados de avaliação não pode ser vazia.")

    # 1. Agregação Micro
    sum_tp = sum(r.tp for r in eval_results)
    sum_fp = sum(r.fp for r in eval_results)
    sum_fn = sum(r.fn for r in eval_results)

    micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (2.0 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    # 2. Agregação Macro
    precisions = [r.precision for r in eval_results]
    recalls = [r.recall for r in eval_results]
    f1s = [r.f1 for r in eval_results]

    all_delays: list[int] = []
    for r in eval_results:
        all_delays.extend(r.delays)

    macro_p = float(np.nanmean(precisions)) if not np.all(np.isnan(precisions)) else float("nan")
    macro_r = float(np.nanmean(recalls)) if not np.all(np.isnan(recalls)) else float("nan")
    macro_f1 = float(np.nanmean(f1s)) if not np.all(np.isnan(f1s)) else float("nan")

    return {
        "micro": {
            "tp": sum_tp,
            "fp": sum_fp,
            "fn": sum_fn,
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
        "macro": {
            "precision_mean": macro_p,
            "recall_mean": macro_r,
            "f1_mean": macro_f1,
            "precision_nan_count": int(np.isnan(precisions).sum()),
            "recall_nan_count": int(np.isnan(recalls).sum()),
            "f1_nan_count": int(np.isnan(f1s).sum()),
            "total_series": len(eval_results),
        },
        "delays": {
            "count": len(all_delays),
            "mean": float(np.mean(all_delays)) if len(all_delays) > 0 else float("nan"),
            "median": float(np.median(all_delays)) if len(all_delays) > 0 else float("nan"),
            "std": float(np.std(all_delays)) if len(all_delays) > 0 else float("nan"),
            "values": all_delays,
        },
    }


def bootstrap_series_micro_f1(
    ground_truth_list: Sequence[Sequence[int]],
    detected_list: Sequence[Sequence[int]],
    k: int = 3,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap de séries completas com reposição para intervalo de confiança do Micro-F1.

    Parameters
    ----------
    ground_truth_list : Sequence[Sequence[int]]
        Lista de ground truths para as N séries.
    detected_list : Sequence[Sequence[int]]
        Lista de detecções para as N séries.
    """
    n_series = len(ground_truth_list)
    if n_series != len(detected_list):
        raise ValueError("ground_truth_list e detected_list devem ter o mesmo tamanho.")

    # Pré-calcula TP, FP, FN por série para acelerar o bootstrap
    series_stats: list[tuple[int, int, int]] = []
    for gt, det in zip(ground_truth_list, detected_list):
        res = evaluate_single_series(gt, det, k=k)
        series_stats.append((res.tp, res.fp, res.fn))

    rng = np.random.default_rng(seed)
    boot_f1s = np.zeros(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sample_indices = rng.choice(n_series, size=n_series, replace=True)
        tp_sum = sum(series_stats[i][0] for i in sample_indices)
        fp_sum = sum(series_stats[i][1] for i in sample_indices)
        fn_sum = sum(series_stats[i][2] for i in sample_indices)

        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        boot_f1s[b] = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

    alpha = 1.0 - ci
    low_perc = 100.0 * (alpha / 2.0)
    high_perc = 100.0 * (1.0 - alpha / 2.0)

    return {
        "ci_lower": float(np.percentile(boot_f1s, low_perc)),
        "ci_upper": float(np.percentile(boot_f1s, high_perc)),
        "mean_bootstrap": float(np.mean(boot_f1s)),
        "std_bootstrap": float(np.std(boot_f1s)),
    }


def paired_wilcoxon_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
) -> dict[str, float]:
    """Teste dos postos sinalizados de Wilcoxon com cálculo explícito de W+, W- e tamanho de efeito.

    Parameters
    ----------
    scores_a, scores_b : Sequence[float]
        Métricas pareadas pelas mesmas sementes (ex.: F1 por série).

    Returns
    -------
    dict com:
        - "w_plus": soma dos postos das diferenças positivas.
        - "w_minus": soma dos postos das diferenças negativas.
        - "statistic": min(W+, W-).
        - "rank_biserial_r": tamanho de efeito r = (W+ - W-) / (n_diff*(n_diff+1)/2).
        - "n_diff": número de pares com diferença não nula.
        - "p_value": valor-p bilateral aproximado (assintótico com correção de continuidade).
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    # Descarta pares com NaN
    valid_mask = (~np.isnan(a)) & (~np.isnan(b))
    diffs = a[valid_mask] - b[valid_mask]

    # Descarta diferenças zero
    non_zero_diffs = diffs[diffs != 0.0]
    n_diff = len(non_zero_diffs)

    if n_diff == 0:
        return {
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "rank_biserial_r": 0.0,
            "n_diff": 0,
            "p_value": 1.0,
        }

    ranks = rankdata(np.abs(non_zero_diffs))
    w_plus = float(np.sum(ranks[non_zero_diffs > 0]))
    w_minus = float(np.sum(ranks[non_zero_diffs < 0]))
    w_stat = min(w_plus, w_minus)

    total_rank_sum = n_diff * (n_diff + 1) / 2.0
    r_biserial = (w_plus - w_minus) / total_rank_sum

    # Cálculo do p-valor assintótico normal com correção de continuidade
    mean_w = total_rank_sum / 2.0
    std_w = np.sqrt(n_diff * (n_diff + 1) * (2 * n_diff + 1) / 24.0)

    z = (w_stat - mean_w + 0.5) / std_w if std_w > 0 else 0.0
    # scipy erf para normal cdf
    from scipy.stats import norm
    p_val = float(2.0 * norm.cdf(-abs(z)))

    return {
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": float(w_stat),
        "rank_biserial_r": float(r_biserial),
        "n_diff": int(n_diff),
        "p_value": p_val,
    }


def holm_bonferroni_correction(p_values: Sequence[float]) -> list[float]:
    """Aplica a correção sequencial de Holm-Bonferroni sobre uma lista de valores-p."""
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m <= 1:
        return [float(p) for p in p_arr]

    order = np.argsort(p_arr)
    adj_p = np.zeros(m, dtype=np.float64)

    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        weight = m - rank_idx
        val = p_arr[orig_idx] * weight
        running_max = max(running_max, val)
        adj_p[orig_idx] = min(1.0, running_max)

    return [float(p) for p in adj_p]


def evaluate_raw_detections_dataframe(
    df: pd.DataFrame,
    tolerances: Sequence[int] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """Processa um DataFrame de detecções brutas em formato Parquet para múltiplos k.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame estruturado com o esquema padrão da ADR-010.
    tolerances : Sequence[int]
        Lista de janelas de tolerância k a computar em pós-processamento.

    Returns
    -------
    pd.DataFrame
        DataFrame expandido com colunas tp, fp, fn, precision, recall, f1, delay para cada k.
    """
    rows = []
    for _, row in df.iterrows():
        gt = row["ground_truth"]
        det = row["indices_detectados"]

        for k in tolerances:
            res = evaluate_single_series(gt, det, k=k)
            rows.append({
                "cenario_id": row.get("cenario_id"),
                "split": row.get("split", "avaliacao"),
                "perfil_ruido": row.get("perfil_ruido"),
                "n": row.get("n"),
                "magnitude": row.get("magnitude"),
                "semente": row.get("semente"),
                "detector": row.get("detector"),
                "k_tol": k,
                "tp": res.tp,
                "fp": res.fp,
                "fn": res.fn,
                "precision": res.precision,
                "recall": res.recall,
                "f1": res.f1,
                "mean_delay": res.mean_delay,
                "median_delay": res.median_delay,
                "tempo_cpu_ms": row.get("tempo_cpu_ms"),
            })

    return pd.DataFrame(rows)

