"""Módulo público de avaliação de Change Point Detection."""

from cpd.evaluation.metrics import (
    SeriesEvaluationResult,
    aggregate_metrics,
    bootstrap_series_micro_f1,
    evaluate_raw_detections_dataframe,
    evaluate_single_series,
    evaluate_stationary_series,
    holm_bonferroni_correction,
    match_greedy_auxiliary,
    match_optimal_bipartite,
    paired_wilcoxon_test,
)

__all__ = [
    "SeriesEvaluationResult",
    "match_optimal_bipartite",
    "match_greedy_auxiliary",
    "evaluate_single_series",
    "evaluate_stationary_series",
    "aggregate_metrics",
    "bootstrap_series_micro_f1",
    "paired_wilcoxon_test",
    "holm_bonferroni_correction",
    "evaluate_raw_detections_dataframe",
]
