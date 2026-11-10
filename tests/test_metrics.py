"""Suíte de testes unitários para o módulo de avaliação e matching temporal (Fase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from cpd.detectors import (
    EDivisiveDetector,
    RobustCUSUMDetector,
    RupturesDetector,
)
from cpd.evaluation.metrics import (
    aggregate_metrics,
    bootstrap_series_micro_f1,
    evaluate_single_series,
    evaluate_stationary_series,
    holm_bonferroni_correction,
    match_greedy_auxiliary,
    match_optimal_bipartite,
    paired_wilcoxon_test,
)


class TestIndexAlignmentIntegration:
    """Verifica a convenção de que o change point é o PRIMEIRO ponto do novo regime."""

    @pytest.fixture
    def step_series_clean(self) -> np.ndarray:
        """Degrau perfeito sem ruído onde o novo regime começa no índice 100."""
        # Índices 0..99 = 50.0; Índices 100..199 = 70.0
        return np.concatenate([np.full(100, 50.0), np.full(100, 70.0)])

    def test_edivisive_exact_index_100(self, step_series_clean: np.ndarray) -> None:
        """E-Divisive deve retornar exatamente índice 100 (sem off-by-one)."""
        det = EDivisiveDetector(alpha=0.05, permutations=50, min_size=20, metric="means", seed=42)
        cps = det.detect(step_series_clean)
        assert cps == [100]

    def test_pelt_exact_index_100_no_endpoint(self, step_series_clean: np.ndarray) -> None:
        """PELT deve retornar exatamente 100 e NUNCA o índice final n=200."""
        det = RupturesDetector(algorithm="pelt", cost_model="l2", pen=10.0, min_size=20)
        cps = det.detect(step_series_clean)
        assert cps == [100]
        assert 200 not in cps

    def test_binseg_exact_index_100(self, step_series_clean: np.ndarray) -> None:
        """Binseg deve retornar exatamente 100."""
        det = RupturesDetector(algorithm="binseg", cost_model="l1", n_bkps=1, min_size=20)
        cps = det.detect(step_series_clean)
        assert cps == [100]

    def test_cusum_alarm_and_estimated_start(self, step_series_clean: np.ndarray) -> None:
        """CUSUM deve ter alarme >= 100 e estimated_start == 100 em série sem ruído."""
        det = RobustCUSUMDetector(h=4.0, k_slack=0.5, init_window=30)
        res = det.detect_detailed(step_series_clean)
        alarm_idx = res.change_points[0]
        est_start = res.metadata["estimated_start_points"][0]
        # O alarme ocorre no momento ou poucos commits após
        assert alarm_idx >= 100
        assert est_start == 100


class TestOptimalBipartiteMatching:
    """Valida o matching 1-para-1 ótimo, disputas e casos de fronteira."""

    def test_exact_match(self) -> None:
        """Correspondência exata gera TP=1, delay=0."""
        matches, delays = match_optimal_bipartite(ground_truth=[100], detected=[100], k=3)
        assert matches == [(100, 100)]
        assert delays == [0]

    def test_boundary_distance_k_and_k_plus_1(self) -> None:
        """Desvio de exatamente k casa; desvio de k+1 é rejeitado."""
        # Com k=3: detecção em 103 deve casar
        m_in, _ = match_optimal_bipartite(ground_truth=[100], detected=[103], k=3)
        assert len(m_in) == 1

        # Com k=3: detecção em 104 está fora
        m_out, _ = match_optimal_bipartite(ground_truth=[100], detected=[104], k=3)
        assert len(m_out) == 0

    def test_two_detections_in_same_window_creates_one_fp(self) -> None:
        """Duas detecções próximas a um só ponto verdadeiro: 1 TP e 1 FP."""
        res = evaluate_single_series(ground_truth=[100], detected=[101, 102], k=3)
        assert res.tp == 1
        assert res.fp == 1
        assert res.fn == 0
        assert np.isclose(res.precision, 0.5)
        assert np.isclose(res.recall, 1.0)
        assert np.isclose(res.f1, 2.0 / 3.0)

    def test_optimal_vs_greedy_conflict(self) -> None:
        """Caso clássico onde guloso toma escolha míope e ótimo maximiza TP=2.

        G = [100, 103], D = [102, 105], k = 3.
        O guloso avalia pares por menor distância:
        Par (103, 102) tem dist=1. Se casar (103, 102), sobra (100) e (105) cuja dist=5 > k=3 -> TP=1.
        O ótimo emparelha (100, 102) com dist=2 e (103, 105) com dist=2 -> TP=2.
        """
        gt = [100, 103]
        det = [102, 105]

        # Guloso míope
        greedy_matches, _ = match_greedy_auxiliary(gt, det, k=3)
        assert len(greedy_matches) == 1  # Guloso empacou em 1 match

        # Ótimo global via Hungarian com custo finito
        opt_matches, _ = match_optimal_bipartite(gt, det, k=3)
        assert len(opt_matches) == 2  # Ótimo encontrou os dois matches
        assert opt_matches == [(100, 102), (103, 105)]

    def test_two_true_points_one_detection(self) -> None:
        """Dois pontos verdadeiros próximos [100, 104] com uma só detecção em 102."""
        res = evaluate_single_series(ground_truth=[100, 104], detected=[102], k=3)
        assert res.tp == 1
        assert res.fn == 1
        assert res.fp == 0
        assert np.isclose(res.precision, 1.0)
        assert np.isclose(res.recall, 0.5)

    def test_delay_signs(self) -> None:
        """Atraso positivo quando detecção é depois; negativo quando é antes."""
        # D = 102, G = 100 -> delay = +2
        _, delays_pos = match_optimal_bipartite(ground_truth=[100], detected=[102], k=5)
        assert delays_pos == [2]

        # D = 98, G = 100 -> delay = -2
        _, delays_neg = match_optimal_bipartite(ground_truth=[100], detected=[98], k=5)
        assert delays_neg == [-2]

    def test_asymmetric_window(self) -> None:
        """Janela assimétrica: aceita até 1 antes e até 5 depois."""
        # Detecção em 98 (delay -2) deve falhar se k_before=1
        m1, _ = match_optimal_bipartite(ground_truth=[100], detected=[98], k_before=1, k_after=5)
        assert len(m1) == 0

        # Detecção em 104 (delay +4) deve ter sucesso se k_after=5
        m2, _ = match_optimal_bipartite(ground_truth=[100], detected=[104], k_before=1, k_after=5)
        assert len(m2) == 1

    def test_boundary_points_zero_and_n_minus_one(self) -> None:
        """Verifica que pontos nos limites exatos das bordas (0 e n-1) casam perfeitamente sem erro."""
        # Borda inicial: commit 0
        r_zero = evaluate_single_series(ground_truth=[0], detected=[0], k=3)
        assert r_zero.tp == 1
        assert r_zero.matches == [(0, 0)]
        assert r_zero.delays == [0]

        # Borda final: commit n-1 (ex.: 299 em série de n=300)
        r_end = evaluate_single_series(ground_truth=[299], detected=[299], k=3)
        assert r_end.tp == 1
        assert r_end.matches == [(299, 299)]
        assert r_end.delays == [0]

        # Detecção na borda 0 com ground truth em 2 (distância 2 <= k=3)
        r_boundary_match = evaluate_single_series(ground_truth=[2], detected=[0], k=3)
        assert r_boundary_match.tp == 1
        assert r_boundary_match.matches == [(2, 0)]
        assert r_boundary_match.delays == [-2]


class TestDegenerateCasesAndAggregation:
    """Verifica convenções matemáticas e agregação Micro vs Macro."""

    def test_degenerate_no_detections(self) -> None:
        """Sem detecções: TP=0, FP=0, FN=N_true, Precision=NaN, Recall=0, F1=0."""
        res = evaluate_single_series(ground_truth=[100, 200], detected=[], k=3)
        assert res.tp == 0
        assert res.fp == 0
        assert res.fn == 2
        assert np.isnan(res.precision)
        assert res.recall == 0.0
        assert res.f1 == 0.0

    def test_degenerate_no_ground_truth_with_alarms(self) -> None:
        """Sem ground truth com alarmes: TP=0, FP=N_det, FN=0, Precision=0, Recall=NaN, F1=0."""
        res = evaluate_single_series(ground_truth=[], detected=[50, 120], k=3)
        assert res.tp == 0
        assert res.fp == 2
        assert res.fn == 0
        assert res.precision == 0.0
        assert np.isnan(res.recall)
        assert res.f1 == 0.0

    def test_degenerate_both_empty(self) -> None:
        """Ambos vazios (série estacionária perfeita sem falso alarme): TP=0, FP=0, FN=0, NaNs."""
        res = evaluate_single_series(ground_truth=[], detected=[], k=3)
        assert res.tp == 0
        assert res.fp == 0
        assert res.fn == 0
        assert np.isnan(res.precision)
        assert np.isnan(res.recall)
        assert np.isnan(res.f1)

    def test_stationary_series_evaluation(self) -> None:
        """Avaliação de falsos alarmes em séries puramente estacionárias."""
        # 10 séries de 500 commits: 8 sem alarme e 2 com alarmes (uma com 1 alarme, outra com 2)
        det_list = [[] for _ in range(8)] + [[150], [200, 350]]
        lens = [500] * 10
        stats = evaluate_stationary_series(det_list, lens)
        # FAR_series = 2 / 10 = 0.20 (20%)
        assert np.isclose(stats["far_series"], 0.20)
        # Total alarmes = 3; total commits = 5000; FAR_1k = (3/5000)*1000 = 0.6 alarmes / 1000 commits
        assert np.isclose(stats["far_1k_commits"], 0.6)

    def test_micro_macro_aggregation(self) -> None:
        """Verifica a agregação Micro e Macro com contabilização de NaNs."""
        r1 = evaluate_single_series(ground_truth=[100], detected=[100], k=3)  # TP=1, FP=0, FN=0 (P=1, R=1, F1=1)
        r2 = evaluate_single_series(ground_truth=[100], detected=[], k=3)     # TP=0, FP=0, FN=1 (P=NaN, R=0, F1=0)
        agg = aggregate_metrics([r1, r2])

        # Micro: sum_tp=1, sum_fp=0, sum_fn=1 -> P=1.0, R=0.5, F1=2*(1*0.5)/(1.5) = 2/3
        assert agg["micro"]["tp"] == 1
        assert agg["micro"]["fp"] == 0
        assert agg["micro"]["fn"] == 1
        assert np.isclose(agg["micro"]["precision"], 1.0)
        assert np.isclose(agg["micro"]["recall"], 0.5)
        assert np.isclose(agg["micro"]["f1"], 2.0 / 3.0)

        # Macro: F1 médio = (1.0 + 0.0)/2 = 0.5, com 1 Precision NaN
        assert np.isclose(agg["macro"]["f1_mean"], 0.5)
        assert agg["macro"]["precision_nan_count"] == 1


class TestStatisticalInferenceAndSignificance:
    """Testes para Bootstrap e Teste Pareado de Wilcoxon."""

    def test_bootstrap_series_confidence_interval(self) -> None:
        """Garante que o bootstrap de séries reproduz limites válidos em [0, 1]."""
        gt_list = [[100]] * 20
        det_list = [[100]] * 18 + [[]] * 2  # 18 séries com acerto perfeito, 2 sem detecção
        boot = bootstrap_series_micro_f1(gt_list, det_list, k=3, n_resamples=200, seed=42)
        assert 0.0 <= boot["ci_lower"] <= boot["ci_upper"] <= 1.0
        assert boot["ci_lower"] > 0.7

    def test_wilcoxon_paired_calculation_and_rank_biserial(self) -> None:
        """Valida o cálculo explícito de W+, W- e correlação rank-biserial sem empates."""
        # Diferenças estritamente distintas (sem empates):
        # A - B = [0.40, 0.25, 0.45, 0.15, 0.20, 0.28, 0.35, 0.18, 0.22, 0.31] (10 valores positivos distintos)
        scores_a = [0.90, 0.85, 0.95, 0.75, 0.80, 0.88, 0.92, 0.78, 0.82, 0.91]
        scores_b = [0.50, 0.60, 0.50, 0.60, 0.60, 0.60, 0.57, 0.60, 0.60, 0.60]
        res = paired_wilcoxon_test(scores_a, scores_b)
        assert res["n_diff"] == 10
        # Como A > B em todas as 10 diferenças (sem empates): W- = 0, W+ = 55, r = +1.0
        assert res["w_minus"] == 0.0
        assert res["w_plus"] == 55.0
        assert np.isclose(res["rank_biserial_r"], 1.0)
        # Asserção conservadora independente da aproximação normal com empates
        assert res["p_value"] < 0.05

    def test_holm_bonferroni_correction(self) -> None:
        """Verifica a monotonicidade e controle da correção de Holm."""
        raw_p = [0.01, 0.04, 0.03]
        adj_p = holm_bonferroni_correction(raw_p)
        assert len(adj_p) == 3
        # 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> running max garante adj[2] >= adj[1] >= adj[0]
        assert adj_p[0] <= adj_p[1] <= 1.0
