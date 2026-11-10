"""Suíte de testes unitários para os detectores de Change Point Detection (Fase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from cpd.detectors.baseline import RupturesDetector
from cpd.detectors.cusum import RobustCUSUMDetector
from cpd.detectors.edivisive import EDivisiveDetector
from cpd.generators.synthetic import NoiseType, generate_synthetic_series


class TestDetectorsCanonicalCases:
    """Validação com casos determinísticos e canônicos conhecidos."""

    @pytest.fixture
    def step_series(self) -> np.ndarray:
        """Série com salto degrau ideal em commit 100 de magnitude 20ms."""
        s1 = np.full(100, 50.0)
        s2 = np.full(100, 70.0)
        return np.concatenate([s1, s2])

    @pytest.fixture
    def flat_series(self) -> np.ndarray:
        """Série perfeitamente constante sem nenhuma mudança."""
        return np.full(200, 100.0)

    def test_cusum_canonical_step(self, step_series: np.ndarray) -> None:
        """CUSUM deve detectar com precisão o degrau limpo."""
        detector = RobustCUSUMDetector(h=4.0, k_slack=0.5, init_window=30)
        cps = detector.detect(step_series)
        assert len(cps) == 1
        # O alarme CUSUM dispara logo nos primeiros commits após a mudança (ex: 100 a 104)
        assert abs(cps[0] - 100) <= 5

    def test_cusum_flat_series(self, flat_series: np.ndarray) -> None:
        """CUSUM não deve disparar falsos positivos em série perfeitamente plana."""
        detector = RobustCUSUMDetector(h=5.0)
        cps = detector.detect(flat_series)
        assert len(cps) == 0

    def test_edivisive_canonical_step(self, step_series: np.ndarray) -> None:
        """E-Divisive deve encontrar a transição exata em t=100."""
        detector = EDivisiveDetector(alpha=0.05, permutations=50, min_size=20, metric="means", seed=42)
        cps = detector.detect(step_series)
        assert len(cps) == 1
        assert cps[0] == 100

    def test_edivisive_flat_series(self, flat_series: np.ndarray) -> None:
        """E-Divisive não deve encontrar pontos em série homogênea."""
        detector = EDivisiveDetector(alpha=0.05, permutations=50, min_size=20, seed=42)
        cps = detector.detect(flat_series)
        assert len(cps) == 0

    def test_edivisive_medians_canonical_step(self) -> None:
        """E-Divisive com medianas deve encontrar a regressão em série com ruído contínuo."""
        series = generate_synthetic_series(n=200, change_points=[100], magnitudes_std=3.0, noise_scale=1.0, seed=42).values
        detector = EDivisiveDetector(alpha=0.05, permutations=50, min_size=20, metric="medians", seed=42)
        cps = detector.detect(series)
        assert len(cps) == 1
        assert abs(cps[0] - 100) <= 2

    def test_ruptures_pelt_canonical_step(self, step_series: np.ndarray) -> None:
        """PELT (L2) deve localizar o degrau em t=100."""
        detector = RupturesDetector(algorithm="pelt", cost_model="l2", pen=10.0, min_size=20)
        cps = detector.detect(step_series)
        assert len(cps) == 1
        assert cps[0] == 100

    def test_ruptures_binseg_canonical_step(self, step_series: np.ndarray) -> None:
        """Binseg (L1) deve localizar o degrau em t=100."""
        detector = RupturesDetector(algorithm="binseg", cost_model="l1", n_bkps=1, min_size=20)
        cps = detector.detect(step_series)
        assert len(cps) == 1
        assert cps[0] == 100


class TestDetectorsRobustnessAndCalibration:
    """Testes com ruído sintético e calibração com sementes disjuntas."""

    def test_cusum_calibration_fpr(self) -> None:
        """Verifica a calibração de h do CUSUM para FPR <= 5% em conjunto sintético."""
        # Conjunto de calibração estacionário
        calib_series = [
            generate_synthetic_series(n=200, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=s).values
            for s in range(1000, 1020)  # Sementes da faixa 1000
        ]
        calibrated_h = RobustCUSUMDetector.calibrate_h_for_fpr(
            calibration_series_list=calib_series,
            target_fpr=0.05,
            h_candidates=np.linspace(2.0, 10.0, 25),
        )
        assert calibrated_h >= 2.0

        # Avaliação em conjunto de teste com sementes estritamente disjuntas (faixa 2000)
        test_series = [
            generate_synthetic_series(n=200, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=s).values
            for s in range(2000, 2020)
        ]
        detector = RobustCUSUMDetector(h=calibrated_h)
        false_alarms = sum(1 for s in test_series if len(detector.detect(s)) > 0)
        empirical_fpr = false_alarms / len(test_series)
        # Deve manter taxa baixa
        assert empirical_fpr <= 0.15

    def test_edivisive_reproducibility(self) -> None:
        """E-Divisive deve produzir exatamente as mesmas detecções com a mesma seed."""
        series = generate_synthetic_series(n=300, change_points=[150], magnitudes_std=3.0, seed=42).values
        d1 = EDivisiveDetector(alpha=0.05, permutations=50, seed=123)
        d2 = EDivisiveDetector(alpha=0.05, permutations=50, seed=123)
        assert d1.detect(series) == d2.detect(series)

    def test_edivisive_outlier_resistance(self) -> None:
        """Compara a robustez de medianas vs médias na presença de um pico isolado."""
        # Série com um único outlier no meio mas SEM regressão permanente
        series = np.full(160, 50.0)
        series[80] = 500.0  # Spike transitório enorme
        # O detector com medianas não deve ser ludibriado pelo spike
        det_med = EDivisiveDetector(alpha=0.01, permutations=30, min_size=20, metric="medians", seed=42)
        assert len(det_med.detect(series)) == 0

    def test_ruptures_penalty_calibration(self) -> None:
        """Verifica a calibração da penalidade no PELT."""
        calib_series = [
            generate_synthetic_series(n=200, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=s).values
            for s in range(3000, 3015)
        ]
        calibrated_pen = RupturesDetector.calibrate_penalty_for_fpr(
            calibration_series_list=calib_series,
            algorithm="pelt",
            cost_model="l2",
            target_fpr=0.05,
            pen_candidates=np.logspace(0.5, 2.5, 10),
            min_size=20,
        )
        assert calibrated_pen > 0
