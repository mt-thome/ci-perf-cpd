"""Testes unitários para o gerador de séries temporais sintéticas de CI."""

from __future__ import annotations

import numpy as np
import pytest

from cpd.generators.synthetic import (
    NoiseType,
    SyntheticSeriesConfig,
    generate_synthetic_series,
)


class TestSyntheticSeriesGenerator:
    """Suíte de testes de integridade e propriedades estocásticas do gerador."""

    @pytest.mark.parametrize("n", [100, 500, 1000, 5000])
    def test_series_length(self, n: int) -> None:
        """Garante que a série gerada tem o comprimento exato solicitado."""
        series = generate_synthetic_series(n=n, seed=42)
        assert series.n == n
        assert len(series.values) == n
        assert len(series.ground_truth_signal) == n
        assert len(series.noise) == n

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Séries com a mesma semente devem ser estritamente idênticas bit a bit."""
        s1 = generate_synthetic_series(n=300, noise_type="gaussian", seed=123)
        s2 = generate_synthetic_series(n=300, noise_type="gaussian", seed=123)
        np.testing.assert_array_equal(s1.values, s2.values)
        assert s1.change_points == s2.change_points

    def test_different_seeds_produce_different_series(self) -> None:
        """Sementes distintas devem produzir trajetórias estocásticas diferentes."""
        s1 = generate_synthetic_series(n=300, seed=10)
        s2 = generate_synthetic_series(n=300, seed=20)
        assert not np.array_equal(s1.values, s2.values)

    @pytest.mark.parametrize(
        "noise_type",
        [
            NoiseType.GAUSSIAN,
            NoiseType.AR1,
            NoiseType.STUDENT_T,
            NoiseType.OUTLIERS,
            NoiseType.BIMODAL,
        ],
    )
    def test_all_noise_types(self, noise_type: NoiseType) -> None:
        """Todos os perfis de ruído devem gerar séries válidas sem NaN ou Inf."""
        series = generate_synthetic_series(
            n=400,
            noise_type=noise_type,
            noise_scale=4.0,
            seed=42,
        )
        assert np.all(np.isfinite(series.values))
        assert np.all(np.isfinite(series.noise))
        assert series.config.noise_type == noise_type

    def test_explicit_change_points(self) -> None:
        """Garante que pontos de mudança especificados manualmente são respeitados."""
        cps = [100, 250, 400]
        series = generate_synthetic_series(
            n=500,
            change_points=cps,
            base_level=50.0,
            magnitudes_std=2.0,
            noise_scale=5.0,
            seed=42,
        )
        assert series.change_points == cps
        assert len(series.segment_means) == len(cps) + 1
        # Verifica os níveis dos degraus no sinal teórico sem ruído
        assert np.isclose(series.ground_truth_signal[0], 50.0)
        assert np.isclose(series.ground_truth_signal[100], 60.0)  # +2 * 5.0 = 10.0
        assert np.isclose(series.ground_truth_signal[250], 70.0)
        assert np.isclose(series.ground_truth_signal[400], 80.0)

    def test_zero_change_points_stationary(self) -> None:
        """Séries sem change points devem ter sinal constante (base para FPR)."""
        series = generate_synthetic_series(
            n=200,
            num_change_points=0,
            base_level=120.0,
            seed=42,
        )
        assert len(series.change_points) == 0
        assert np.all(series.ground_truth_signal == 120.0)
        assert len(series.segment_means) == 1

    @pytest.mark.parametrize("magnitude_mult", [0.5, 1.0, 2.0, 5.0])
    def test_magnitude_multipliers(self, magnitude_mult: float) -> None:
        """Valida que as magnitudes em múltiplos de sigma são injetadas corretamente."""
        sigma = 4.0
        series = generate_synthetic_series(
            n=300,
            change_points=[150],
            noise_scale=sigma,
            magnitudes_std=magnitude_mult,
            base_level=100.0,
            seed=42,
        )
        expected_jump = magnitude_mult * sigma
        step_diff = series.ground_truth_signal[150] - series.ground_truth_signal[149]
        assert np.isclose(step_diff, expected_jump)

    def test_min_distance_constraint(self) -> None:
        """Garante que os change points sorteados respeitam a distância mínima."""
        min_dist = 50
        series = generate_synthetic_series(
            n=1000,
            num_change_points=5,
            min_distance=min_dist,
            seed=999,
        )
        cps = [0] + series.change_points + [series.n]
        diffs = np.diff(cps)
        assert np.all(diffs >= min_dist)

    def test_invalid_parameters_raise_error(self) -> None:
        """Validações de segurança para parâmetros fora do domínio admissível."""
        with pytest.raises(ValueError, match="n deve ser positivo"):
            SyntheticSeriesConfig(n=0)

        with pytest.raises(ValueError, match="|ar1_phi| deve ser estritamente menor que 1.0"):
            SyntheticSeriesConfig(ar1_phi=1.0)

        with pytest.raises(ValueError, match="student_t_df deve ser maior que 2.0"):
            SyntheticSeriesConfig(student_t_df=1.5)

        with pytest.raises(ValueError, match="outlier_prob deve estar em"):
            SyntheticSeriesConfig(outlier_prob=1.5)

        with pytest.raises(ValueError, match="bimodal_weight deve estar em"):
            SyntheticSeriesConfig(bimodal_weight=1.0)

    def test_serialization(self) -> None:
        """Garante que o objeto to_dict serializa corretamente."""
        series = generate_synthetic_series(n=100, seed=42)
        d = series.to_dict()
        assert d["n"] == 100
        assert len(d["values"]) == 100
        assert "change_points" in d
        assert "metadata" in d

