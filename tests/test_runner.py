"""Testes de integridade do runner e determinismo da Fase 4."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from cpd.experiments.runner import (
    compute_config_hash,
    compute_effective_snr,
    compute_hazard_levels,
    derive_stable_seed,
    generate_scenario_series,
    validate_calibration_manifest,
)


def test_stable_seed_derivation_cross_processes() -> None:
    """Garante que a derivação de semente é 100% determinística e imutável."""
    s1 = derive_stable_seed("sc_01", 42)
    s2 = derive_stable_seed("sc_01", 42)
    assert s1 == s2
    assert isinstance(s1, int)

    # Sementes diferentes para cenários diferentes
    s3 = derive_stable_seed("sc_02", 42)
    assert s1 != s3


def test_deterministic_series_generation() -> None:
    """Garante que o mesmo cenário gera valores idênticos."""
    cfg = {
        "cenario_id": "test_sc",
        "n": 100,
        "perfil_ruido": "gaussian",
        "noise_params": {},
        "k_true": 1,
        "magnitude_sigma": 2.0,
    }
    s1 = generate_scenario_series(cfg, seed_idx=10)
    s2 = generate_scenario_series(cfg, seed_idx=10)
    np.testing.assert_array_equal(s1.values, s2.values)
    assert s1.change_points == s2.change_points


def test_effective_snr_calculation() -> None:
    """Valida o cálculo de SNR efetiva com dispersão MAD."""
    noise = np.random.normal(0, 4.0, size=500)
    sigma_mad, snr = compute_effective_snr(magnitude_val=8.0, noise_series=noise)
    assert 3.0 < sigma_mad < 5.0
    assert 1.5 < snr < 2.5


def test_hazard_level_estimation() -> None:
    """Valida a estimativa do Hazard Level = log(media_antes/media_depois) com confronto ground truth."""
    # Degrau de 100ms para 50ms (dobro de velocidade -> log(100/50) = log(2) ~ 0.693)
    values = np.concatenate([np.full(50, 100.0), np.full(50, 50.0)])
    gt_signal = np.concatenate([np.full(50, 100.0), np.full(50, 50.0)])
    hazards = compute_hazard_levels(
        values=values,
        ground_truth_signal=gt_signal,
        ground_truth_cps=[50],
        detected_cps=[50],
    )
    assert len(hazards) == 1
    assert np.isclose(hazards[0]["hazard_level_est"], np.log(2.0), atol=1e-3)
    assert np.isclose(hazards[0]["hazard_level_true"], np.log(2.0), atol=1e-3)
    assert np.isclose(hazards[0]["hazard_level_abs_err"], 0.0, atol=1e-3)


def test_validate_calibration_manifest_rejects_provisional_and_hash_mismatch(tmp_path: Path) -> None:
    """Garante que o grid completo recusa calibração provisória ou com hash de config divergente."""
    cfg_file = tmp_path / "experiments.yaml"
    cfg_file.write_text("grid:\n  tamanhos_n: [100]\n", encoding="utf-8")
    cfg_hash = compute_config_hash(cfg_file)

    manifest_prov = {"_meta": {"provisorio": True, "config_sha256": cfg_hash}}
    # Piloto pode aceitar provisória
    validate_calibration_manifest(manifest_prov, cfg_file, allow_provisional=True)
    # Grid completo deve recusar
    with pytest.raises(ValueError, match="PROVISÓRIA"):
        validate_calibration_manifest(manifest_prov, cfg_file, allow_provisional=False)

    manifest_bad_hash = {"_meta": {"provisorio": False, "config_sha256": "deadbeef"}}
    with pytest.raises(ValueError, match="Hash de configuração"):
        validate_calibration_manifest(manifest_bad_hash, cfg_file, allow_provisional=False)

    manifest_ok = {"_meta": {"provisorio": False, "config_sha256": cfg_hash}}
    validate_calibration_manifest(manifest_ok, cfg_file, allow_provisional=False)

