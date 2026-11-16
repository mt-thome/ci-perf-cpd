"""Testes de integridade do runner e determinismo da Fase 4."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pytest

from cpd.experiments.calibration import calibrate_pair, poisson_rate_ci_95
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


def test_calibration_parallel_equals_sequential_and_poisson_ci() -> None:
    """Garante equivalência exata entre calibração sequencial e paralela por sub-lotes."""
    ci0 = poisson_rate_ci_95(0, 1000)
    assert ci0[0] == 0.0 and ci0[1] > 0.0

    seq = calibrate_pair(
        n=100,
        perfil="gaussian",
        n_calib=6,
        n_val=4,
        calib_seed_start=10000,
        val_seed_start=20000,
        targets_far_1k=[0.2, 1.0],
        executor=None,
        num_chunks=1,
    )
    with ProcessPoolExecutor(max_workers=2) as pool:
        par = calibrate_pair(
            n=100,
            perfil="gaussian",
            n_calib=6,
            n_val=4,
            calib_seed_start=10000,
            val_seed_start=20000,
            targets_far_1k=[0.2, 1.0],
            executor=pool,
            num_chunks=4,
        )

    assert seq["per_target"] == par["per_target"]


def test_all_detectors_regimes_match_manifest_and_fail_on_missing_key() -> None:
    """Para TODO detector x regime x perfil x n, garante parâmetro igual ao manifesto e falha sem chave."""
    import json
    import yaml
    from cpd.experiments.runner import execute_single_cell, instantiate_detector

    repo_root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((repo_root / "configs/experiments.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((repo_root / "results/calibration/calibration_manifest.json").read_text(encoding="utf-8"))

    perfis = [p["tipo"] for p in cfg["grid"]["perfis_ruido"]]
    tamanhos_n = [int(n) for n in cfg["grid"]["tamanhos_n"]]

    for det_cfg in cfg["detectores"]:
        cls_name = det_cfg["classe"]
        regime = det_cfg["regime_calibracao"]
        for perfil in perfis:
            for n in tamanhos_n:
                det_fast = dict(det_cfg)
                if "EDiv" in cls_name:
                    det_fast["params"] = {**det_cfg.get("params", {}), "permutations": 1}

                # 1. Verifica instantiate_detector e execute_single_cell (parâmetro gravado no formato Parquet)
                sc_cfg = {
                    "cenario_id": f"test_{perfil}_{n}",
                    "n": 100 if "EDiv" in cls_name else n,
                    "perfil_ruido": perfil,
                    "noise_params": {},
                    "k_true": 0,
                    "magnitude_sigma": 0.0,
                }
                # instantiate_detector usa n real do grid para lookup do manifesto
                det_obj = instantiate_detector(det_fast, manifest, perfil, n, stable_seed=42)
                if cls_name == "RobustCUSUMDetector":
                    expected_h = (
                        manifest["CUSUM_gauss"][str(n)]
                        if regime == "calibrado_gaussiano"
                        else manifest["CUSUM_perfil"][perfil][str(n)]
                    )
                    assert det_obj.h == pytest.approx(float(expected_h))
                elif cls_name == "RupturesDetector":
                    alg = det_cfg["algoritmo"].upper()
                    cost = det_cfg["cost_model"].upper()
                    if regime == "calibrado_gaussiano":
                        expected_pen = manifest[f"{alg}_{cost}_gauss"][str(n)]
                    else:
                        expected_pen = manifest[f"{alg}_{cost}_perfil"][perfil][str(n)]
                    assert det_obj.pen == pytest.approx(float(expected_pen))
                elif cls_name in ("EDivisiveMeansDetector", "EDivisiveMediansDetector"):
                    # Regime base (0.20) e regime calibrado_perfil (alpha' do manifesto)
                    assert det_obj.alpha == pytest.approx(float(det_cfg["params"]["alpha"]))
                    ediv_key = "EDiv_Means" if cls_name == "EDivisiveMeansDetector" else "EDiv_Medians"
                    cal_cfg = {**det_fast, "regime_calibracao": "calibrado_perfil"}
                    det_cal = instantiate_detector(cal_cfg, manifest, perfil, n, stable_seed=42)
                    assert det_cal.alpha == pytest.approx(float(manifest[ediv_key][perfil][str(n)]))

                # Executa uma célula com n do grid (apenas para EDiv usa n=100 acima se já testado; para demais usa n real)
                if "EDiv" not in cls_name:
                    row = execute_single_cell(sc_cfg, det_fast, seed_idx=1, calibration_dict=manifest, git_hash="test")
                    recorded = json.loads(row["parametros"])
                    if cls_name == "RobustCUSUMDetector":
                        assert recorded["h"] == pytest.approx(float(expected_h))
                    elif cls_name == "RupturesDetector":
                        assert recorded["pen"] == pytest.approx(float(expected_pen))

                # 2. Garante que falta de chave no manifesto lança KeyError (sem fallback silencioso)
                if regime in ("calibrado_gaussiano", "calibrado_perfil"):
                    with pytest.raises(KeyError):
                        instantiate_detector(det_fast, {}, perfil, n, stable_seed=42)
                else:
                    cal_cfg = {**det_fast, "regime_calibracao": "calibrado_perfil"}
                    with pytest.raises(KeyError):
                        instantiate_detector(cal_cfg, {}, perfil, n, stable_seed=42)


