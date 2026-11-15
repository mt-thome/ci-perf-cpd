"""Validação independente (sementes 30000+) para CUSUM e PELT_L1 recalibrados (ADR-016)."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import pandas as pd

from cpd.detectors import RobustCUSUMDetector, RupturesDetector
from cpd.experiments.calibration import calibration_sizes_for_n, poisson_rate_ci_95
from cpd.generators.synthetic import NoiseType, generate_synthetic_series

REPO_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_DIR / "results/calibration/calibration_manifest.json"
OUT_CSV = REPO_DIR / "results/tables/recalibration_validation_30k.csv"


def _worker_chunk(payload: dict[str, Any]) -> list[tuple[str, str, int, int]]:
    n = int(payload["n"])
    perfil = payload["perfil"]
    ntype = NoiseType(perfil)
    seeds: list[int] = payload["seeds"]
    targets: dict[str, dict[str, float]] = payload["targets"]

    counts: dict[tuple[str, str], int] = {}
    ser_with: dict[tuple[str, str], int] = {}
    for tgt_key in targets:
        for det_key in ("CUSUM", "PELT_L1"):
            counts[(tgt_key, det_key)] = 0
            ser_with[(tgt_key, det_key)] = 0

    for s in seeds:
        arr = generate_synthetic_series(n=n, num_change_points=0, noise_type=ntype, seed=s).values
        cache: dict[tuple[str, float], int] = {}
        for tgt_key, pmap in targets.items():
            for det_key in ("CUSUM", "PELT_L1"):
                pval = float(pmap[det_key])
                ck = (det_key, pval)
                if ck not in cache:
                    if det_key == "CUSUM":
                        cache[ck] = len(RobustCUSUMDetector(h=pval).detect(arr))
                    else:
                        cache[ck] = len(RupturesDetector("pelt", "l1", pen=pval, min_size=25, jump=1).detect(arr))
                cnt = cache[ck]
                counts[(tgt_key, det_key)] += cnt
                if cnt > 0:
                    ser_with[(tgt_key, det_key)] += 1

    return [(tgt_key, det_key, counts[(tgt_key, det_key)], ser_with[(tgt_key, det_key)]) for (tgt_key, det_key) in counts]


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    details = manifest["detalhes_por_alvo"]
    workers = max(1, (os.cpu_count() or 2) - 2)
    rows = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n_val in (1000, 500, 100):
            _, n_val_size = calibration_sizes_for_n(n_val)
            seeds = list(range(30000, 30000 + n_val_size))
            chunks = [seeds[i::(workers * 2)] for i in range(workers * 2) if seeds[i::(workers * 2)]]
            for perfil in sorted(details.keys()):
                tgt_map = details[perfil][str(n_val)]
                targets_payload = {
                    tkey: {"CUSUM": float(trec["CUSUM"]["param"]), "PELT_L1": float(trec["PELT_L1"]["param"])}
                    for tkey, trec in tgt_map.items()
                }
                chunk_res = list(ex.map(
                    _worker_chunk,
                    [{"n": n_val, "perfil": perfil, "seeds": ch, "targets": targets_payload} for ch in chunks],
                ))
                agg_al: dict[tuple[str, str], int] = {}
                agg_sw: dict[tuple[str, str], int] = {}
                for part in chunk_res:
                    for tkey, dkey, al, sw in part:
                        agg_al[(tkey, dkey)] = agg_al.get((tkey, dkey), 0) + al
                        agg_sw[(tkey, dkey)] = agg_sw.get((tkey, dkey), 0) + sw

                exposure = n_val_size * n_val
                for tkey in sorted(tgt_map.keys()):
                    tgt_float = float(tkey)
                    for dkey in ("CUSUM", "PELT_L1"):
                        al = agg_al[(tkey, dkey)]
                        sw = agg_sw[(tkey, dkey)]
                        f1k = (al / exposure) * 1000.0
                        fser = sw / n_val_size
                        ci_l, ci_h = poisson_rate_ci_95(al, exposure)
                        rows.append({
                            "alvo_far_1k": tgt_float,
                            "perfil_ruido": perfil,
                            "n": n_val,
                            "detector": dkey,
                            "param": targets_payload[tkey][dkey],
                            "n_series_30k": n_val_size,
                            "alarmes_obs": al,
                            "alarmes_esp": (tgt_float / 1000.0) * exposure,
                            "far_1k_30k": f1k,
                            "ci95_low_30k": ci_l,
                            "ci95_high_30k": ci_h,
                            "far_series_30k": fser,
                            "validacao_30k_falhou": bool(ci_l > tgt_float),
                        })

    out_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Salvo {OUT_CSV} ({len(out_df)} linhas)")
    print("Falhas em 30000+ (alvo 0.2):", int(out_df[out_df["alvo_far_1k"] == 0.2]["validacao_30k_falhou"].sum()))
    print("Falhas em 30000+ (alvo 1.0):", int(out_df[out_df["alvo_far_1k"] == 1.0]["validacao_30k_falhou"].sum()))


if __name__ == "__main__":
    main()
