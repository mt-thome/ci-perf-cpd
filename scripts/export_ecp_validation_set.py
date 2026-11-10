"""Script gerador do pacote de validação cruzada entre Python E-Div-Means e R ecp::e.divisive.

Gera 20 séries temporais fixas (com ground truth conhecido em múltiplos perfis de ruído e tamanhos),
exporta para CSVs individuais e um arquivo consolidado, além de gerar o script R reprodutível.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from cpd.generators.synthetic import NoiseType, generate_synthetic_series
from cpd.detectors import EDivisiveMeansDetector

OUT_DIR = Path('/home/mtthome/Projetos/ci-perf-cpd/data/reference_ecp')
OUT_DIR.mkdir(parents=True, exist_ok=True)

configs = [
    # (id, n, noise_type, cps, mag, seed)
    (1, 100, NoiseType.GAUSSIAN, [50], 3.0, 101),
    (2, 100, NoiseType.GAUSSIAN, [40], 2.0, 102),
    (3, 100, NoiseType.AR1, [50], 3.0, 103),
    (4, 100, NoiseType.STUDENT_T, [45], 3.0, 104),
    (5, 100, NoiseType.OUTLIERS, [50], 4.0, 105),
    (6, 200, NoiseType.GAUSSIAN, [70, 140], 2.5, 106),
    (7, 200, NoiseType.GAUSSIAN, [80, 150], 3.0, 107),
    (8, 200, NoiseType.AR1, [80, 130], 2.5, 108),
    (9, 200, NoiseType.STUDENT_T, [75, 145], 3.0, 109),
    (10, 200, NoiseType.BIMODAL, [70, 130], 3.5, 110),
    (11, 300, NoiseType.GAUSSIAN, [100, 200], 2.0, 111),
    (12, 300, NoiseType.GAUSSIAN, [90, 210], 3.0, 112),
    (13, 300, NoiseType.AR1, [100, 200], 3.0, 113),
    (14, 300, NoiseType.STUDENT_T, [100, 190], 3.0, 114),
    (15, 300, NoiseType.OUTLIERS, [100, 200], 4.0, 115),
    (16, 500, NoiseType.GAUSSIAN, [150, 320], 2.0, 116),
    (17, 500, NoiseType.AR1, [160, 330], 3.0, 117),
    (18, 500, NoiseType.STUDENT_T, [150, 300], 3.0, 118),
    (19, 500, NoiseType.OUTLIERS, [170, 340], 4.0, 119),
    (20, 500, NoiseType.BIMODAL, [150, 350], 3.0, 120),
]

manifest = []
detector_py = EDivisiveMeansDetector(alpha=0.05, permutations=99, min_size=25, seed=42)

for cid, n, ntype, cps, mag, seed in configs:
    s = generate_synthetic_series(
        n=n,
        base_level=100.0,
        noise_scale=4.0,
        noise_type=ntype,
        change_points=cps,
        magnitudes_std=mag,
        min_distance=25,
        seed=seed,
    )
    
    csv_file = f"series_{cid:02d}_n{n}_{ntype.value}.csv"
    csv_path = OUT_DIR / csv_file
    
    # Salvar CSV com valores
    df = pd.DataFrame({"index": np.arange(n), "value": s.values})
    df.to_csv(csv_path, index=False)
    
    # Rodar detecção Python E-Div-Means
    py_res = detector_py.detect_detailed(s.values)
    
    manifest.append({
        "series_id": cid,
        "filename": csv_file,
        "n": n,
        "noise_type": ntype.value,
        "ground_truth_cps": cps,
        "magnitude_std": mag,
        "seed": seed,
        "py_ediv_means_cps": py_res.change_points,
        "py_p_values": [float(p) for p in py_res.metadata["p_values"]],
    })

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(OUT_DIR / "manifest_series.csv", index=False)
with open(OUT_DIR / "manifest_series.json", "w") as f:
    json.dump(manifest, f, indent=2)

print("20 séries geradas e salvas em:", OUT_DIR)

