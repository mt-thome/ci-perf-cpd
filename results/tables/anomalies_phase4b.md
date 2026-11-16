# Auditoria de Anomalias (Fase 4b — Sem Interpretação)

## sempre_vazio (7)

- `Binseg_L2_perfil` | `ar1` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `Binseg_L2_perfil` | `bimodal` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `CUSUM_perfil` | `ar1` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `PELT_L1_perfil` | `ar1` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `PELT_L2_perfil` | `ar1` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `PELT_L2_perfil` | `bimodal` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula
- `PELT_L2_perfil` | `outliers` | n=100 | m=0.5 | k_true=1: 0 detecções em todas as 50 séries da célula

## indices_invalidos_ou_duplicados (1)

- `TODOS` | `TODOS` | n=0 | m=nan | k_true=-1: fora_de_intervalo=0, duplicados_na_serie=0

## divergencia_far_grid_vs_validacao (16)

- `Binseg_L2_gauss` | `gaussian` | n=1000 | m=0.0 | k_true=0: grid_far1k=0.000 vs val_far1k=0.220 (disjoint_ci=True)
- `Binseg_L2_perfil` | `gaussian` | n=1000 | m=0.0 | k_true=0: grid_far1k=0.000 vs val_far1k=0.220 (disjoint_ci=True)
- `CUSUM_gauss` | `gaussian` | n=100 | m=0.0 | k_true=0: grid_far1k=0.600 vs val_far1k=0.070 (disjoint_ci=False)
- `CUSUM_perfil` | `gaussian` | n=100 | m=0.0 | k_true=0: grid_far1k=0.600 vs val_far1k=0.070 (disjoint_ci=False)
- `EDiv_Means_calibrado` | `ar1` | n=100 | m=0.0 | k_true=0: grid_far1k=5.200 vs val_far1k=4.040 (disjoint_ci=False)
- `EDiv_Means_calibrado` | `ar1` | n=500 | m=0.0 | k_true=0: grid_far1k=5.240 vs val_far1k=2.060 (disjoint_ci=True)
- `EDiv_Means_calibrado` | `ar1` | n=1000 | m=0.0 | k_true=0: grid_far1k=4.400 vs val_far1k=1.550 (disjoint_ci=True)
- `EDiv_Means_calibrado_far1` | `ar1` | n=100 | m=0.0 | k_true=0: grid_far1k=5.200 vs val_far1k=4.040 (disjoint_ci=False)
- `EDiv_Means_calibrado_far1` | `ar1` | n=500 | m=0.0 | k_true=0: grid_far1k=5.240 vs val_far1k=2.060 (disjoint_ci=True)
- `EDiv_Means_calibrado_far1` | `ar1` | n=1000 | m=0.0 | k_true=0: grid_far1k=4.400 vs val_far1k=1.550 (disjoint_ci=True)
- `EDiv_Medians_calibrado` | `ar1` | n=100 | m=0.0 | k_true=0: grid_far1k=4.000 vs val_far1k=2.990 (disjoint_ci=False)
- `EDiv_Medians_calibrado` | `ar1` | n=500 | m=0.0 | k_true=0: grid_far1k=2.520 vs val_far1k=1.080 (disjoint_ci=True)
- `EDiv_Medians_calibrado` | `ar1` | n=1000 | m=0.0 | k_true=0: grid_far1k=2.200 vs val_far1k=1.070 (disjoint_ci=True)
- `EDiv_Medians_calibrado_far1` | `ar1` | n=100 | m=0.0 | k_true=0: grid_far1k=4.000 vs val_far1k=2.990 (disjoint_ci=False)
- `EDiv_Medians_calibrado_far1` | `ar1` | n=500 | m=0.0 | k_true=0: grid_far1k=2.520 vs val_far1k=1.080 (disjoint_ci=True)
- `EDiv_Medians_calibrado_far1` | `ar1` | n=1000 | m=0.0 | k_true=0: grid_far1k=2.200 vs val_far1k=1.070 (disjoint_ci=True)

## flag_calibracao_alvo_0.2 (6)

- `EDiv_Means` | `ar1` | n=1000 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=1.550)
- `EDiv_Medians` | `ar1` | n=1000 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=1.070)
- `EDiv_Means` | `ar1` | n=500 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=2.060)
- `EDiv_Medians` | `ar1` | n=500 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=1.080)
- `EDiv_Means` | `ar1` | n=100 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=4.040)
- `EDiv_Medians` | `ar1` | n=100 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=2.990)

## flag_calibracao_alvo_1.0 (22)

- `EDiv_Means` | `gaussian` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.390)
- `EDiv_Medians` | `gaussian` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.350)
- `EDiv_Means` | `gaussian` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.500)
- `EDiv_Medians` | `gaussian` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.520)
- `EDiv_Means` | `gaussian` | n=100 | m=0.0 | k_true=0: validacao_falhou (param=0.1, val_far1k=1.270)
- `EDiv_Means` | `ar1` | n=1000 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=1.550)
- `EDiv_Means` | `ar1` | n=500 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=2.060)
- `EDiv_Medians` | `ar1` | n=500 | m=0.0 | k_true=0: saturado_inferior (param=0.01, val_far1k=1.080)
- `EDiv_Means` | `ar1` | n=100 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=4.040)
- `EDiv_Medians` | `ar1` | n=100 | m=0.0 | k_true=0: saturado_inferior, validacao_falhou (param=0.01, val_far1k=2.990)
- `EDiv_Means` | `student_t` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.240)
- `EDiv_Medians` | `student_t` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.230)
- `EDiv_Means` | `student_t` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.380)
- `EDiv_Medians` | `student_t` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.440)
- `EDiv_Means` | `outliers` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.340)
- `EDiv_Medians` | `outliers` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.330)
- `EDiv_Means` | `outliers` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.580)
- `EDiv_Medians` | `outliers` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.580)
- `EDiv_Means` | `bimodal` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.380)
- `EDiv_Medians` | `bimodal` | n=1000 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.300)
- `EDiv_Means` | `bimodal` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.680)
- `EDiv_Medians` | `bimodal` | n=500 | m=0.0 | k_true=0: saturado_superior (param=0.2, val_far1k=0.680)
