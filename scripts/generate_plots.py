"""Gera as figuras da Fase 1 e salva em reports/."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cpd.generators.synthetic import (
    NoiseType,
    SyntheticSeriesConfig,
    generate_synthetic_series,
)

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.titlesize': 14,
    'figure.dpi': 150,
})

REPORTS_DIR = Path('/home/mtthome/Projetos/ci-perf-cpd/reports')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Comparação dos Perfis de Ruído
fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

noise_types = [
    (NoiseType.GAUSSIAN, 'Ruído Gaussiano N(0, σ²)', 'steelblue'),
    (NoiseType.AR1, 'Ruído Autocorrelacionado AR(1) (φ=0.7)', 'darkorange'),
    (NoiseType.STUDENT_T, 'Ruído com Cauda Pesada (t-Student, ν=3)', 'forestgreen'),
    (NoiseType.OUTLIERS, 'Ruído com Outliers Esporádicos (GC Pauses, p=0.03)', 'crimson'),
    (NoiseType.BIMODAL, 'Ruído Bimodal (Runners Heterogêneos de CI, w=0.3)', 'purple'),
]

common_cps = [120, 280, 420]
common_mags = [2.0, -1.5, 3.0]

for ax, (ntype, title, color) in zip(axes, noise_types):
    series = generate_synthetic_series(
        n=500,
        base_level=100.0,
        noise_scale=4.0,
        noise_type=ntype,
        change_points=common_cps,
        magnitudes_std=common_mags,
        ar1_phi=0.7,
        student_t_df=3.0,
        outlier_prob=0.03,
        seed=42,
    )
    ax.plot(series.values, label='Série Observada (Métrica de CI)', color=color, alpha=0.85, lw=1.2)
    ax.plot(series.ground_truth_signal, label='Ground Truth (Nível Médio Real)', color='black', lw=2.0, ls='--')
    for cp in series.change_points:
        ax.axvline(cp, color='firebrick', linestyle=':', lw=1.5, alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Tempo (ms)')
    ax.legend(loc='upper left', frameon=True)

axes[-1].set_xlabel('Índice do Commit (Ordem Cronológica)')
plt.tight_layout()
plt.savefig(REPORTS_DIR / 'fig01_noise_profiles_comparison.png', bbox_inches='tight')
plt.close()

# 2. Impacto da Magnitude do Salto
magnitudes = [0.5, 1.0, 2.0, 5.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)
axes = axes.flatten()

for ax, mag in zip(axes, magnitudes):
    series = generate_synthetic_series(
        n=300,
        base_level=100.0,
        noise_scale=5.0,
        noise_type=NoiseType.GAUSSIAN,
        change_points=[150],
        magnitudes_std=mag,
        seed=42,
    )
    ax.plot(series.values, color='steelblue', alpha=0.7, lw=1.2, label='Observado')
    ax.plot(series.ground_truth_signal, color='red', lw=2.0, ls='--', label='Ground Truth')
    ax.axvline(150, color='black', ls=':', lw=1.5, label='Change Point (Commit 150)')
    snr = series.metadata['empirical_snr']
    ax.set_title(f'Salto: Δ = {mag}σ ({mag * 5.0:.1f} ms) | SNR = {snr:.2f}', fontweight='bold')
    ax.set_ylabel('Tempo (ms)')
    ax.set_xlabel('Commit')
    ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'fig02_magnitude_snr_comparison.png', bbox_inches='tight')
plt.close()

# 3. Séries Estacionárias
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
s_clean = generate_synthetic_series(n=400, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=100)
axes[0].plot(s_clean.values, color='teal', lw=1.2)
axes[0].axhline(100.0, color='black', ls='--', lw=2.0, label='Média Constante (μ=100)')
axes[0].set_title('Série Estacionária: Ruído Gaussiano (0 Change Points)', fontweight='bold')
axes[0].set_xlabel('Commit')
axes[0].set_ylabel('Tempo (ms)')
axes[0].legend(loc='upper right')

s_spikes = generate_synthetic_series(n=400, num_change_points=0, noise_type=NoiseType.OUTLIERS, outlier_prob=0.02, seed=100)
axes[1].plot(s_spikes.values, color='crimson', lw=1.2)
axes[1].axhline(100.0, color='black', ls='--', lw=2.0, label='Média Constante (μ=100)')
axes[1].set_title('Série Estacionária com Outliers Isolados (Desafio para CPD)', fontweight='bold')
axes[1].set_xlabel('Commit')
axes[1].set_ylabel('Tempo (ms)')
axes[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'fig03_stationary_series_fpr.png', bbox_inches='tight')
plt.close()

# 4. Distribuição Empírica e ACF
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
n_samples = 2000
noise_gauss = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=42).noise
noise_t = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.STUDENT_T, seed=42).noise
noise_bimodal = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.BIMODAL, seed=42).noise

sns.kdeplot(noise_gauss, ax=axes[0], label='Gaussiano', color='steelblue', lw=2)
sns.kdeplot(noise_t, ax=axes[0], label='t-Student (ν=3)', color='forestgreen', lw=2)
axes[0].set_title('Densidade Marginal: Gaussiano vs t-Student (Caudas)', fontweight='bold')
axes[0].set_xlabel('Ruído (ms)')
axes[0].legend()

sns.kdeplot(noise_bimodal, ax=axes[1], label='Bimodal', color='purple', lw=2)
axes[1].set_title('Densidade Marginal: Ruído Bimodal (Runners)', fontweight='bold')
axes[1].set_xlabel('Ruído (ms)')
axes[1].legend()

noise_ar1 = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.AR1, ar1_phi=0.7, seed=42).noise
lags = 20
acf_vals = [np.corrcoef(noise_ar1[:-k], noise_ar1[k:])[0, 1] if k > 0 else 1.0 for k in range(lags)]
axes[2].stem(range(lags), acf_vals, basefmt='k-')
axes[2].set_title('Função de Autocorrelação (ACF) - AR(1) com φ=0.7', fontweight='bold')
axes[2].set_xlabel('Lag temporal (commits)')
axes[2].set_ylabel('Autocorrelação')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'fig04_noise_properties_kde_acf.png', bbox_inches='tight')
plt.close()

print('Figuras geradas com sucesso em:', REPORTS_DIR)

