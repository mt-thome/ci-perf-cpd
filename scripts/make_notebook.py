"""Script gerador do notebook 01_synthetic_generator_exploration.ipynb."""

import json
from pathlib import Path

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 01. Exploração e Visualização do Gerador de Séries Sintéticas de CI\n",
                "\n",
                "**Projeto**: Detecção de Regressões de Desempenho em Pipelines de Integração Contínua via Mineração de Séries Temporais  \n",
                "**Curso**: Mineração de Dados (Pós-Graduação, UNESP)  \n",
                "**Objetivo desta etapa**: Demonstrar as propriedades estocásticas e o ground truth controlado das séries temporais geradas pelo módulo `cpd.generators.synthetic`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from pathlib import Path\n",
                "\n",
                "from cpd.generators.synthetic import (\n",
                "    NoiseType,\n",
                "    SyntheticSeriesConfig,\n",
                "    generate_synthetic_series,\n",
                ")\n",
                "\n",
                "# Configuração de estilo visual acadêmico\n",
                "plt.style.use('seaborn-v0_8-whitegrid')\n",
                "plt.rcParams.update({\n",
                "    'font.size': 11,\n",
                "    'axes.labelsize': 12,\n",
                "    'axes.titlesize': 13,\n",
                "    'figure.titlesize': 14,\n",
                "    'figure.dpi': 150,\n",
                "})\n",
                "\n",
                "REPORTS_DIR = Path('../reports')\n",
                "REPORTS_DIR.mkdir(parents=True, exist_ok=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Comparação dos Perfis Estocásticos de Ruído\n",
                "\n",
                "Em pipelines de CI, a métrica de desempenho observada ao longo dos commits sofre perturbações que raramente são ruído branco gaussiano puro:\n",
                "- **Gaussiano**: Ruído de referência (i.i.d.).\n",
                "- **AR(1)**: Dependência temporal (aquecimento de JIT, caching progressivo).\n",
                "- **t-Student**: Caudas pesadas (flutuações amplas frequentes).\n",
                "- **Outliers**: Impulsos pontuais esporádicos (pausas de Garbage Collection, picos de I/O de rede) que **não** são regressões persistentes.\n",
                "- **Bimodal**: Variação de nós de build em clouds elásticas (execução em runners com CPUs heterogêneas)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)\n",
                "\n",
                "noise_types = [\n",
                "    (NoiseType.GAUSSIAN, 'Ruído Gaussiano N(0, σ²)', 'steelblue'),\n",
                "    (NoiseType.AR1, 'Ruído Autocorrelacionado AR(1) (φ=0.7)', 'darkorange'),\n",
                "    (NoiseType.STUDENT_T, 'Ruído com Cauda Pesada (t-Student, ν=3)', 'forestgreen'),\n",
                "    (NoiseType.OUTLIERS, 'Ruído com Outliers Esporádicos (GC Pauses, p=0.03)', 'crimson'),\n",
                "    (NoiseType.BIMODAL, 'Ruído Bimodal (Runners Heterogêneos de CI, w=0.3)', 'purple'),\n",
                "]\n",
                "\n",
                "common_cps = [120, 280, 420]\n",
                "common_mags = [2.0, -1.5, 3.0]  # Saltos em múltiplos de sigma\n",
                "\n",
                "for ax, (ntype, title, color) in zip(axes, noise_types):\n",
                "    series = generate_synthetic_series(\n",
                "        n=500,\n",
                "        base_level=100.0,\n",
                "        noise_scale=4.0,\n",
                "        noise_type=ntype,\n",
                "        change_points=common_cps,\n",
                "        magnitudes_std=common_mags,\n",
                "        ar1_phi=0.7,\n",
                "        student_t_df=3.0,\n",
                "        outlier_prob=0.03,\n",
                "        seed=42,\n",
                "    )\n",
                "    \n",
                "    # Série observada\n",
                "    ax.plot(series.values, label='Série Observada (Métrica de CI)', color=color, alpha=0.85, lw=1.2)\n",
                "    # Ground truth (sinal em degraus sem ruído)\n",
                "    ax.plot(series.ground_truth_signal, label='Ground Truth (Nível Médio Real)', color='black', lw=2.0, ls='--')\n",
                "    \n",
                "    # Marcadores dos change points\n",
                "    for cp in series.change_points:\n",
                "        ax.axvline(cp, color='firebrick', linestyle=':', lw=1.5, alpha=0.8)\n",
                "    \n",
                "    ax.set_title(title, fontweight='bold')\n",
                "    ax.set_ylabel('Tempo (ms)')\n",
                "    ax.legend(loc='upper left', frameon=True)\n",
                "\n",
                "axes[-1].set_xlabel('Índice do Commit (Ordem Cronológica)')\n",
                "plt.tight_layout()\n",
                "plt.savefig(REPORTS_DIR / 'fig01_noise_profiles_comparison.png', bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Impacto da Magnitude do Salto (Relação Sinal-Ruído: SNR)\n",
                "\n",
                "Avaliamos como a detectabilidade da regressão se comporta variando a magnitude do salto em múltiplos do desvio-padrão nominal:\n",
                "- $\\delta = 0.5\\sigma$: Regressão sutil, fortemente mascarada pelo ruído.\n",
                "- $\\delta = 1.0\\sigma$: Regressão moderada.\n",
                "- $\\delta = 2.0\\sigma$: Regressão típica em CI.\n",
                "- $\\delta = 5.0\\sigma$: Regressão severa (catastrófica)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "magnitudes = [0.5, 1.0, 2.0, 5.0]\n",
                "fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=False)\n",
                "axes = axes.flatten()\n",
                "\n",
                "for ax, mag in zip(axes, magnitudes):\n",
                "    series = generate_synthetic_series(\n",
                "        n=300,\n",
                "        base_level=100.0,\n",
                "        noise_scale=5.0,\n",
                "        noise_type=NoiseType.GAUSSIAN,\n",
                "        change_points=[150],\n",
                "        magnitudes_std=mag,\n",
                "        seed=42,\n",
                "    )\n",
                "    \n",
                "    ax.plot(series.values, color='steelblue', alpha=0.7, lw=1.2, label='Observado')\n",
                "    ax.plot(series.ground_truth_signal, color='red', lw=2.0, ls='--', label='Ground Truth')\n",
                "    ax.axvline(150, color='black', ls=':', lw=1.5, label='Change Point (Commit 150)')\n",
                "    \n",
                "    snr = series.metadata['empirical_snr']\n",
                "    ax.set_title(f'Salto: Δ = {mag}σ ({mag * 5.0:.1f} ms) | SNR = {snr:.2f}', fontweight='bold')\n",
                "    ax.set_ylabel('Tempo (ms)')\n",
                "    ax.set_xlabel('Commit')\n",
                "    ax.legend(loc='upper left', frameon=True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(REPORTS_DIR / 'fig02_magnitude_snr_comparison.png', bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Séries Estacionárias (Base para Avaliação de Falsos Positivos - FPR)\n",
                "\n",
                "Em CI, alertas falsos geram custos elevados e fadiga de alertas na equipe de desenvolvimento. Séries com $k=0$ change points permitem mensurar a taxa de falsos positivos pura dos detectores sob diferentes cenários de ruído."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 4))\n",
                "\n",
                "# Série Gaussiana Estacionária\n",
                "s_clean = generate_synthetic_series(n=400, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=100)\n",
                "axes[0].plot(s_clean.values, color='teal', lw=1.2)\n",
                "axes[0].axhline(100.0, color='black', ls='--', lw=2.0, label='Média Constante (μ=100)')\n",
                "axes[0].set_title('Série Estacionária: Ruído Gaussiano (0 Change Points)', fontweight='bold')\n",
                "axes[0].set_xlabel('Commit')\n",
                "axes[0].set_ylabel('Tempo (ms)')\n",
                "axes[0].legend(loc='upper right')\n",
                "\n",
                "# Série com Outliers Estacionária\n",
                "s_spikes = generate_synthetic_series(n=400, num_change_points=0, noise_type=NoiseType.OUTLIERS, outlier_prob=0.02, seed=100)\n",
                "axes[1].plot(s_spikes.values, color='crimson', lw=1.2)\n",
                "axes[1].axhline(100.0, color='black', ls='--', lw=2.0, label='Média Constante (μ=100)')\n",
                "axes[1].set_title('Série Estacionária com Outliers Isolados (Desafio para CPD)', fontweight='bold')\n",
                "axes[1].set_xlabel('Commit')\n",
                "axes[1].set_ylabel('Tempo (ms)')\n",
                "axes[1].legend(loc='upper right')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(REPORTS_DIR / 'fig03_stationary_series_fpr.png', bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Distribuição Empírica e Autocorrelação dos Ruídos\n",
                "\n",
                "Examinamos o comportamento das distribuições marginais dos ruídos gerados e a função de autocorrelação (ACF)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
                "\n",
                "# Histograma / KDE comparativo de cauda e bimodalidade\n",
                "n_samples = 2000\n",
                "noise_gauss = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.GAUSSIAN, seed=42).noise\n",
                "noise_t = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.STUDENT_T, seed=42).noise\n",
                "noise_bimodal = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.BIMODAL, seed=42).noise\n",
                "\n",
                "sns.kdeplot(noise_gauss, ax=axes[0], label='Gaussiano', color='steelblue', lw=2)\n",
                "sns.kdeplot(noise_t, ax=axes[0], label='t-Student (ν=3)', color='forestgreen', lw=2)\n",
                "axes[0].set_title('Densidade Marginal: Gaussiano vs t-Student (Caudas)', fontweight='bold')\n",
                "axes[0].set_xlabel('Ruído (ms)')\n",
                "axes[0].legend()\n",
                "\n",
                "sns.kdeplot(noise_bimodal, ax=axes[1], label='Bimodal', color='purple', lw=2)\n",
                "axes[1].set_title('Densidade Marginal: Ruído Bimodal (Runners)', fontweight='bold')\n",
                "axes[1].set_xlabel('Ruído (ms)')\n",
                "axes[1].legend()\n",
                "\n",
                "# ACF do AR(1)\n",
                "noise_ar1 = generate_synthetic_series(n=n_samples, num_change_points=0, noise_type=NoiseType.AR1, ar1_phi=0.7, seed=42).noise\n",
                "lags = 20\n",
                "acf_vals = [np.corrcoef(noise_ar1[:-k], noise_ar1[k:])[0, 1] if k > 0 else 1.0 for k in range(lags)]\n",
                "axes[2].stem(range(lags), acf_vals, basefmt='k-')\n",
                "axes[2].set_title('Função de Autocorrelação (ACF) - AR(1) com φ=0.7', fontweight='bold')\n",
                "axes[2].set_xlabel('Lag temporal (commits)')\n",
                "axes[2].set_ylabel('Autocorrelação')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig(REPORTS_DIR / 'fig04_noise_properties_kde_acf.png', bbox_inches='tight')\n",
                "plt.show()\n",
                "print('Figuras salvas com sucesso em reports/')"
            ]
        }
    ],
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.12"
        },
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_path = Path("/home/mtthome/Projetos/ci-perf-cpd/notebooks/01_synthetic_generator_exploration.ipynb")
nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Notebook criado com sucesso em: {nb_path}")

