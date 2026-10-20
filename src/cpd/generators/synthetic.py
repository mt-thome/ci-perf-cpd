"""Módulo gerador de séries temporais sintéticas para benchmark de Change Point Detection (CPD).

Simula métricas de desempenho em pipelines de CI (ex.: tempo de execução de testes/benchmarks)
com perturbações realistas e ground truth controlado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np


class NoiseType(str, Enum):
    """Tipos de perfil estocástico suportados para modelagem de ruído."""

    GAUSSIAN = "gaussian"
    AR1 = "ar1"
    STUDENT_T = "student_t"
    OUTLIERS = "outliers"
    BIMODAL = "bimodal"


@dataclass(frozen=True)
class SyntheticSeriesConfig:
    """Configuração dos parâmetros do gerador sintético de séries temporais de CI.

    Parameters
    ----------
    n : int
        Número total de observações (ex.: commits sucessivos).
    base_level : float
        Nível basal inicial da métrica de desempenho (ex.: 100.0 ms).
    noise_scale : float
        Desvio-padrão nominal do ruído base (sigma).
    noise_type : NoiseType | str
        Tipo de perturbação estocástica a injetar.
    ar1_phi : float
        Coeficiente autorregressivo para ruído AR(1), deve satisfazer |phi| < 1.0.
    student_t_df : float
        Graus de liberdade para ruído t-Student (df > 2 para variância finita).
    outlier_prob : float
        Probabilidade independente de ocorrência de outlier por observação.
    outlier_scale_multiplier : float
        Magnitude do outlier expressa em múltiplos de sigma (ex.: 5.0 = 5 * sigma).
    bimodal_weight : float
        Proporção (peso) da modalidade secundária de hardware no ruído bimodal (0.0 < w < 1.0).
    bimodal_mean_shift_std : float
        Separação entre os dois modos de máquina em múltiplos de sigma.
    num_change_points : int
        Número de pontos de mudança a injetar (ignorado se change_points for especificado).
    change_points : list[int] | None
        Lista explícita de índices temporais de início de novos regimes. Se None,
        são sorteados respeitando min_distance.
    min_distance : int
        Distância mínima permitida entre pontos de mudança e em relação às extremidades da série.
    magnitudes_std : float | Sequence[float]
        Magnitude do salto em cada change point expressa em múltiplos de sigma (ex.: 1.0, 2.0, 5.0).
    direction : str
        Direção da transição de nível: "positive" (regressão de latência), "negative" (otimização),
        "alternating" ou "random".
    seed : int | None
        Semente pseudo-aleatória fixa para assegurar estrita reprodutibilidade.
    """

    n: int = 500
    base_level: float = 100.0
    noise_scale: float = 5.0
    noise_type: NoiseType | str = NoiseType.GAUSSIAN
    ar1_phi: float = 0.6
    student_t_df: float = 3.0
    outlier_prob: float = 0.02
    outlier_scale_multiplier: float = 5.0
    bimodal_weight: float = 0.3
    bimodal_mean_shift_std: float = 3.0
    num_change_points: int = 2
    change_points: list[int] | None = None
    min_distance: int = 30
    magnitudes_std: float | Sequence[float] = 2.0
    direction: str = "positive"
    seed: int | None = 42

    def __post_init__(self) -> None:
        """Valida os parâmetros da configuração."""
        if self.n <= 0:
            raise ValueError(f"O tamanho da série n deve ser positivo, recebido: {self.n}")
        if self.noise_scale < 0:
            raise ValueError(f"noise_scale deve ser não-negativo, recebido: {self.noise_scale}")
        if isinstance(self.noise_type, str):
            try:
                object.__setattr__(self, "noise_type", NoiseType(self.noise_type))
            except ValueError:
                valid = [t.value for t in NoiseType]
                raise ValueError(f"noise_type inválido '{self.noise_type}'. Opções: {valid}")
        if abs(self.ar1_phi) >= 1.0:
            raise ValueError(f"|ar1_phi| deve ser estritamente menor que 1.0 para estacionariedade, recebido: {self.ar1_phi}")
        if self.student_t_df <= 2.0:
            raise ValueError(f"student_t_df deve ser maior que 2.0 para variância finita, recebido: {self.student_t_df}")
        if not (0.0 <= self.outlier_prob <= 1.0):
            raise ValueError(f"outlier_prob deve estar em [0.0, 1.0], recebido: {self.outlier_prob}")
        if not (0.0 < self.bimodal_weight < 1.0):
            raise ValueError(f"bimodal_weight deve estar em (0.0, 1.0), recebido: {self.bimodal_weight}")
        if self.min_distance < 1:
            raise ValueError(f"min_distance deve ser >= 1, recebido: {self.min_distance}")


@dataclass
class SyntheticSeries:
    """Representação de uma série temporal sintética gerada com ground truth explícito."""

    values: np.ndarray
    change_points: list[int]
    ground_truth_signal: np.ndarray
    noise: np.ndarray
    segment_means: list[float]
    config: SyntheticSeriesConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Tamanho da série."""
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        """Converte a série e metadados para dicionário serializável."""
        return {
            "n": self.n,
            "values": self.values.tolist(),
            "change_points": self.change_points,
            "segment_means": self.segment_means,
            "noise_type": self.config.noise_type.value if isinstance(self.config.noise_type, NoiseType) else str(self.config.noise_type),
            "noise_scale": self.config.noise_scale,
            "seed": self.config.seed,
            "metadata": self.metadata,
        }


def _generate_noise(
    config: SyntheticSeriesConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gera o componente estocástico de ruído conforme o perfil selecionado."""
    n = config.n
    sigma = config.noise_scale
    if sigma == 0.0:
        return np.zeros(n, dtype=np.float64)

    noise_type = config.noise_type

    if noise_type == NoiseType.GAUSSIAN:
        return rng.normal(loc=0.0, scale=sigma, size=n)

    elif noise_type == NoiseType.AR1:
        # e_t = phi * e_{t-1} + sqrt(1 - phi^2) * sigma * epsilon_t
        # Assegura que Var(e_t) = sigma^2 em regime estacionário
        phi = config.ar1_phi
        shock_scale = sigma * np.sqrt(1.0 - phi**2)
        shocks = rng.normal(loc=0.0, scale=shock_scale, size=n)
        e = np.zeros(n, dtype=np.float64)
        e[0] = rng.normal(loc=0.0, scale=sigma)
        for t in range(1, n):
            e[t] = phi * e[t - 1] + shocks[t]
        return e

    elif noise_type == NoiseType.STUDENT_T:
        # Variância teórica de t_df é df / (df - 2) para df > 2.
        # Reescalonamos para que Var(e_t) = sigma^2
        df = config.student_t_df
        raw = rng.standard_t(df=df, size=n)
        theoretical_std = np.sqrt(df / (df - 2.0))
        return raw * (sigma / theoretical_std)

    elif noise_type == NoiseType.OUTLIERS:
        # Ruído base gaussiano + picos esporádicos independentes
        base = rng.normal(loc=0.0, scale=sigma, size=n)
        is_outlier = rng.uniform(0.0, 1.0, size=n) < config.outlier_prob
        # Em testes de CI, gargalos transitórios (ex: GC pause) elevam a latência
        outlier_magnitude = config.outlier_scale_multiplier * sigma
        spikes = np.where(is_outlier, outlier_magnitude, 0.0)
        return base + spikes

    elif noise_type == NoiseType.BIMODAL:
        # Mistura de duas gaussianas simulando instâncias de runners heterogêneas
        w = config.bimodal_weight
        delta = config.bimodal_mean_shift_std * sigma
        # Subpopulação 0 tem média 0, Subpopulação 1 tem média delta
        # Centramos para que a perturbação média do ruído seja zero:
        # E[ruído] = (1 - w)*0 + w*delta = w * delta
        mean_offset = w * delta
        is_subpop1 = rng.uniform(0.0, 1.0, size=n) < w
        offsets = np.where(is_subpop1, delta - mean_offset, -mean_offset)
        fluctuations = rng.normal(loc=0.0, scale=sigma, size=n)
        return offsets + fluctuations

    else:
        raise ValueError(f"NoiseType não suportado: {noise_type}")


def _resolve_change_points(
    config: SyntheticSeriesConfig,
    rng: np.random.Generator,
) -> list[int]:
    """Determina a lista ordenada de índices dos change points."""
    if config.change_points is not None:
        cps = sorted(list(config.change_points))
        # Validação
        for cp in cps:
            if not (0 < cp < config.n):
                raise ValueError(f"Change point {cp} fora dos limites permitidos (0, {config.n})")
        return cps

    k = config.num_change_points
    if k <= 0:
        return []

    min_dist = config.min_distance
    min_required = (k + 1) * min_dist
    if config.n < min_required:
        raise ValueError(
            f"Série de tamanho n={config.n} é insuficiente para {k} change points com min_distance={min_dist} (necessário ao menos {min_required})"
        )

    # Amostragem ordenada com restrição de distância mínima
    available_range = config.n - (k + 1) * min_dist
    shifts = sorted(rng.choice(available_range + 1, size=k, replace=True))
    cps: list[int] = []
    for i, shift in enumerate(shifts):
        cp = (i + 1) * min_dist + shift
        cps.append(int(cp))

    return sorted(list(set(cps)))


def _resolve_magnitudes(
    config: SyntheticSeriesConfig,
    num_cps: int,
    rng: np.random.Generator,
) -> list[float]:
    """Calcula o vetor de magnitudes absolutas para cada salto de change point."""
    if num_cps == 0:
        return []

    sigma = config.noise_scale
    if isinstance(config.magnitudes_std, (int, float)):
        base_mags = [float(config.magnitudes_std) * sigma] * num_cps
    else:
        seq = list(config.magnitudes_std)
        if len(seq) < num_cps:
            # Cicla caso haja menos magnitudes do que change points
            base_mags = [float(seq[i % len(seq)]) * sigma for i in range(num_cps)]
        else:
            base_mags = [float(seq[i]) * sigma for i in range(num_cps)]

    # Determinar direção dos saltos
    signed_mags: list[float] = []
    for i, mag in enumerate(base_mags):
        if config.direction == "positive":
            signed_mags.append(mag)
        elif config.direction == "negative":
            signed_mags.append(-mag)
        elif config.direction == "alternating":
            sign = 1.0 if (i % 2 == 0) else -1.0
            signed_mags.append(sign * mag)
        elif config.direction == "random":
            sign = 1.0 if rng.uniform() > 0.5 else -1.0
            signed_mags.append(sign * mag)
        else:
            raise ValueError(f"Direção inválida: {config.direction}")

    return signed_mags


def generate_synthetic_series(
    config: SyntheticSeriesConfig | None = None,
    **kwargs: Any,
) -> SyntheticSeries:
    """Gera uma série temporal sintética com níveis médios e ruído controlados.

    Pode receber uma instância de `SyntheticSeriesConfig` ou parâmetros via kwargs.

    Returns
    -------
    SyntheticSeries
        Instância contendo a série com ruído, ground truth puro, lista de change points e metadados.
    """
    if config is None:
        config = SyntheticSeriesConfig(**kwargs)
    elif kwargs:
        raise ValueError("Forneça config OU kwargs, não ambos.")

    rng = np.random.default_rng(config.seed)

    cps = _resolve_change_points(config, rng)
    magnitudes = _resolve_magnitudes(config, len(cps), rng)

    # Construção do sinal em degraus (ground truth signal)
    signal = np.full(config.n, config.base_level, dtype=np.float64)
    current_level = float(config.base_level)
    segment_means = [current_level]

    last_idx = 0
    for cp, mag in zip(cps, magnitudes):
        signal[last_idx:cp] = current_level
        current_level += mag
        segment_means.append(current_level)
        last_idx = cp
    signal[last_idx:] = current_level

    # Componente estocástico de ruído
    noise = _generate_noise(config, rng)
    values = signal + noise

    metadata = {
        "num_change_points": len(cps),
        "injected_magnitudes": magnitudes,
        "empirical_snr": float(np.ptp(signal) / (np.std(noise) + 1e-9)),
    }

    return SyntheticSeries(
        values=values,
        change_points=cps,
        ground_truth_signal=signal,
        noise=noise,
        segment_means=segment_means,
        config=config,
        metadata=metadata,
    )

