"""Interface base e contratos para algoritmos de detecção de pontos de mudança (CPD)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DetectionResult:
    """Estrutura com os resultados retornados por um detector.

    Parameters
    ----------
    change_points : list[int]
        Lista ordenada de índices temporais detectados como inícios de novos regimes.
    scores : np.ndarray | None
        Série de pontuações ou estatísticas de teste ao longo do tempo (quando aplicável).
    metadata : dict[str, Any]
        Metadados da execução (parâmetros calibrados, tempos, estatísticas intermediárias).
    """

    change_points: list[int]
    scores: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDetector(ABC):
    """Classe base abstrata para todos os detectores de Change Point Detection."""

    @abstractmethod
    def detect(self, series: np.ndarray | list[float]) -> list[int]:
        """Detecta pontos de mudança na série temporal fornecida.

        Parameters
        ----------
        series : np.ndarray | list[float]
            Vetor univariado 1D de observações cronológicas.

        Returns
        -------
        list[int]
            Lista ordenada com os índices dos change points detectados.
        """
        pass

    def detect_detailed(self, series: np.ndarray | list[float]) -> DetectionResult:
        """Executa a detecção retornando objeto detalhado com scores e metadados.

        A implementação padrão invoca `detect` e encapsula o resultado. Subclasses
        podem sobrescrever este método para expor curvas de estatísticas contínuas.
        """
        cps = self.detect(series)
        return DetectionResult(change_points=cps)

    @staticmethod
    def _validate_series(series: np.ndarray | list[float]) -> np.ndarray:
        """Garante formato 1D contiguous float64 sem valores nulos."""
        arr = np.asarray(series, dtype=np.float64)
        if arr.ndim != 1:
            if arr.ndim == 2 and (arr.shape[0] == 1 or arr.shape[1] == 1):
                arr = arr.flatten()
            else:
                raise ValueError(f"A série temporal deve ser 1D, shape recebido: {arr.shape}")
        if len(arr) == 0:
            raise ValueError("A série temporal não pode ser vazia.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("A série temporal contém valores NaN ou infinitos.")
        return arr

