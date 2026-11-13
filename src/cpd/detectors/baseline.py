"""Wrappers para baselines da literatura de CPD utilizando a biblioteca ruptures.

Implementa PELT (Pruned Exact Linear Time) e Binseg (Binary Segmentation) com suporte a
modelos de custo L1 (mediana) e L2 (média), calibração de penalidade e varredura de thresholds.
"""

from __future__ import annotations

from typing import Literal, Sequence
import numpy as np
import ruptures as rpt

from cpd.detectors.base import BaseDetector, DetectionResult


class RupturesDetector(BaseDetector):
    """Detector unificado baseado em algoritmos clássicos da biblioteca ruptures.

    Parameters
    ----------
    algorithm : Literal["pelt", "binseg"]
        Algoritmo de segmentação:
        - "pelt": Pruned Exact Linear Time (Killick et al., 2012).
        - "binseg": Binary Segmentation gulosa (Scott & Knott, 1974).
    cost_model : Literal["l1", "l2", "normal"]
        Função de custo de discrepância:
        - "l1": desvios absolutos em relação à mediana (robusto a outliers).
        - "l2": desvios quadráticos em relação à média (assume ruído gaussiano).
    pen : float
        Valor da penalidade de regularização adicionada por novo ponto de mudança.
    n_bkps : int | None
        Número fixo de pontos de mudança a buscar (usado pelo Binseg se especificado).
        Se None, o particionamento é guiado pelo parâmetro `pen`.
    min_size : int
        Número mínimo de observações por segmento (padrão: 30).
    jump : int
        Subamostragem de candidatos a mudança para acelerar o particionamento (padrão: 1).
    """

    def __init__(
        self,
        algorithm: Literal["pelt", "binseg"] = "pelt",
        cost_model: Literal["l1", "l2"] = "l2",
        pen: float = 10.0,
        n_bkps: int | None = None,
        min_size: int = 30,
        jump: int = 1,
    ) -> None:
        if algorithm not in ("pelt", "binseg"):
            raise ValueError(f"Algoritmo não suportado: {algorithm}")
        if cost_model not in ("l1", "l2"):
            raise ValueError(f"Modelo de custo não suportado: {cost_model}")
        if pen < 0.0:
            raise ValueError(f"pen deve ser >= 0, recebido: {pen}")
        if min_size < 2:
            raise ValueError(f"min_size deve ser >= 2, recebido: {min_size}")

        self.algorithm = algorithm
        self.cost_model = cost_model
        self.pen = float(pen)
        self.n_bkps = n_bkps
        self.min_size = int(min_size)
        self.jump = int(jump)

    def _build_model(self) -> rpt.base.BaseEstimator:
        if self.algorithm == "pelt":
            return rpt.Pelt(model=self.cost_model, min_size=self.min_size, jump=self.jump)
        elif self.algorithm == "binseg":
            return rpt.Binseg(model=self.cost_model, min_size=self.min_size, jump=self.jump)
        else:
            raise ValueError(f"Algoritmo desconhecido: {self.algorithm}")

    def detect(self, series: np.ndarray | list[float]) -> list[int]:
        """Executa a detecção retornando os índices dos change points."""
        res = self.detect_detailed(series)
        return res.change_points

    def detect_detailed(self, series: np.ndarray | list[float]) -> DetectionResult:
        """Executa a detecção retornando os pontos e metadados."""
        x = self._validate_series(series)
        n = len(x)

        if n < 2 * self.min_size:
            return DetectionResult(change_points=[], metadata={"algorithm": self.algorithm})

        model = self._build_model()
        model.fit(x)

        if self.algorithm == "binseg" and self.n_bkps is not None:
            raw_bkps = model.predict(n_bkps=self.n_bkps)
        else:
            raw_bkps = model.predict(pen=self.pen)

        # A biblioteca ruptures inclui por convenção o índice final 'n' como último breakpoint.
        # Nós filtramos o ponto final 'n' para manter estrita concordância com nossa interface.
        cleaned_cps = sorted([int(cp) for cp in raw_bkps if 0 < cp < n])

        metadata = {
            "algorithm": self.algorithm,
            "cost_model": self.cost_model,
            "pen": self.pen,
            "n_bkps": self.n_bkps,
            "min_size": self.min_size,
        }

        return DetectionResult(change_points=cleaned_cps, metadata=metadata)

    @classmethod
    def calibrate_penalty_for_fpr(
        cls,
        calibration_series_list: Sequence[np.ndarray],
        algorithm: Literal["pelt", "binseg"] = "pelt",
        cost_model: Literal["l1", "l2"] = "l2",
        target_fpr: float = 0.01,
        pen_candidates: np.ndarray | None = None,
        min_size: int = 30,
    ) -> float:
        """Calibra a penalidade pen em séries sem regressão para fixar FPR <= target_fpr."""
        if pen_candidates is None:
            # Varredura logarítmica de penalidades
            pen_candidates = np.logspace(0, 3, 50)

        num_series = len(calibration_series_list)
        if num_series == 0:
            raise ValueError("A lista de séries de calibração não pode ser vazia.")

        best_pen = float(pen_candidates[-1])

        # OTIMIZAÇÃO: Pré-ajustar (fit) os modelos uma única vez por série
        fitted_models = []
        for s in calibration_series_list:
            x = cls._validate_series(s)
            if algorithm == "pelt":
                m = rpt.Pelt(model=cost_model, min_size=min_size).fit(x)
            else:
                m = rpt.Binseg(model=cost_model, min_size=min_size).fit(x)
            fitted_models.append((m, len(x)))

        # Busca monótona decrescente: começa pela maior penalidade (mais rápida) e para ao exceder target_fpr
        pen_desc = sorted((float(p) for p in pen_candidates), reverse=True)
        best_pen = pen_desc[0]
        for pen_val in pen_desc:
            false_alarms = 0
            for model, n in fitted_models:
                raw_bkps = model.predict(pen=pen_val)
                cps = [cp for cp in raw_bkps if 0 < cp < n]
                if len(cps) > 0:
                    false_alarms += 1
            fpr = false_alarms / num_series
            if fpr <= target_fpr:
                best_pen = pen_val
            else:
                break

        return best_pen
