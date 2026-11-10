"""Detector CUSUM Tabular Bicaudal Robusto para métricas de desempenho em CI.

Utiliza estimadores robustos (mediana e MAD reescalonado) calculados em janela inicial
pré-regime, com reset pós-alarme e período de guarda temporal configurável.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from cpd.detectors.base import BaseDetector, DetectionResult


class RobustCUSUMDetector(BaseDetector):
    """Detector CUSUM tabular bicaudal robusto com estimadores locais.

    Parameters
    ----------
    h : float
        Limiar de decisão em unidades de desvio robusto (sigma_mad).
        Um alarme é disparado quando S+ > h ou S- > h.
    k_slack : float
        Parâmetro de folga/tolerância (slack) em unidades de desvio robusto.
        Padrão: 0.5 (otimizado para detectar desvios de 1.0 sigma).
    init_window : int
        Número de observações iniciais utilizadas para estimar o baseline (mu0, sigma0).
        Padrão: 30 commits.
    guard_window : int
        Janela de guarda (número de observações) após um alarme na qual novos alarmes
        são suprimidos para evitar falsas detecções na cauda do mesmo degrau.
        Padrão: 15 commits.
    update_mean_after_alarm : bool
        Se True, reavalia o nível de referência basal (mu0) usando a mediana
        das observações subsequentes ao alarme.
    """

    def __init__(
        self,
        h: float = 5.0,
        k_slack: float = 0.5,
        init_window: int = 30,
        guard_window: int = 15,
        update_mean_after_alarm: bool = True,
    ) -> None:
        if h <= 0.0:
            raise ValueError(f"O limiar h deve ser estritamente positivo, recebido: {h}")
        if k_slack < 0.0:
            raise ValueError(f"k_slack deve ser não-negativo, recebido: {k_slack}")
        if init_window < 3:
            raise ValueError(f"init_window deve ser >= 3, recebido: {init_window}")
        if guard_window < 0:
            raise ValueError(f"guard_window deve ser >= 0, recebido: {guard_window}")

        self.h = float(h)
        self.k_slack = float(k_slack)
        self.init_window = int(init_window)
        self.guard_window = int(guard_window)
        self.update_mean_after_alarm = bool(update_mean_after_alarm)

    @staticmethod
    def estimate_robust_location_scale(sample: np.ndarray) -> tuple[float, float]:
        """Estima a localização (mediana) e a dispersão (MAD reescalonado) de forma robusta.

        sigma = 1.4826 * median(|X - median(X)|)
        """
        med = float(np.median(sample))
        mad = float(np.median(np.abs(sample - med)))
        sigma = 1.4826 * mad
        # Evita divisão por zero caso a janela seja perfeitamente idêntica
        if sigma <= 1e-12:
            std_fallback = float(np.std(sample))
            sigma = std_fallback if std_fallback > 1e-12 else 1e-6
        return med, sigma

    def detect(self, series: np.ndarray | list[float]) -> list[int]:
        """Retorna os índices onde o CUSUM detectou novos regimes."""
        res = self.detect_detailed(series)
        return res.change_points

    def detect_detailed(self, series: np.ndarray | list[float]) -> DetectionResult:
        """Executa a detecção bicaudal retornando séries S+, S- e metadados."""
        x = self._validate_series(series)
        n = len(x)
        w_init = min(self.init_window, n)

        # Baseline inicial baseado nas primeiras w_init observações
        init_slice = x[:w_init]
        mu, sigma = self.estimate_robust_location_scale(init_slice)

        k = self.k_slack * sigma
        threshold = self.h * sigma

        s_pos = np.zeros(n, dtype=np.float64)
        s_neg = np.zeros(n, dtype=np.float64)

        change_points: list[int] = []
        change_points_estimated_start: list[int] = []
        guard_until = -1
        last_zero_pos = 0
        last_zero_neg = 0

        for t in range(n):
            if t == 0:
                delta = x[t] - mu
                s_pos[t] = max(0.0, delta - k)
                s_neg[t] = max(0.0, -delta - k)
            else:
                delta = x[t] - mu
                s_pos[t] = max(0.0, s_pos[t - 1] + delta - k)
                s_neg[t] = max(0.0, s_neg[t - 1] - delta - k)

            if s_pos[t] <= 0.0:
                last_zero_pos = t
            if s_neg[t] <= 0.0:
                last_zero_neg = t

            # Verificação de disparo de alarme
            if (s_pos[t] > threshold or s_neg[t] > threshold) and (t > guard_until):
                change_points.append(t)
                est_start = (last_zero_pos + 1) if s_pos[t] > threshold else (last_zero_neg + 1)
                change_points_estimated_start.append(min(t, est_start))
                guard_until = t + self.guard_window

                # Reset pós-detecção
                s_pos[t] = 0.0
                s_neg[t] = 0.0
                last_zero_pos = t
                last_zero_neg = t

                if self.update_mean_after_alarm:
                    # Atualiza o nível de referência usando as observações do novo regime
                    post_slice = x[t : min(n, t + self.init_window)]
                    if len(post_slice) >= 3:
                        mu_new, sigma_new = self.estimate_robust_location_scale(post_slice)
                        mu = mu_new
                        # Atualiza sigma moderadamente para manter estabilidade
                        sigma = 0.5 * (sigma + sigma_new)
                        k = self.k_slack * sigma
                        threshold = self.h * sigma

        max_scores = np.maximum(s_pos, s_neg) / sigma

        metadata = {
            "initial_mu": mu,
            "initial_sigma": sigma,
            "h": self.h,
            "k_slack": self.k_slack,
            "guard_window": self.guard_window,
            "estimated_start_points": change_points_estimated_start,
        }

        return DetectionResult(
            change_points=change_points,
            scores=max_scores,
            metadata=metadata,
        )

    @classmethod
    def calibrate_h_for_fpr(
        cls,
        calibration_series_list: Sequence[np.ndarray],
        target_fpr: float = 0.01,
        h_candidates: np.ndarray | None = None,
        k_slack: float = 0.5,
        init_window: int = 30,
        guard_window: int = 15,
    ) -> float:
        """Calibra o valor de h em séries sem regressão para fixar FPR <= target_fpr.

        Parameters
        ----------
        calibration_series_list : Sequence[np.ndarray]
            Lista de séries puramente estacionárias (ground_truth = sem regressão).
        target_fpr : float
            Taxa máxima de falsos alarmes tolerada (ex.: 0.01 = 1%).
            Definida como a proporção de séries estacionárias que tiveram pelo menos 1 falso alarme.
        h_candidates : np.ndarray | None
            Grade de candidatos a h a testar (padrão de 2.0 a 15.0 em passos de 0.2).

        Returns
        -------
        float
            Menor h tal que a proporção de falsos alarmes <= target_fpr.
        """
        if h_candidates is None:
            h_candidates = np.linspace(2.0, 15.0, 66)

        num_series = len(calibration_series_list)
        if num_series == 0:
            raise ValueError("A lista de séries de calibração não pode ser vazia.")

        best_h = float(h_candidates[-1])

        for h_val in sorted(h_candidates):
            detector = cls(
                h=h_val,
                k_slack=k_slack,
                init_window=init_window,
                guard_window=guard_window,
            )
            false_alarms = 0
            for s in calibration_series_list:
                cps = detector.detect(s)
                if len(cps) > 0:
                    false_alarms += 1
            fpr = false_alarms / num_series
            if fpr <= target_fpr:
                best_h = float(h_val)
                break

        return best_h

