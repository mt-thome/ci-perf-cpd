"""Implementação do algoritmo E-Divisive (Energy Statistics Change Point Detection).

Suporta formulação clássica baseada em médias amostrais (Matteson & James, 2014; Daly et al., 2020)
e formulação robusta baseada em medianas de distâncias pareadas (James et al., 2014).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple
import numpy as np
from more_itertools import pairwise
from numba import njit

from cpd.detectors.base import BaseDetector, DetectionResult


@njit(cache=True)
def _bit_add(bit: np.ndarray, idx: int, delta: int, k_max: int) -> None:
    while idx <= k_max:
        bit[idx] += delta
        idx += idx & (-idx)


@njit(cache=True)
def _bit_kth(bit: np.ndarray, k_zero_based: int, k_max: int, max_pow2: int) -> int:
    pos = 0
    acc = 0
    step = max_pow2
    target = k_zero_based + 1
    while step > 0:
        nxt = pos + step
        if nxt <= k_max and acc + bit[nxt] < target:
            acc += bit[nxt]
            pos = nxt
        step >>= 1
    return pos + 1


@njit(cache=True)
def _bit_median(
    bit: np.ndarray,
    count: int,
    unique_vals: np.ndarray,
    k_max: int,
    max_pow2: int,
) -> float:
    if count <= 0:
        return 0.0
    if count & 1:
        r = _bit_kth(bit, count >> 1, k_max, max_pow2)
        return float(unique_vals[r - 1])
    r1 = _bit_kth(bit, (count >> 1) - 1, k_max, max_pow2)
    r2 = _bit_kth(bit, count >> 1, k_max, max_pow2)
    return 0.5 * (float(unique_vals[r1 - 1]) + float(unique_vals[r2 - 1]))


@njit(cache=True)
def _t_stats_medians_bit(
    rank_mat: np.ndarray,
    unique_vals: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Cálculo exato em O(N^2 log N) da estatística de medianas via Árvore de Fenwick."""
    n_total = rank_mat.shape[0]
    stats = np.zeros(n_total, dtype=np.float64)
    if n_total < 2 * min_size:
        return stats

    k_max = len(unique_vals)
    max_pow2 = 1
    while (max_pow2 << 1) <= k_max:
        max_pow2 <<= 1

    bit_xx = np.zeros(k_max + 1, dtype=np.int32)
    bit_yy = np.zeros(k_max + 1, dtype=np.int32)
    bit_xy = np.zeros(k_max + 1, dtype=np.int32)

    for i in range(n_total):
        for j in range(i + 1, n_total):
            _bit_add(bit_yy, rank_mat[i, j], 1, k_max)

    for tau in range(n_total - min_size + 1):
        if tau > 0:
            t = tau - 1
            for i in range(t):
                r = rank_mat[i, t]
                _bit_add(bit_xy, r, -1, k_max)
                _bit_add(bit_xx, r, 1, k_max)
            for j in range(t + 1, n_total):
                r = rank_mat[t, j]
                _bit_add(bit_yy, r, -1, k_max)
                _bit_add(bit_xy, r, 1, k_max)

        n = tau
        m = n_total - tau
        if n >= min_size and m >= min_size:
            c_xx = (n * (n - 1)) // 2
            c_yy = (m * (m - 1)) // 2
            c_xy = n * m
            m_xx = _bit_median(bit_xx, c_xx, unique_vals, k_max, max_pow2)
            m_yy = _bit_median(bit_yy, c_yy, unique_vals, k_max, max_pow2)
            m_xy = _bit_median(bit_xy, c_xy, unique_vals, k_max, max_pow2)
            e_stat = 2.0 * m_xy - m_xx - m_yy
            val = ((n * m) / (n + m)) * e_stat
            if val > 0.0:
                stats[tau] = val

    return stats


def _prepare_rank_matrix(dist_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Comprime as distâncias únicas em ranks 1-based para uso na Árvore de Fenwick."""
    n = len(dist_matrix)
    i_u, j_u = np.triu_indices(n, k=1)
    unique_vals = np.unique(dist_matrix[i_u, j_u])
    rank_mat = (np.searchsorted(unique_vals, dist_matrix) + 1).astype(np.int32)
    return unique_vals, rank_mat


def warmup_edivisive_jit() -> None:
    """Aquece o compilador JIT do Numba para excluir tempo de compilação do profiling."""
    dummy = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    dm = np.abs(dummy[:, np.newaxis] - dummy)
    uv, rm = _prepare_rank_matrix(dm)
    _ = _t_stats_medians_bit(rm, uv, 5)


class EDivisiveDetector(BaseDetector):
    """Detector não-paramétrico hierárquico E-Divisive baseado em distâncias de energia.

    Parameters
    ----------
    alpha : float
        Nível de significância estatística para o teste de permutação de Monte Carlo (padrão: 0.05).
    permutations : int
        Número de permutações aleatórias no teste de hipótese (padrão: 100).
    min_size : int
        Tamanho mínimo de segmento permitido entre change points sucessivos (padrão: 30).
    metric : Literal["means", "medians"]
        Métrica de agregação das distâncias pareadas:
        - "means": Formulação clássica de Matteson & James (2014) usada em Daly et al. (2020).
        - "medians": Formulação robusta a outliers de James, Kejariwal & Matteson (2014).
    seed : int | None
        Semente pseudo-aleatória fixa para reprodutibilidade das permutações de Monte Carlo.
    exact_medians : bool
        Quando `metric == "medians"`, se True executa o cálculo exato em todos os
        candidatos `tau` e todos os pares (sem subamostragem). Se False (padrão),
        utiliza subamostragem determinística para séries grandes (n > 150).
    """

    def __init__(
        self,
        alpha: float = 0.05,
        permutations: int = 100,
        min_size: int = 30,
        metric: Literal["means", "medians"] = "means",
        seed: int | None = 42,
        exact_medians: bool = True,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha deve estar no intervalo (0, 1), recebido: {alpha}")
        if permutations < 0:
            raise ValueError(f"permutations deve ser >= 0, recebido: {permutations}")
        if min_size < 2:
            raise ValueError(f"min_size deve ser >= 2, recebido: {min_size}")
        if metric not in ("means", "medians"):
            raise ValueError(f"metric deve ser 'means' ou 'medians', recebido: {metric}")

        self.alpha = float(alpha)
        self.permutations = int(permutations)
        self.min_size = int(min_size)
        self.metric = metric
        self.seed = seed
        self.exact_medians = bool(exact_medians)

    @staticmethod
    def compute_distance_matrix(x: np.ndarray) -> np.ndarray:
        """Calcula a matriz simétrica N x N de distâncias euclidianas |x_i - x_j|."""
        # Operação totalmente vetorizada
        return np.abs(x[:, np.newaxis] - x)

    def _calculate_t_stats_means(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Calcula a estatística de divergência Q(tau) em O(N) após matriz pré-computada."""
        n_total = len(dist_matrix)
        stats = np.zeros(n_total, dtype=np.float64)

        if n_total < 2 * self.min_size:
            return stats

        # Acumula somas parciais
        # xy: soma de distâncias entre X e Y
        # x: soma de distâncias dentro de X
        # y: soma de distâncias dentro de Y
        xy = 0.0
        x = 0.0
        y = float(np.sum(np.triu(dist_matrix, k=1)))

        for tau in range(n_total):
            n = tau
            m = n_total - tau

            if n >= self.min_size and m >= self.min_size:
                # Médias amostrais
                x_avg = (2.0 * x) / (n * n) if n > 0 else 0.0
                y_avg = (2.0 * y) / (m * m) if m > 0 else 0.0
                xy_avg = xy / (n * m) if (n > 0 and m > 0) else 0.0

                # E-statistic: 2*E|X - Y| - E|X - X'| - E|Y - Y'|
                e_stat = 2.0 * xy_avg - x_avg - y_avg
                # Multiplicador amostral: (n * m) / (n + m)
                stats[tau] = max(0.0, ((n * m) / (n + m)) * e_stat)

            # Atualização incremental de somas ao mover a fronteira tau
            if tau < n_total:
                col_delta = float(np.sum(dist_matrix[:tau, tau]))
                row_delta = float(np.sum(dist_matrix[tau, tau + 1 :]))
                xy = xy - col_delta + row_delta
                x = x + col_delta
                y = y - row_delta

        return stats

    def _calculate_t_stats_medians_exact_naive(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Referência ingênua exata O(N^3) para validação unitária contra a Árvore de Fenwick."""
        n_total = len(dist_matrix)
        stats = np.zeros(n_total, dtype=np.float64)
        if n_total < 2 * self.min_size:
            return stats
        for tau in range(self.min_size, n_total - self.min_size + 1):
            n = tau
            m = n_total - tau
            within_x = dist_matrix[:tau, :tau]
            within_y = dist_matrix[tau:, tau:]
            between_xy = dist_matrix[:tau, tau:]
            i_x, j_x = np.triu_indices(n, k=1)
            i_y, j_y = np.triu_indices(m, k=1)
            m_xx = float(np.median(within_x[i_x, j_x])) if len(i_x) > 0 else 0.0
            m_yy = float(np.median(within_y[i_y, j_y])) if len(i_y) > 0 else 0.0
            m_xy = float(np.median(between_xy)) if between_xy.size > 0 else 0.0
            e_stat = 2.0 * m_xy - m_xx - m_yy
            stats[tau] = max(0.0, ((n * m) / (n + m)) * e_stat)
        return stats

    def _calculate_t_stats_medians(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Calcula a divergência baseada em medianas (Twitter EDM / BreakoutDetection).

        Quando `self.exact_medians=True`, executa o algoritmo exato em O(N^2 log N) via
        Árvore de Fenwick (Numba). Quando `False`, usa subamostragem determinística.
        """
        n_total = len(dist_matrix)
        if n_total < 2 * self.min_size:
            return np.zeros(n_total, dtype=np.float64)

        if self.exact_medians:
            unique_vals, rank_mat = _prepare_rank_matrix(dist_matrix)
            return _t_stats_medians_bit(rank_mat, unique_vals, self.min_size)

        stats = np.zeros(n_total, dtype=np.float64)
        # Em n grande, subamostragem do passo temporal
        step = 1 if n_total <= 150 else (2 if n_total <= 300 else 5)
        taus = list(range(self.min_size, n_total - self.min_size + 1, step))
        if taus[-1] != (n_total - self.min_size):
            taus.append(n_total - self.min_size)

        for tau in taus:
            n = tau
            m = n_total - tau

            within_x = dist_matrix[:tau, :tau]
            within_y = dist_matrix[tau:, tau:]
            between_xy = dist_matrix[:tau, tau:]

            # Triangular superior
            i_x, j_x = np.triu_indices(n, k=1)
            i_y, j_y = np.triu_indices(m, k=1)

            # Subamostragem de pares em blocos massivos para manter O(1) por partição
            if len(i_x) > 500:
                idx_sub_x = np.linspace(0, len(i_x) - 1, 300, dtype=int)
                m_xx = float(np.median(within_x[i_x[idx_sub_x], j_x[idx_sub_x]]))
            else:
                m_xx = float(np.median(within_x[i_x, j_x])) if len(i_x) > 0 else 0.0

            if len(i_y) > 500:
                idx_sub_y = np.linspace(0, len(i_y) - 1, 300, dtype=int)
                m_yy = float(np.median(within_y[i_y[idx_sub_y], j_y[idx_sub_y]]))
            else:
                m_yy = float(np.median(within_y[i_y, j_y])) if len(i_y) > 0 else 0.0

            if between_xy.size > 1000:
                idx_sub_xy = np.linspace(0, between_xy.size - 1, 500, dtype=int)
                m_xy = float(np.median(between_xy.ravel()[idx_sub_xy]))
            else:
                m_xy = float(np.median(between_xy)) if between_xy.size > 0 else 0.0

            e_stat = 2.0 * m_xy - m_xx - m_yy
            stats[tau] = max(0.0, ((n * m) / (n + m)) * e_stat)

        return stats

    def _calculate_t_stats(self, dist_matrix: np.ndarray) -> np.ndarray:
        if self.metric == "medians":
            return self._calculate_t_stats_medians(dist_matrix)
        return self._calculate_t_stats_means(dist_matrix)

    def _get_next_significant_change_point(
        self,
        dist_matrix: np.ndarray,
        change_points: list[int],
        memo: Dict[Tuple[int, int], Tuple[int, float]],
        rng: np.random.Generator,
    ) -> int | None:
        """Localiza a próxima partição divisiva mais significativa via teste de permutação."""
        n_total = len(dist_matrix)
        windows = [0] + sorted(change_points) + [n_total]
        candidates: List[Tuple[int, float, Tuple[int, int]]] = []

        for a, b in pairwise(windows):
            if (b - a) < 2 * self.min_size:
                continue

            bounds = (a, b)
            if bounds in memo:
                idx, val = memo[bounds]
            else:
                sub_dist = dist_matrix[a:b, a:b]
                stats = self._calculate_t_stats(sub_dist)
                idx = int(np.argmax(stats))
                val = float(stats[idx])
                memo[bounds] = (idx, val)

            if val > 0.0:
                candidates.append((idx + a, val, bounds))

        if not candidates:
            return None

        # Seleciona o melhor candidato global
        best_candidate = max(candidates, key=lambda x: x[1])
        best_cp, best_score, target_bounds = best_candidate

        # Teste de significância por permutação de Monte Carlo
        if self.permutations == 0:
            return best_cp

        a_win, b_win = target_bounds
        window_size = b_win - a_win
        sub_dist_target = dist_matrix[a_win:b_win, a_win:b_win]

        greater_or_equal = 0
        indices = np.arange(window_size)

        if self.metric == "medians" and self.exact_medians:
            unique_vals, rank_mat = _prepare_rank_matrix(sub_dist_target)
            for _ in range(self.permutations):
                permuted_indices = rng.permutation(indices)
                shuffled_ranks = rank_mat[np.ix_(permuted_indices, permuted_indices)]
                perm_stats = _t_stats_medians_bit(shuffled_ranks, unique_vals, self.min_size)
                max_perm = float(np.max(perm_stats)) if len(perm_stats) > 0 else 0.0
                if max_perm >= best_score:
                    greater_or_equal += 1
        else:
            for _ in range(self.permutations):
                permuted_indices = rng.permutation(indices)
                shuffled_dist = sub_dist_target[np.ix_(permuted_indices, permuted_indices)]
                perm_stats = self._calculate_t_stats(shuffled_dist)
                max_perm = float(np.max(perm_stats)) if len(perm_stats) > 0 else 0.0
                if max_perm >= best_score:
                    greater_or_equal += 1

        p_value = (1.0 + greater_or_equal) / (1.0 + self.permutations)

        if p_value <= self.alpha:
            return best_cp
        return None

    def _get_next_significant_change_point_detailed(
        self,
        dist_matrix: np.ndarray,
        change_points: list[int],
        memo: Dict[Tuple[int, int], Tuple[int, float]],
        rng: np.random.Generator,
    ) -> Tuple[int, float, Tuple[int, int]] | None:
        """Localiza a próxima partição divisiva retornando (cp, p_value, bounds_pai) se p_value <= alpha."""
        n_total = len(dist_matrix)
        windows = [0] + sorted(change_points) + [n_total]
        candidates: List[Tuple[int, float, Tuple[int, int]]] = []

        for a, b in pairwise(windows):
            if (b - a) < 2 * self.min_size:
                continue

            bounds = (a, b)
            if bounds in memo:
                idx, val = memo[bounds]
            else:
                sub_dist = dist_matrix[a:b, a:b]
                stats = self._calculate_t_stats(sub_dist)
                idx = int(np.argmax(stats))
                val = float(stats[idx])
                memo[bounds] = (idx, val)

            if val > 0.0:
                candidates.append((idx + a, val, bounds))

        if not candidates:
            return None

        best_candidate = max(candidates, key=lambda x: x[1])
        best_cp, best_score, target_bounds = best_candidate

        if self.permutations == 0:
            return best_cp, 0.0, target_bounds

        a_win, b_win = target_bounds
        window_size = b_win - a_win
        sub_dist_target = dist_matrix[a_win:b_win, a_win:b_win]

        greater_or_equal = 0
        indices = np.arange(window_size)

        if self.metric == "medians" and self.exact_medians:
            unique_vals, rank_mat = _prepare_rank_matrix(sub_dist_target)
            for _ in range(self.permutations):
                permuted_indices = rng.permutation(indices)
                shuffled_ranks = rank_mat[np.ix_(permuted_indices, permuted_indices)]
                perm_stats = _t_stats_medians_bit(shuffled_ranks, unique_vals, self.min_size)
                max_perm = float(np.max(perm_stats)) if len(perm_stats) > 0 else 0.0
                if max_perm >= best_score:
                    greater_or_equal += 1
        else:
            for _ in range(self.permutations):
                permuted_indices = rng.permutation(indices)
                shuffled_dist = sub_dist_target[np.ix_(permuted_indices, permuted_indices)]
                perm_stats = self._calculate_t_stats(shuffled_dist)
                max_perm = float(np.max(perm_stats)) if len(perm_stats) > 0 else 0.0
                if max_perm >= best_score:
                    greater_or_equal += 1

        p_value = (1.0 + greater_or_equal) / (1.0 + self.permutations)

        if p_value <= self.alpha:
            return best_cp, p_value, target_bounds
        return None

    def detect(self, series: np.ndarray | list[float]) -> list[int]:
        """Executa a partição divisiva recursiva retornando a lista de change points."""
        res = self.detect_detailed(series)
        return res.change_points

    def detect_detailed(self, series: np.ndarray | list[float]) -> DetectionResult:
        """Executa a detecção retornando os pontos, p-valores, histórico de divisão e metadados."""
        x = self._validate_series(series)
        rng = np.random.default_rng(self.seed)

        dist_matrix = self.compute_distance_matrix(x)
        change_points: list[int] = []
        p_values: list[float] = []
        parents: list[int | None] = []
        # Mapeia fronteiras ativas para o índice do change point que a gerou
        boundary_parent: Dict[Tuple[int, int], int | None] = {(0, len(x)): None}
        memo: Dict[Tuple[int, int], Tuple[int, float]] = {}

        while True:
            candidate_info = self._get_next_significant_change_point_detailed(dist_matrix, change_points, memo, rng)
            if candidate_info is None:
                break
            cp, p_val, target_bounds = candidate_info
            change_points.append(cp)
            p_values.append(p_val)
            parent_cp = boundary_parent.get(target_bounds, None)
            parents.append(parent_cp)

            # Subdivide o segmento pai em duas novas regiões filhas associadas a este cp
            a_win, b_win = target_bounds
            boundary_parent[(a_win, cp)] = cp
            boundary_parent[(cp, b_win)] = cp

        metadata = {
            "alpha": self.alpha,
            "permutations": self.permutations,
            "min_size": self.min_size,
            "metric": self.metric,
            "exact_medians": self.exact_medians if self.metric == "medians" else None,
            "num_detected": len(change_points),
            "p_values": p_values,
            "parent_change_points": parents,
            "division_history": [
                {"change_point": cp, "p_value": p, "parent": par}
                for cp, p, par in zip(change_points, p_values, parents)
            ],
        }

        # Retorna ordenado cronologicamente preservando a lista de pares
        return DetectionResult(
            change_points=sorted(change_points),
            scores=np.asarray(p_values, dtype=np.float64) if p_values else np.array([]),
            metadata=metadata,
        )



class EDivisiveMeansDetector(EDivisiveDetector):
    """E-Div-Means: Algoritmo de Matteson & James (2014) utilizado por Daly et al. (ICPE 2020) na MongoDB."""

    def __init__(
        self,
        alpha: float = 0.05,
        permutations: int = 100,
        min_size: int = 30,
        seed: int | None = 42,
    ) -> None:
        super().__init__(
            alpha=alpha,
            permutations=permutations,
            min_size=min_size,
            metric="means",
            seed=seed,
        )


class EDivisiveMediansDetector(EDivisiveDetector):
    """E-Div-Medians: Variante robusta com medianas de James, Kejariwal & Matteson (arXiv 2014 / IEEE 2016)."""

    def __init__(
        self,
        alpha: float = 0.05,
        permutations: int = 100,
        min_size: int = 30,
        seed: int | None = 42,
        exact_medians: bool = True,
    ) -> None:
        super().__init__(
            alpha=alpha,
            permutations=permutations,
            min_size=min_size,
            metric="medians",
            seed=seed,
            exact_medians=exact_medians,
        )


def filter_edivisive_by_alpha(
    division_history: list[dict[str, Any]],
    target_alpha: float,
) -> list[int]:
    """Filtra a árvore de divisões do E-Divisive para um alpha mais estrito.

    Uma divisão só é aceita se ela e TODOS os seus ancestrais na recursão tiverem
    p_value <= target_alpha.
    """
    accepted_cps: set[int] = set()
    result_cps: list[int] = []

    for entry in division_history:
        cp = entry["change_point"]
        p_val = entry["p_value"]
        parent = entry["parent"]

        # Se tiver pai, o pai obrigatoriamente precisa ter sido aceito
        parent_valid = (parent is None) or (parent in accepted_cps)

        if parent_valid and (p_val <= target_alpha):
            accepted_cps.add(cp)
            result_cps.append(cp)

    return sorted(result_cps)
