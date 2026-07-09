"""Динамическая трансформация времени (DTW) и связанная статистика."""
from __future__ import annotations

import numpy as np


def dtw_distance(a, b, want_path: bool = False):
    """Динамическая трансформация времени (расстояние, нормированное по длине пути).

    Если want_path=True, дополнительно возвращает путь выравнивания —
    список пар индексов (i, j) от начала маршрута к концу.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if b.ndim == 1:
        b = b[:, None]
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            step_cost = np.sqrt(((a[i - 1] - b[j - 1]) ** 2).sum())
            cost[i, j] = step_cost + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    distance = cost[n, m] / (n + m)
    if not want_path:
        return distance

    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return distance, path[::-1]


def t_critical(dof: int, alpha: float = 0.05) -> float:
    """Критическое значение t-статистики Стьюдента (двусторонний тест).

    Использует scipy, если доступен; иначе — приближение по нормальному
    распределению (1.96 при alpha=0.05).
    """
    try:
        from scipy import stats
        return stats.t.ppf(1 - alpha / 2, dof)
    except ImportError:
        return 1.96
