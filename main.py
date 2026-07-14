"""Анализ траекторий рейса SU10 (SVO -> LED) методом DTW.

Читает треки ADS-B из data/raw/SU10_*.csv, экспортирует их в KML/CSV для
Google Maps, считает DTW-расстояния между маршрутами и строит графики.

Запуск: py main.py
Результаты: results/exports/ (KML, CSV), results/figures/ (PNG)
"""
from __future__ import annotations

import logging
import sys

import numpy as np

from analysis import coordinate_dependence, compute_dtw_matrices, visited_towns
from config import LABELS, ORDER, RESULTS_EXPORTS, RESULTS_FIGURES
from figures import plot_dependence, plot_dtw_matrices, plot_deviation, plot_trajectories, plot_waypoints
from io_data import export_dtw_matrices, export_gmaps_csv, export_kml, load_all_tracks

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def print_matrix(title: str, matrix: np.ndarray) -> None:
    log.info("\n%s:", title)
    log.info("        " + "".join(f"{LABELS[f]:>9}" for f in ORDER))
    for i, flight_id in enumerate(ORDER):
        log.info(f"{LABELS[flight_id]:>7} " + "".join(f"{matrix[i, j]:9.3f}" for j in range(len(ORDER))))


def main() -> None:
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS_EXPORTS.mkdir(parents=True, exist_ok=True)

    log.info("Этап 1. Загрузка и характеристики данных")
    tracks = load_all_tracks()
    for flight_id in ORDER:
        track = tracks[flight_id]
        log.info(
            f"{LABELS[flight_id]}: точек={len(track):5d}  "
            f"старт=({track.lat.iloc[0]:.3f},{track.lon.iloc[0]:.3f})  "
            f"финиш=({track.lat.iloc[-1]:.3f},{track.lon.iloc[-1]:.3f})"
        )

    log.info("\nЭтап 2. Экспорт для Google Maps")
    export_gmaps_csv(tracks)
    export_kml(tracks)
    log.info("Сохранены: su10_routes.kml и gmaps_*.csv")

    log.info("\nЭтап 3. Пересечения (общие населённые пункты)")
    visited = visited_towns(tracks)
    for flight_id in ORDER:
        log.info(f"{LABELS[flight_id]} ({len(visited[flight_id])} пунктов): "
                  f"{' -> '.join(visited[flight_id])}")
    log.info("\nМатрица общих пунктов:")
    log.info("        " + "".join(f"{LABELS[f]:>9}" for f in ORDER))
    for a in ORDER:
        log.info(f"{LABELS[a]:>7} " + "".join(
            f"{len(set(visited[a]) & set(visited[b])):>9}" for b in ORDER))

    log.info("\nЭтап 4. DTW-расстояния")
    matrices = compute_dtw_matrices(tracks)
    log.info("\n-- Нормированные D/(n+m) --")
    print_matrix("2D (φ,λ)", matrices.full_2d)
    print_matrix("только φ", matrices.lat_only)
    print_matrix("только λ", matrices.lon_only)
    log.info("\n-- Сырые D (сумма стоимости пути, без нормировки) --")
    print_matrix("2D (φ,λ)", matrices.full_2d_raw)
    print_matrix("только φ", matrices.lat_only_raw)
    print_matrix("только λ", matrices.lon_only_raw)
    export_dtw_matrices(matrices)
    log.info("\nСохранены: dtw_2d.csv, dtw_lat_only.csv, dtw_lon_only.csv "
              "и dtw_*_raw.csv (сырые расстояния)")

    log.info("\nЭтап 5. Зависимость φ и λ (корреляция, R2, критерий Стьюдента)")
    log.info(f"{'День':8}{'N':>6}{'r':>9}{'R2':>9}{'t':>11}{'t_кр':>9}{'значимо':>9}")
    for dep in coordinate_dependence(tracks):
        log.info(
            f"{LABELS[dep.flight_id]:8}{dep.n:>6}{dep.r:>9.3f}{dep.r2:>9.3f}"
            f"{dep.t_stat:>11.1f}{dep.t_crit:>9.3f}{'да' if dep.significant else 'нет':>9}"
        )

    log.info("\nПостроение рисунков")
    plot_trajectories(tracks, RESULTS_FIGURES / "fig1_trajectories.png")
    plot_dependence(tracks, RESULTS_FIGURES / "fig3_dependence.png")
    plot_dtw_matrices(matrices, RESULTS_FIGURES / "fig4_dtw_matrices.png")
    reference = plot_deviation(tracks, RESULTS_FIGURES / "fig6_deviation.png")
    plot_waypoints(tracks, visited, RESULTS_FIGURES / "fig7_waypoints.png")

    log.info("Сохранены: fig1, fig3, fig4, fig6, fig7 (.png)")
    log.info(f"Эталон для fig6 (медоид): {LABELS[reference]}")
    log.info("\nГотово.")


if __name__ == "__main__":
    main()
