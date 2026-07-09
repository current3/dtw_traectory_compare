"""Анализ маршрутов: общие населённые пункты, DTW-матрицы, зависимость координат."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import ORDER, R_KM, TOWNS
from dtw import dtw_distance, t_critical
from geo import haversine_km, resample_track


def towns_on_route(track: pd.DataFrame, r_km: float = R_KM) -> list[str]:
    """Населённые пункты, мимо которых маршрут прошёл в пределах r_km."""
    return [
        town for town, (t_lat, t_lon) in TOWNS.items()
        if haversine_km(track.lat.values, track.lon.values, t_lat, t_lon).min() <= r_km
    ]


def visited_towns(tracks: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    return {flight_id: towns_on_route(tracks[flight_id]) for flight_id in ORDER}


@dataclass(frozen=True)
class DtwMatrices:
    full_2d: np.ndarray   # DTW по (широта, долгота)
    lat_only: np.ndarray  # DTW только по широте
    lon_only: np.ndarray  # DTW только по долготе


def compute_dtw_matrices(tracks: dict[str, pd.DataFrame], n_points: int = 200) -> DtwMatrices:
    """Считает попарные DTW-расстояния между всеми рейсами после z-нормализации."""
    resampled = {flight_id: resample_track(tracks[flight_id], n_points) for flight_id in ORDER}
    all_lat = np.concatenate([resampled[f][0] for f in ORDER])
    all_lon = np.concatenate([resampled[f][1] for f in ORDER])
    lat_mean, lat_std = all_lat.mean(), all_lat.std()
    lon_mean, lon_std = all_lon.mean(), all_lon.std()

    z_2d = {f: np.column_stack([(resampled[f][0] - lat_mean) / lat_std,
                                 (resampled[f][1] - lon_mean) / lon_std]) for f in ORDER}
    z_lat = {f: (resampled[f][0] - lat_mean) / lat_std for f in ORDER}
    z_lon = {f: (resampled[f][1] - lon_mean) / lon_std for f in ORDER}

    def pairwise(z: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([[dtw_distance(z[a], z[b]) for b in ORDER] for a in ORDER])

    return DtwMatrices(pairwise(z_2d), pairwise(z_lat), pairwise(z_lon))


@dataclass(frozen=True)
class CoordinateDependence:
    flight_id: str
    n: int
    r: float
    r2: float
    t_stat: float
    t_crit: float

    @property
    def significant(self) -> bool:
        return abs(self.t_stat) > self.t_crit


def coordinate_dependence(tracks: dict[str, pd.DataFrame]) -> list[CoordinateDependence]:
    """Линейная зависимость долготы от широты по каждому рейсу (корреляция, R², t-критерий)."""
    results = []
    for flight_id in ORDER:
        track = tracks[flight_id]
        n = len(track)
        r = np.corrcoef(track.lat, track.lon)[0, 1]
        slope, intercept = np.polyfit(track.lat, track.lon, 1)
        predicted = intercept + slope * track.lat
        r2 = 1 - ((track.lon - predicted) ** 2).sum() / ((track.lon - track.lon.mean()) ** 2).sum()
        t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
        results.append(CoordinateDependence(flight_id, n, r, r2, t_stat, t_critical(n - 2)))
    return results


def find_reference_flight(tracks_km: dict[str, np.ndarray]) -> str:
    """Медоид: рейс с минимальной суммой DTW-расстояний до всех остальных."""
    return min(ORDER, key=lambda f: sum(dtw_distance(tracks_km[f], tracks_km[k]) for k in ORDER))
