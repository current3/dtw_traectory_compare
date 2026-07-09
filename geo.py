"""Геометрия маршрутов: расстояния и передискретизация трека по длине пути."""
from __future__ import annotations

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2: float, lon2: float):
    """Приближённое расстояние в км; долгота масштабируется по cos(средней широты)."""
    lat_mean = np.radians((lat1 + lat2) / 2)
    return np.hypot((lon2 - lon1) * 111.32 * np.cos(lat_mean), (lat2 - lat1) * 110.57)


def resample_track(track: pd.DataFrame, n_points: int = 200, in_km: bool = False):
    """Приводит трек к n_points точкам, равномерно распределённым по длине пути.

    Проекция строится в локальной декартовой системе (км) с фиксированным
    центром (37°в.д., 56°с.ш.), чтобы разные треки оставались сравнимы между собой.
    """
    lat, lon = track.lat.values, track.lon.values
    lat_mean = np.radians(lat.mean())
    x = (lon - 37.0) * 111.32 * np.cos(lat_mean)
    y = (lat - 56.0) * 110.57
    path_length = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    samples = np.linspace(0, path_length[-1], n_points)
    if in_km:
        return np.column_stack([np.interp(samples, path_length, x), np.interp(samples, path_length, y)])
    return np.interp(samples, path_length, lat), np.interp(samples, path_length, lon)
