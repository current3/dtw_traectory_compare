"""Загрузка треков ADS-B и экспорт результатов (Google Maps CSV, KML)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import DtwMatrices
from config import DATA_RAW, KML_COLORS, LABELS, ORDER, RESULTS_EXPORTS


def load_track(path: str) -> pd.DataFrame:
    """Читает CSV трека ADS-B и разбивает Position на широту/долготу."""
    df = pd.read_csv(path)
    df["lat"] = df["Position"].str.split(",").str[0].astype(float)
    df["lon"] = df["Position"].str.split(",").str[1].astype(float)
    return df.sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)


def load_all_tracks(data_dir: Path = DATA_RAW) -> dict[str, pd.DataFrame]:
    """Загружает все SU10_*.csv из data_dir; ключ словаря — идентификатор рейса."""
    tracks: dict[str, pd.DataFrame] = {}
    for path in sorted(glob.glob(str(data_dir / "SU10_*.csv"))):
        flight_id = os.path.basename(path).removeprefix("SU10_").removesuffix(".csv")
        tracks[flight_id] = load_track(path)
    return tracks


def export_gmaps_csv(tracks: dict[str, pd.DataFrame], out_dir: Path = RESULTS_EXPORTS) -> None:
    """Сохраняет по одному CSV на трек — для импорта в Google My Maps."""
    for flight_id in ORDER:
        track = tracks[flight_id]
        name = LABELS[flight_id].replace(" ", "")
        track[["lat", "lon", "Altitude", "Speed", "UTC"]].to_csv(
            out_dir / f"gmaps_{name}.csv", index=False, encoding="utf-8-sig")


def export_kml(tracks: dict[str, pd.DataFrame], out_dir: Path = RESULTS_EXPORTS) -> None:
    """Сохраняет все треки в один KML-файл с раскраской по рейсу."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>SU10</name>',
    ]
    for flight_id in ORDER:
        track = tracks[flight_id]
        coords = " ".join(f"{lon},{lat},0" for lat, lon in zip(track.lat, track.lon))
        lines += [
            f'<Style id="s_{flight_id}"><LineStyle><color>{KML_COLORS[flight_id]}</color>'
            f'<width>3</width></LineStyle></Style>',
            f'<Placemark><name>{LABELS[flight_id]}</name><styleUrl>#s_{flight_id}</styleUrl>'
            f'<LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates>'
            f'</LineString></Placemark>',
        ]
    lines.append("</Document></kml>")
    (out_dir / "su10_routes.kml").write_text("\n".join(lines), encoding="utf-8")


def _matrix_to_frame(matrix: np.ndarray) -> pd.DataFrame:
    labels = [LABELS[f] for f in ORDER]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def export_dtw_matrices(matrices: DtwMatrices, out_dir: Path = RESULTS_EXPORTS) -> None:
    """Сохраняет DTW-матрицы (2D, только φ, только λ) как CSV — нормированные и «сырые»."""
    _matrix_to_frame(matrices.full_2d).to_csv(out_dir / "dtw_2d.csv", encoding="utf-8-sig")
    _matrix_to_frame(matrices.lat_only).to_csv(out_dir / "dtw_lat_only.csv", encoding="utf-8-sig")
    _matrix_to_frame(matrices.lon_only).to_csv(out_dir / "dtw_lon_only.csv", encoding="utf-8-sig")
    _matrix_to_frame(matrices.full_2d_raw).to_csv(out_dir / "dtw_2d_raw.csv", encoding="utf-8-sig")
    _matrix_to_frame(matrices.lat_only_raw).to_csv(out_dir / "dtw_lat_only_raw.csv", encoding="utf-8-sig")
    _matrix_to_frame(matrices.lon_only_raw).to_csv(out_dir / "dtw_lon_only_raw.csv", encoding="utf-8-sig")
