"""Построение графиков (matplotlib) по результатам анализа маршрутов."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import DtwMatrices, find_reference_flight
from config import COLORS, LABELS, LED, ORDER, R_KM, SVO, TOWNS
from dtw import dtw_distance
from geo import resample_track


def plot_trajectories(tracks: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    for flight_id in ORDER:
        track = tracks[flight_id]
        plt.plot(track.lon, track.lat, color=COLORS[flight_id], lw=1.6,
                  label=LABELS[flight_id], alpha=.85)
    plt.scatter([SVO[1]], [SVO[0]], c="k", marker="s", s=60)
    plt.annotate("SVO", (SVO[1], SVO[0]))
    plt.scatter([LED[1]], [LED[0]], c="k", marker="*", s=80)
    plt.annotate("LED", (LED[1], LED[0]))
    plt.xlabel("Долгота λ, °в.д.")
    plt.ylabel("Широта φ, °с.ш.")
    plt.title("Траектории рейса SU10 за 5 дней")
    plt.legend()
    plt.grid(alpha=.3)
    plt.gca().set_aspect(1 / np.cos(np.radians(58)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dependence(tracks: dict[str, pd.DataFrame], out_path: Path) -> None:
    plt.figure(figsize=(7.5, 6))
    for flight_id in ORDER:
        track = tracks[flight_id]
        plt.scatter(track.lat, track.lon, s=6, color=COLORS[flight_id], alpha=.45, label=LABELS[flight_id])
    all_lat = np.concatenate([tracks[f].lat for f in ORDER])
    all_lon = np.concatenate([tracks[f].lon for f in ORDER])
    slope, intercept = np.polyfit(all_lat, all_lon, 1)
    xs = np.linspace(all_lat.min(), all_lat.max(), 50)
    r2 = 1 - ((all_lon - (intercept + slope * all_lat)) ** 2).sum() / ((all_lon - all_lon.mean()) ** 2).sum()
    plt.plot(xs, intercept + slope * xs, "k--", lw=2,
             label=f"λ = {intercept:.1f}{slope:+.2f}·φ  (R²={r2:.2f})")
    plt.xlabel("Широта φ, °с.ш.")
    plt.ylabel("Долгота λ, °в.д.")
    plt.title("Зависимость долготы от широты")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_dtw_matrices(matrices: DtwMatrices, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    labels = [LABELS[f] for f in ORDER]
    panels = [
        (matrices.full_2d, "2D: φ и λ"),
        (matrices.lat_only, "1D: только φ"),
        (matrices.lon_only, "1D: только λ"),
    ]
    for ax, (matrix, title) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_xticks(range(len(ORDER)))
        ax.set_yticks(range(len(ORDER)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_deviation(tracks: dict[str, pd.DataFrame], out_path: Path, n_points: int = 200) -> str:
    """Строит отклонение каждого маршрута от эталона (медоида); возвращает id эталона."""
    tracks_km = {f: resample_track(tracks[f], n_points, in_km=True) for f in ORDER}
    reference = find_reference_flight(tracks_km)
    ref_track = tracks_km[reference]
    n_ref = len(ref_track)

    plt.figure(figsize=(11, 5))
    for flight_id in ORDER:
        if flight_id == reference:
            continue
        _, path = dtw_distance(ref_track, tracks_km[flight_id], want_path=True)
        deviation = np.zeros(n_ref)
        counts = np.zeros(n_ref)
        for i, j in path:
            deviation[i] += np.hypot(*(ref_track[i] - tracks_km[flight_id][j]))
            counts[i] += 1
        deviation /= np.maximum(counts, 1)
        plt.plot(np.arange(n_ref) / (n_ref - 1), deviation, color=COLORS[flight_id],
                  lw=2, label=LABELS[flight_id])
    plt.axhline(5, color="gray", ls="--", lw=1)
    plt.xlabel(f"Доля пути (0 = SVO … 1 = LED), эталон — {LABELS[reference]}")
    plt.ylabel("Отклонение от эталона, км")
    plt.title("Участки совпадения и расхождения маршрутов")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return reference


def plot_waypoints(tracks: dict[str, pd.DataFrame], visited: dict[str, list[str]], out_path: Path) -> None:
    plt.figure(figsize=(8, 9))
    for flight_id in ORDER:
        track = tracks[flight_id]
        plt.plot(track.lon, track.lat, lw=1.2, alpha=.6, color=COLORS[flight_id], label=LABELS[flight_id])
    for town, (t_lat, t_lon) in TOWNS.items():
        count = sum(town in visited[f] for f in ORDER)
        plt.scatter(t_lon, t_lat, s=40 + count * 25, c="black", zorder=5)
        plt.annotate(f"{town} ({count})", (t_lon, t_lat), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xlabel("Долгота λ")
    plt.ylabel("Широта φ")
    plt.title(f"Населённые пункты на маршрутах (в скобках — число дней), R={R_KM:.0f} км")
    plt.legend()
    plt.grid(alpha=.3)
    plt.gca().set_aspect(1 / np.cos(np.radians(58)))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
