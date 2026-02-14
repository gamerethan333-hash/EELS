#!/usr/bin/env python3
"""Analyze and plot EELS spectrum images.

Supported input formats
- .npy: NumPy array with shape (ny, nx, nE)
- .npz: archive containing `data` with shape (ny, nx, nE), optional `energy`
- .dm3/.dm4: DigitalMicrograph spectrum-images (requires HyperSpy + RosettaSciIO)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def require_module(module_name: str, install_hint: str) -> None:
    """Raise a helpful error if a module is unavailable."""
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"Missing dependency `{module_name}`. Install with: {install_hint}")


def load_dm_spectrum_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load DM3/DM4 EELS spectrum-image via HyperSpy."""
    require_module("hyperspy", "pip install hyperspy rosettasciio")
    hs = importlib.import_module("hyperspy.api")

    signal = hs.load(str(path))
    if isinstance(signal, list):
        if len(signal) != 1:
            raise ValueError(f"Expected one signal in {path}, found {len(signal)}")
        signal = signal[0]

    data = np.asarray(signal.data, dtype=float)
    if not getattr(signal.axes_manager, "signal_axes", None):
        raise ValueError("DM file does not contain a signal axis")

    signal_axis = signal.axes_manager.signal_axes[0]
    data = np.moveaxis(data, signal_axis.index_in_array, -1)
    if data.ndim != 3:
        raise ValueError(f"Expected spectrum-image with shape (ny, nx, nE); got {data.shape}")

    energy = np.asarray(signal_axis.axis, dtype=float)
    if energy.size != data.shape[-1]:
        energy = np.arange(data.shape[-1], dtype=float)
    return data, energy


def load_spectrum_image(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load input SI and return (data, energy_axis)."""
    ext = path.suffix.lower()
    if ext == ".npy":
        data = np.load(path)
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array in {path}, got shape {data.shape}")
        energy = np.arange(data.shape[-1], dtype=float)
        return data.astype(float), energy

    if ext == ".npz":
        content = np.load(path)
        if "data" not in content:
            raise ValueError("NPZ file must contain a `data` array")
        data = content["data"]
        if data.ndim != 3:
            raise ValueError(f"Expected 3D `data`, got shape {data.shape}")
        energy = content["energy"] if "energy" in content else np.arange(data.shape[-1], dtype=float)
        return data.astype(float), np.asarray(energy, dtype=float)

    if ext in {".dm3", ".dm4"}:
        return load_dm_spectrum_image(path)

    raise ValueError("Unsupported file. Use .npy, .npz, .dm3, or .dm4")


def average_spectrum(data: np.ndarray) -> np.ndarray:
    return data.reshape(-1, data.shape[-1]).mean(axis=0)


def power_law_background(spectrum: np.ndarray, energy: np.ndarray, bg_start: float, bg_end: float) -> np.ndarray:
    """Fit I(E)=A*E^-r on the pre-edge window."""
    mask = (energy >= bg_start) & (energy <= bg_end)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Background window must include at least 3 channels")

    e = np.clip(energy[mask], 1e-6, None)
    i = np.clip(spectrum[mask], 1e-6, None)
    slope, intercept = np.polyfit(np.log(e), np.log(i), 1)
    return np.exp(intercept) * np.clip(energy, 1e-6, None) ** (-slope)


def integrated_edge_map(
    data: np.ndarray,
    energy: np.ndarray,
    int_start: float,
    int_end: float,
    bg_start: float,
    bg_end: float,
) -> np.ndarray:
    """Compute background-subtracted integrated edge map."""
    ny, nx, _ = data.shape
    out = np.zeros((ny, nx), dtype=float)
    int_mask = (energy >= int_start) & (energy <= int_end)
    if np.count_nonzero(int_mask) < 2:
        raise ValueError("Integration window must include at least 2 channels")

    for y in range(ny):
        for x in range(nx):
            spec = data[y, x, :]
            bg = power_law_background(spec, energy, bg_start, bg_end)
            out[y, x] = np.trapz((spec - bg)[int_mask], energy[int_mask])
    return out


def plot_results(energy: np.ndarray, avg: np.ndarray, bg: np.ndarray, edge_map: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(energy, avg, label="Average spectrum", lw=1.5)
    ax1.plot(energy, bg, label="Power-law background", ls="--")
    ax1.plot(energy, avg - bg, label="Background subtracted", alpha=0.8)
    ax1.set_xlabel("Energy loss (eV)")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title("Average EELS Spectrum")
    ax1.legend()
    ax1.grid(alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(output_dir / "average_spectrum.png", dpi=180)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(edge_map, cmap="inferno")
    ax2.set_title("Integrated Edge Map")
    ax2.set_xlabel("x (px)")
    ax2.set_ylabel("y (px)")
    cbar = fig2.colorbar(im, ax=ax2)
    cbar.set_label("Integrated intensity (a.u.)")
    fig2.tight_layout()
    fig2.savefig(output_dir / "integrated_edge_map.png", dpi=180)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and plot EELS spectrum-image")
    parser.add_argument("input", type=Path, help="Path to .npy, .npz, .dm3, or .dm4")
    parser.add_argument("--bg-start", type=float, required=True, help="Background fit start (eV)")
    parser.add_argument("--bg-end", type=float, required=True, help="Background fit end (eV)")
    parser.add_argument("--int-start", type=float, required=True, help="Edge integration start (eV)")
    parser.add_argument("--int-end", type=float, required=True, help="Edge integration end (eV)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for saved figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, energy = load_spectrum_image(args.input)

    avg = average_spectrum(data)
    bg = power_law_background(avg, energy, args.bg_start, args.bg_end)
    edge = integrated_edge_map(data, energy, args.int_start, args.int_end, args.bg_start, args.bg_end)

    print(f"Loaded SI shape: {data.shape}")
    print(f"Energy range: {energy.min():.3f} to {energy.max():.3f} eV")
    print(f"Edge map min/max: {edge.min():.4g} / {edge.max():.4g}")
    plot_results(energy, avg, bg, edge, args.output_dir)


if __name__ == "__main__":
    main()
