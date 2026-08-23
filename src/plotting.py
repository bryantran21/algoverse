"""Shared plot helpers.

House rules, enforced here so they cannot be forgotten per-figure:

- every layerwise curve carries a bootstrap CI band, because a bare mean line
  invites reading trends that the data does not support;
- every plot that has a chance level or a control condition draws it;
- figures are saved at dpi>=150 and the file path is returned, so a caller can
  commit it without guessing where it went.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NAVY, CRIMSON, STEEL, GREEN = "#1E2761", "#C0392B", "#4A6FA5", "#2E7D32"


def boot_ci(curves, n_boot: int = 1000, seed: int = 0, pct=(2.5, 97.5)):
    """Mean and bootstrap CI across items. `curves` is [n_items, n_layers]."""
    C = np.asarray(curves)
    rng = np.random.default_rng(seed)
    draws = C[rng.integers(0, len(C), size=(n_boot, len(C)))].mean(axis=1)
    return C.mean(axis=0), np.percentile(draws, pct[0], axis=0), np.percentile(draws, pct[1], axis=0)


def emergence(curve, thresh: float = 0.5) -> int | None:
    """First index where `curve` exceeds `thresh`, or None."""
    hits = np.where(np.asarray(curve) > thresh)[0]
    return int(hits[0]) if len(hits) else None


def curve_panel(ax, curves, color, marker="o-", label=None, seed=0):
    """Plot mean + CI band for one set of per-item curves."""
    m, lo, hi = boot_ci(curves, seed=seed)
    x = np.arange(len(m))
    ax.plot(x, m, marker, color=color, ms=4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    return m


def layerwise(series: list[tuple], title: str, ylabel: str, out_png: str | Path | None = None,
              chance: float | None = None, hlines: list[tuple] | None = None,
              xlabel: str = "layer (0-indexed over decoder blocks)", figsize=(8, 4.5)):
    """Standard layerwise figure.

    series: list of (curves, color, marker, label).
    hlines: list of (y, color, label) reference lines, e.g. clean/corrupt levels.
    """
    fig, ax = plt.subplots(figsize=figsize)
    means = [curve_panel(ax, c, col, mk, lbl) for c, col, mk, lbl in series]
    if chance is not None:
        ax.axhline(chance, color="gray", ls=":", label=f"chance ({chance:g})")
    for y, col, lbl in (hlines or []):
        ax.axhline(y, color=col, ls=":", label=lbl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
    plt.show()
    return means


def sweep_panel(ax, x, y, yerr=None, color=NAVY, marker="^-", label=None):
    """Font/DPI sweep point series with 95% CI error bars."""
    if yerr is None:
        ax.plot(x, y, marker, color=color, label=label)
    else:
        ax.errorbar(x, y, yerr=yerr, fmt=marker, color=color, capsize=4, label=label)


def ci_from_sd(sd, n, z: float = 1.96):
    """Half-width of a normal-approximation CI from a stored sd and n."""
    return z * np.asarray(sd) / np.sqrt(np.asarray(n))
