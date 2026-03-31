"""Matplotlib theme presets for boostwatch visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt

THEMES = {
    "default": {
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#cccccc",
        "axes.labelcolor": "#333333",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "grid.color": "#e5e5e5",
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "patch.edgecolor": "#cccccc",
        "font.family": "sans-serif",
    },
    "dark": {
        "axes.facecolor": "#1e1e2e",
        "figure.facecolor": "#181825",
        "axes.edgecolor": "#45475a",
        "axes.labelcolor": "#cdd6f4",
        "axes.titlecolor": "#cdd6f4",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.color": "#a6adc8",
        "ytick.color": "#a6adc8",
        "text.color": "#cdd6f4",
        "grid.color": "#313244",
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
        "font.family": "sans-serif",
    },
    "minimal": {
        "axes.facecolor": "#fafafa",
        "figure.facecolor": "#ffffff",
        "axes.edgecolor": "#dddddd",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#222222",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "grid.color": "#eeeeee",
        "grid.linestyle": "-",
        "lines.linewidth": 1.5,
        "font.family": "sans-serif",
    },
}


def apply_theme(name: str = "default") -> None:
    """Apply a boostwatch matplotlib theme globally.

    This updates ``matplotlib.rcParams`` for all subsequent plots in the
    current session.  Call ``matplotlib.rcdefaults()`` to reset.

    Args:
        name: Theme name — one of ``"default"``, ``"dark"``, or
              ``"minimal"``.

    Raises:
        ValueError: If the requested theme name is not recognised.
    """
    if name not in THEMES:
        raise ValueError(
            "Unknown theme '{}'. Available themes: {}.".format(
                name, ", ".join(sorted(THEMES.keys()))
            )
        )
    plt.rcParams.update(THEMES[name])
