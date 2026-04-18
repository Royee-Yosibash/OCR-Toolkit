"""Plotting and analysis utilities for OCR evaluation results.

Loads aggregate and per-image results produced by the evaluation
pipeline and generates comparative visualisations across OCR modules.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluation_pipeline import AGGREGATE_RESULTS_FILE
from evaluation.metrics import metric_class_display_name
from utils.json_utils import load_json

logger = logging.getLogger(__name__)


def load_aggregate(results_dir: str | Path) -> dict:
    """Load the aggregate results JSON from an evaluation output directory.

    Args:
        results_dir: Path to the evaluation output directory.

    Returns:
        The parsed aggregate results dict keyed by module label.
    """
    return load_json(Path(results_dir) / AGGREGATE_RESULTS_FILE)


def plot_metric_comparison(
    aggregate: dict,
    stat: str = "mean",
    ci_level: str = "95",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing all modules across every metric.

    Each metric gets a group of bars, one per module. When ``stat`` is
    ``mean``, asymmetric error bars are drawn from the percentile-based
    confidence interval stored under the requested ``ci_level``.

    Args:
        aggregate: Aggregate results dict from ``load_aggregate``.
        stat: Which statistic to plot (mean, median, min, max).
        ci_level: Confidence interval key to use for error bars
            (default ``"95"``).
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    labels = list(aggregate.keys())
    metrics = list(next(iter(aggregate.values())).keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 2), 6))

    for i, label in enumerate(labels):
        values = np.array([aggregate[label][m][stat] for m in metrics])
        yerr = None
        if stat == "mean":
            ci_bounds = [aggregate[label][m].get("ci", {}).get(ci_level) for m in metrics]
            if all(b is not None for b in ci_bounds):
                lower = np.array([values[j] - b[0] for j, b in enumerate(ci_bounds)])
                upper = np.array([b[1] - values[j] for j, b in enumerate(ci_bounds)])
                yerr = np.array([lower, upper])
        ax.bar(x + i * width, values, width, label=label, yerr=yerr, capsize=3)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([metric_class_display_name(m) for m in metrics], rotation=30, ha="right")
    ax.set_ylabel(stat.capitalize())
    ax.set_title(f"OCR Module Comparison ({stat})")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_radar(
    aggregate: dict,
    stat: str = "mean",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Radar (spider) chart comparing modules across all metrics.

    All values are clipped to [0, 1] for display so the chart axes
    remain comparable.

    Args:
        aggregate: Aggregate results dict from ``load_aggregate``.
        stat: Which statistic to plot (mean, median, min, max).
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    labels = list(aggregate.keys())
    metrics = list(next(iter(aggregate.values())).keys())
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for label in labels:
        values = [min(aggregate[label][m][stat], 1.0) for m in metrics]
        values.append(values[0])
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    label_radius = 1.18
    display_names = [metric_class_display_name(m) for m in metrics]
    for angle, name in zip(angles[:-1], display_names, strict=True):
        angle_deg = np.degrees(angle)
        rotation = angle_deg - 90 if angle_deg <= 180 else angle_deg + 90
        ha = "center"
        if not (np.isclose(angle, 0) or np.isclose(angle, np.pi)):
            ha = "left" if 0 < angle < np.pi else "right"
        ax.text(
            angle,
            label_radius,
            name,
            size=9,
            ha=ha,
            va="center",
            rotation=rotation,
            rotation_mode="anchor",
        )

    ax.set_ylim(0, 1.05)
    ax.set_title(f"OCR Radar Chart ({stat})", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_stat_range(
    aggregate: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Box-style range plot showing min, mean, and max per metric per module.

    Args:
        aggregate: Aggregate results dict from ``load_aggregate``.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Figure.
    """
    labels = list(aggregate.keys())
    metrics = list(next(iter(aggregate.values())).keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=True):
        means = [aggregate[label][metric]["mean"] for label in labels]
        mins = [aggregate[label][metric]["min"] for label in labels]
        maxs = [aggregate[label][metric]["max"] for label in labels]

        x = np.arange(len(labels))
        ax.errorbar(
            x,
            means,
            yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            fmt="o",
            capsize=5,
            capthick=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric_class_display_name(metric), fontsize=10)

    fig.suptitle("Metric Ranges (min / mean / max)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def summary_table(aggregate: dict, stat: str = "mean") -> str:
    """Return a formatted text table of aggregate results.

    Args:
        aggregate: Aggregate results dict from ``load_aggregate``.
        stat: Which statistic to tabulate.

    Returns:
        A multi-line string table suitable for printing.
    """
    labels = list(aggregate.keys())
    metrics = list(next(iter(aggregate.values())).keys())

    col_width = max(len(metric_class_display_name(m)) for m in metrics) + 2
    label_width = max(len(label) for label in labels) + 2

    header = " " * label_width + "".join(metric_class_display_name(m).rjust(col_width) for m in metrics)
    lines = [header, "-" * len(header)]

    for label in labels:
        row = label.ljust(label_width) + "".join(f"{aggregate[label][m][stat]:.4f}".rjust(col_width) for m in metrics)
        lines.append(row)

    return "\n".join(lines)

