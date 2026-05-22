"""Plotting and analysis utilities for OCR evaluation results.

Loads aggregate and per-image results produced by the evaluation
pipeline and generates comparative visualisations across OCR modules.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.evaluation_pipeline import ALL_TAGS_KEY, AggregateResult
from evaluation.metrics import Metric

logger = logging.getLogger(__name__)


def plot_metric_comparison(
    aggregate: AggregateResult,
    stat: str = "mean",
    ci_level: str = "95",
    save_path: str | Path | None = None,
    tag: str = ALL_TAGS_KEY,
) -> plt.Figure:
    """Bar chart comparing all modules across every metric.

    Each metric gets a group of bars, one per module. When ``stat`` is
    ``mean``, asymmetric error bars are drawn from the percentile-based
    confidence interval stored under the requested ``ci_level``.

    Args:
        aggregate: An :class:`AggregateResult` (e.g. from
            ``AggregateResult.from_path(...)`` or
            :class:`EvaluationResult.aggregate` of a live run).
        stat: Which statistic to plot (mean, median, min, max).
        ci_level: Confidence interval key to use for error bars
            (default ``"95"``).
        save_path: If given, save the figure to this path.
        tag: Which tag's slice to plot. Defaults to ``ALL_TAGS_KEY``
            (aggregated across all images).

    Returns:
        The matplotlib Figure.

    Raises:
        KeyError: If ``tag`` is not present in ``aggregate``.
    """
    view = aggregate.view_for_tag(tag)

    x = np.arange(len(view.metrics))
    width = 0.8 / len(view.labels)

    fig, ax = plt.subplots(figsize=(max(10, len(view.metrics) * 2), 6))

    for i, label in enumerate(view.labels):
        values = np.array([view.data[label][m][stat] for m in view.metrics])
        yerr = None
        if stat == "mean":
            ci_bounds = [view.data[label][m].get("ci", {}).get(ci_level) for m in view.metrics]
            if all(b is not None for b in ci_bounds):
                lower = np.array([values[j] - b[0] for j, b in enumerate(ci_bounds)])
                upper = np.array([b[1] - values[j] for j, b in enumerate(ci_bounds)])
                yerr = np.array([lower, upper])
        ax.bar(x + i * width, values, width, label=label, yerr=yerr, capsize=3)

    ax.set_xticks(x + width * (len(view.labels) - 1) / 2)
    ax.set_xticklabels([Metric._registry[m].display_name for m in view.metrics], rotation=30, ha="right")
    ax.set_ylabel(stat.capitalize())
    ax.set_title(f"OCR Module Comparison ({stat})")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def plot_radar(
    aggregate: AggregateResult,
    stat: str = "mean",
    save_path: str | Path | None = None,
    tag: str = ALL_TAGS_KEY,
) -> plt.Figure:
    """Radar (spider) chart comparing modules across all metrics.

    All values are clipped to [0, 1] for display so the chart axes
    remain comparable.

    Args:
        aggregate: An :class:`AggregateResult` (e.g. from
            ``AggregateResult.from_path(...)`` or
            :class:`EvaluationResult.aggregate` of a live run).
        stat: Which statistic to plot (mean, median, min, max).
        save_path: If given, save the figure to this path.
        tag: Which tag's slice to plot. Defaults to ``ALL_TAGS_KEY``
            (aggregated across all images).

    Returns:
        The matplotlib Figure.

    Raises:
        KeyError: If ``tag`` is not present in ``aggregate``.
    """
    view = aggregate.view_for_tag(tag)
    n_metrics = len(view.metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for label in view.labels:
        values = [min(view.data[label][m][stat], 1.0) for m in view.metrics]
        values.append(values[0])
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])

    label_radius = 1.18
    display_names = [Metric._registry[m].display_name for m in view.metrics]
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
    aggregate: AggregateResult,
    save_path: str | Path | None = None,
    tag: str = ALL_TAGS_KEY,
) -> plt.Figure:
    """Box-style range plot showing min, mean, and max per metric per module.

    Args:
        aggregate: An :class:`AggregateResult` (e.g. from
            ``AggregateResult.from_path(...)`` or
            :class:`EvaluationResult.aggregate` of a live run).
        save_path: If given, save the figure to this path.
        tag: Which tag's slice to plot. Defaults to ``ALL_TAGS_KEY``
            (aggregated across all images).

    Returns:
        The matplotlib Figure.

    Raises:
        KeyError: If ``tag`` is not present in ``aggregate``.
    """
    view = aggregate.view_for_tag(tag)

    fig, axes = plt.subplots(1, len(view.metrics), figsize=(4 * len(view.metrics), 5), sharey=False)
    if len(view.metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, view.metrics, strict=True):
        means = [view.data[label][metric]["mean"] for label in view.labels]
        mins = [view.data[label][metric]["min"] for label in view.labels]
        maxs = [view.data[label][metric]["max"] for label in view.labels]

        x = np.arange(len(view.labels))
        ax.errorbar(
            x,
            means,
            yerr=[np.array(means) - np.array(mins), np.array(maxs) - np.array(means)],
            fmt="o",
            capsize=5,
            capthick=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(view.labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(Metric._registry[metric].display_name, fontsize=10)

    fig.suptitle("Metric Ranges (min / mean / max)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def summary_table(aggregate: AggregateResult, stat: str = "mean", tag: str = ALL_TAGS_KEY) -> str:
    """Return a formatted text table of aggregate results.

    Args:
        aggregate: An :class:`AggregateResult` (e.g. from
            ``AggregateResult.from_path(...)`` or
            :class:`EvaluationResult.aggregate` of a live run).
        stat: Which statistic to tabulate.
        tag: Which tag's slice to tabulate. Defaults to ``ALL_TAGS_KEY``
            (aggregated across all images).

    Returns:
        A multi-line string table suitable for printing.

    Raises:
        KeyError: If ``tag`` is not present in ``aggregate``.
    """
    view = aggregate.view_for_tag(tag)

    col_width = max(len(Metric._registry[m].display_name) for m in view.metrics) + 2
    label_width = max(len(label) for label in view.labels) + 2

    header = " " * label_width + "".join(Metric._registry[m].display_name.rjust(col_width) for m in view.metrics)
    lines = [header, "-" * len(header)]

    for label in view.labels:
        row = label.ljust(label_width) + "".join(
            f"{view.data[label][m][stat]:.4f}".rjust(col_width) for m in view.metrics
        )
        lines.append(row)

    return "\n".join(lines)
