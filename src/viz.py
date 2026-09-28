"""
Visualization helpers for the Physarum network experiment.

Two main entry points:

  plot_network(...)      -> static snapshot of one network
  plot_evolution(...)    -> grid of snapshots showing how D_ij evolves over time
  plot_pareto(...)       -> TL vs MD scatter (the paper's Fig. 3A style)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
import numpy as np
import networkx as nx

from .metrics import build_graph, minimum_spanning_tree, delaubay_graph


# A consistent colour scheme across all figures.
COLOR_PHYSARUM = "#d62728"   # red
COLOR_RAIL     = "#2ca02c"   # green
COLOR_MST      = "#1f77b4"   # blue
COLOR_DELAUNAY = "#ff7f0e"   # orange
COLOR_NODE     = "#222222"


def plot_network(
    coords: np.ndarray,
    edge_index: np.ndarray,
    *,
    edge_weight: np.ndarray | None = None,
    edge_length: np.ndarray | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    color: str = COLOR_PHYSARUM,
    show_nodes: bool = True,
    linewidth_scale: float = 2.0,
) -> Axes:
    """Draw a network. Edge width can scale with `edge_weight` (e.g. D_ij)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    if len(edge_index) > 0:
        if edge_weight is not None:
            w = np.asarray(edge_weight)
            w = w / max(w.max(), 1e-9)
            lw = linewidth_scale * w
        else:
            lw = np.full(len(edge_index), linewidth_scale * 0.5)

        for (a, b), line_w in zip(edge_index, lw):
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color=color,
                linewidth=line_w,
                solid_capstyle="round",
                alpha=0.85,
            )

    if show_nodes:
        ax.scatter(coords[:, 0], coords[:, 1], s=40,
                   c=COLOR_NODE, zorder=5, edgecolors="white", linewidths=0.8)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11)
    return ax


def plot_overlay(
    coords: np.ndarray,
    physarum_edges: np.ndarray,
    *,
    mst_edges: np.ndarray | None = None,
    title: str = "Physarum solution vs MST",
    save_path: str | Path | None = None,
) -> None:
    """Overlay Physarum solution on top of the MST baseline."""
    fig, ax = plt.subplots(figsize=(7, 7))
    if mst_edges is not None:
        for a, b in mst_edges:
            ax.plot(
                [coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                color=COLOR_MST, linewidth=3, alpha=0.4,
                linestyle="--", label=None,
            )
            # remove duplicate legend entries
        ax.plot([], [], color=COLOR_MST, linewidth=3, alpha=0.4,
                linestyle="--", label="MST (baseline)")
    for a, b in physarum_edges:
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            color=COLOR_PHYSARUM, linewidth=2.2, alpha=0.9,
        )
    ax.plot([], [], color=COLOR_PHYSARUM, linewidth=2.2,
            alpha=0.9, label="Physarum solution")
    ax.scatter(coords[:, 0], coords[:, 1], s=40,
               c=COLOR_NODE, zorder=5, edgecolors="white", linewidths=0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(title, fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_evolution(
    state_history: list[dict],
    coords: np.ndarray,
    edge_index: np.ndarray,
    snapshot_indices: list[int],
    *,
    title: str = "Network evolution",
    save_path: str | Path | None = None,
    prune_threshold: float = 1e-3,
) -> None:
    """Plot a grid of snapshots showing how the network evolves over time."""
    n_snapshots = len(snapshot_indices)
    cols = min(4, n_snapshots)
    rows = (n_snapshots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, idx in enumerate(snapshot_indices):
        snap = state_history[idx]
        alive = snap["D"] > prune_threshold
        ax = axes[i]
        plot_network(
            coords,
            edge_index[alive],
            edge_weight=snap["D"][alive],
            ax=ax,
            title=f"step {snap['step']}",
            linewidth_scale=3.0,
        )

    # Hide unused subplots
    for j in range(n_snapshots, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto(
    results: list[dict],
    *,
    reference: dict | None = None,
    save_path: str | Path | None = None,
    title: str = "Cost vs transport efficiency",
) -> None:
    """TL_mst vs MD_mst scatter (paper Fig. 3A style).

    `results` is a list of dicts each with keys 'TL_mst', 'MD_mst', 'label',
    'color', 'marker'. Optional `reference` can add MST/rail/Delaunay points.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Group by label so we can plot each cluster with one legend entry.
    groups: dict[str, list[dict]] = {}
    for r in results:
        groups.setdefault(r["label"], []).append(r)

    for label, items in groups.items():
        xs = [r["TL_mst"] for r in items]
        ys = [r["MD_mst"] for r in items]
        color = items[0].get("color", "gray")
        marker = items[0].get("marker", "o")
        ax.scatter(xs, ys, c=color, marker=marker, s=80,
                   label=label, alpha=0.8, edgecolors="white", linewidths=0.6)

    if reference:
        for label, (x, y, color, marker) in reference.items():
            ax.scatter([x], [y], c=color, marker=marker, s=140,
                       label=label, edgecolors="black", linewidths=1.0, zorder=10)

    ax.set_xlabel(r"Total Length  $TL_{MST}$  (cost, higher = more expensive)", fontsize=11)
    ax.set_ylabel(r"Mean Distance  $MD_{MST}$  (lower = more efficient)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # MST is at (1, 1)
    ax.scatter([1.0], [1.0], c=COLOR_MST, marker="*", s=300,
               edgecolors="black", linewidths=1.0, zorder=10, label="MST")
    ax.annotate("MST", (1.0, 1.0), xytext=(8, -15),
                textcoords="offset points", fontsize=10, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_robustness_curve(
    failure_curves: dict[str, list[tuple[int, float]]],
    *,
    save_path: str | Path | None = None,
    title: str = "Robustness under progressive edge failure",
) -> None:
    """Plot fraction-of-nodes-reachable vs number of removed edges.

    `failure_curves` maps a label like "Physarum" / "MST" / "Delaunay" to a
    list of (n_removed, fraction_reachable) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, pts in failure_curves.items():
        pts = sorted(pts)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=label, linewidth=2)
    ax.set_xlabel("Number of edges removed", fontsize=11)
    ax.set_ylabel("Fraction of node pairs still connected", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Quick smoke test: visualize a random point set + MST.
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((15, 2))
    mst = minimum_spanning_tree(coords)
    ei = np.array([(u, v) for u, v in mst.edges()], dtype=np.int64)
    plot_network(coords, ei, title="MST smoke test")
    out = Path(__file__).resolve().parent.parent / "figures" / "viz_smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved {out}")
