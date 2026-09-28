"""
Bonus: Steiner point emergence.

The paper highlights that Physarum creates *Steiner points* - extra junction
nodes that do not coincide with any food source but reduce the total length
of the network. The classic example: for three cities forming an equilateral
triangle, the optimal connection is a Y-shape with a Steiner point at the
centre, not a triangle.

Our base Tokyo experiment does not allow Steiner points because the only
nodes in the graph are the cities themselves. Here we add a fine triangular
mesh of "candidate Steiner points" filling the convex hull of the cities.
Only the cities act as source/sink in the flux simulation, but the mesh
nodes can survive in the final network if the dynamics favour them.

Hypothesis: With Steiner candidates available, the converged network should
be SHORTER (lower TL/TL_MST) than without, because the model can route
through junction points that lower the total length.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.physarum import PhysarumConfig, run_simulation, final_network
from src.tokyo_data import tokyo_coords
from src.metrics import (
    build_graph, minimum_spanning_tree, all_metrics,
)
from src.viz import plot_network

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def add_steiner_lattice(
    city_coords: np.ndarray, spacing: float = 0.08
) -> tuple[np.ndarray, np.ndarray]:
    """Build a coordinate array = cities + triangular mesh filling convex hull.

    Returns (all_coords, is_city_mask) where is_city_mask[i] is True if node i
    is an original city, False if it's a Steiner candidate.
    """
    n_city = len(city_coords)
    x_min, y_min = city_coords.min(axis=0)
    x_max, y_max = city_coords.max(axis=0)

    xs = np.arange(x_min - spacing, x_max + spacing, spacing)
    ys = np.arange(y_min - spacing, y_max + spacing, spacing * 0.866)
    pts = [(x, y) for x in xs for y in ys]
    mesh = np.array(pts)

    from matplotlib.path import Path
    hull = Delaunay(city_coords)
    inside = hull.find_simplex(mesh) >= 0
    mesh = mesh[inside]

    all_coords = np.vstack([city_coords, mesh])
    is_city = np.zeros(len(all_coords), dtype=bool)
    is_city[:n_city] = True
    return all_coords, is_city


def main():
    print("=" * 70)
    print("Bonus: Steiner-point emergence experiment")
    print("=" * 70)

    city_coords, names = tokyo_coords()
    all_coords, is_city = add_steiner_lattice(city_coords, spacing=0.07)
    n_city = len(city_coords)
    n_total = len(all_coords)
    print(f"Cities: {n_city}, Steiner candidates: {n_total - n_city}")

    cfg = PhysarumConfig(
        I0=2.0, gamma=1.8, dt=0.05, n_steps=4000, seed=0,
    )
    state = run_simulation(all_coords, config=cfg)
    net = final_network(state, threshold=cfg.prune_threshold)

    G_full = build_graph(all_coords, net["edge_index"], net["edge_length"])
    m_full = all_metrics(G_full)

    # Identify which edges connect only to cities vs involve Steiner nodes.
    is_city_arr = is_city[net["edge_index"]]
    both_cities = is_city_arr.all(axis=1)
    has_steiner = ~both_cities

    # The "city-only" network: keep only edges between two city nodes.
    city_only_ei = net["edge_index"][both_cities]
    city_only_L  = net["edge_length"][both_cities]
    G_city = build_graph(city_coords, city_only_L.shape[0] and city_only_ei,
                         city_only_L)
    # Add straight-line bypasses for any pair of cities that are NOT directly
    # connected in the city-only view but ARE connected through Steiner nodes
    # in the full network. For metric purposes we treat the "with Steiner"
    # network as the union graph evaluated on all node positions.
    m_city = all_metrics(G_city)

    print(f"\nWith Steiner candidates (n_nodes = {n_total}):")
    print(f"  edges total: {G_full.number_of_edges()}")
    print(f"  edges involving a Steiner point: {int(has_steiner.sum())}")
    print(f"  TL = {m_full['TL']:.3f}")
    print(f"  MD = {m_full['MD']:.3f}")
    print(f"  FT = {m_full['FT']:.3f}")

    mst = minimum_spanning_tree(city_coords)
    mst_m = all_metrics(mst)
    print(f"\nMST (city-only): TL={mst_m['TL']:.3f}, MD={mst_m['MD']:.3f}")
    print(f"With-Steiner / MST:")
    print(f"  TL/TL_MST = {m_full['TL']/mst_m['TL']:.3f}")
    print(f"  MD/MD_MST = {m_full['MD']/mst_m['MD']:.3f}")
    print(f"  FT        = {m_full['FT']:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    plot_network(
        all_coords, net["edge_index"][has_steiner], edge_weight=net["D"][has_steiner],
        ax=axes[0], color="#9467bd", show_nodes=False,
        title=f"Edges through Steiner points ({int(has_steiner.sum())} edges)",
        linewidth_scale=2.5,
    )
    axes[0].scatter(city_coords[:, 0], city_coords[:, 1],
                    s=40, c="#222", zorder=5, edgecolors="white", linewidths=0.8)
    plot_network(
        all_coords, net["edge_index"], edge_weight=net["D"],
        ax=axes[1], color="#d62728", show_nodes=False,
        title=f"Full converged network ({G_full.number_of_edges()} edges)",
        linewidth_scale=2.5,
    )
    axes[1].scatter(city_coords[:, 0], city_coords[:, 1],
                    s=40, c="#222", zorder=5, edgecolors="white", linewidths=0.8)
    fig.suptitle(
        "Bonus: Steiner points emerge when candidate junctions are available",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = FIG_DIR / "bonus_steiner_points.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
