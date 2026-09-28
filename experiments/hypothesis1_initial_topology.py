"""
Hypothesis 1: The converged Physarum network is (approximately) independent
of the initial graph topology.

Biological motivation
---------------------
The actual slime mold starts from a contiguous foraging margin and resolves
into tubes. The mathematical model lets us ask: does the *same* attractor
emerge regardless of where we start? If yes, this means the network is a
true self-organized steady state of the local feedback rule, not an artifact
of any particular initial geometry.

Protocol
--------
Run the model on Tokyo with three different initial graphs:
    - Delaunay triangulation  (paper's default)
    - 8-nearest-neighbour graph
    - Complete graph
Each uses identical parameters (I_0=2.0, γ=1.8) and the same RNG seed.

Compare the final networks via:
    - Edge set Jaccard similarity (topology overlap)
    - Pearson correlation of edge-wise conductivity over the union edge set
    - The three quality metrics TL, MD, FT

If the converged networks are highly similar despite very different starts,
the hypothesis is supported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.physarum import (
    PhysarumConfig,
    run_simulation,
    final_network,
    build_delaunay_edges,
    build_complete_edges,
    build_grid_edges,
)
from src.tokyo_data import tokyo_coords
from src.metrics import build_graph, all_metrics
from src.viz import plot_network

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def edge_set_jaccard(edges_a: np.ndarray, edges_b: np.ndarray) -> float:
    """Jaccard similarity between two edge sets (treated as undirected)."""
    set_a = {tuple(sorted(int(x) for x in e)) for e in edges_a}
    set_b = {tuple(sorted(int(x) for x in e)) for e in edges_b}
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def conductivity_correlation(
    edges_a: np.ndarray, D_a: np.ndarray,
    edges_b: np.ndarray, D_b: np.ndarray,
) -> float:
    """Pearson correlation of D over the union edge set (0 if missing).

    For edges that exist in only one network, the other network's D is 0.
    """
    map_a = {tuple(sorted(int(x) for x in e)): float(d)
             for e, d in zip(edges_a, D_a)}
    map_b = {tuple(sorted(int(x) for x in e)): float(d)
             for e, d in zip(edges_b, D_b)}
    union = sorted(set(map_a) | set(map_b))
    vec_a = np.array([map_a.get(e, 0.0) for e in union])
    vec_b = np.array([map_b.get(e, 0.0) for e in union])
    if vec_a.std() < 1e-12 or vec_b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(vec_a, vec_b)[0, 1])


def main():
    print("=" * 70)
    print("Hypothesis 1: convergence is independent of initial graph topology")
    print("=" * 70)

    coords, _ = tokyo_coords()
    cfg = PhysarumConfig(I0=2.0, gamma=1.8, dt=0.05, n_steps=6000, seed=0)

    initial_builders = {
        "Delaunay (paper default)": build_delaunay_edges,
        "8-NN graph":               lambda c: build_grid_edges(c, k=8),
        "Complete graph":           build_complete_edges,
    }

    results = {}
    for label, builder in initial_builders.items():
        edge_index, edge_length = builder(coords)
        state = run_simulation(
            coords,
            edge_index=edge_index,
            edge_length=edge_length,
            config=cfg,
        )
        net = final_network(state, threshold=cfg.prune_threshold)
        G = build_graph(coords, net["edge_index"], net["edge_length"])
        m = all_metrics(G)
        results[label] = {
            "edge_index": net["edge_index"],
            "edge_length": net["edge_length"],
            "D": net["D"],
            "metrics": m,
            "n_edges": len(net["edge_index"]),
        }
        print(f"{label:<28}: edges={len(net['edge_index']):>3}, "
              f"TL_mst={m['TL']/3.172:.3f}, MD_mst={m['MD']/0.526:.3f}, "
              f"FT={m['FT']:.3f}")

    print("\n--- Pairwise comparison of converged networks ---")
    labels = list(results.keys())
    print(f"{'pair':<55} {'Jaccard':>10} {'D-corr':>10}")
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            ra, rb = results[a], results[b]
            jac = edge_set_jaccard(ra["edge_index"], rb["edge_index"])
            corr = conductivity_correlation(
                ra["edge_index"], ra["D"],
                rb["edge_index"], rb["D"],
            )
            print(f"{a[:24]+' vs '+b[:24]:<55} {jac:>10.3f} {corr:>10.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, label in zip(axes, labels):
        r = results[label]
        plot_network(
            coords, r["edge_index"], edge_weight=r["D"],
            ax=ax, title=f"{label}\n({r['n_edges']} edges, "
                         f"FT={r['metrics']['FT']:.2f})",
            linewidth_scale=3.5,
        )
    fig.suptitle(
        f"Hypothesis 1: converged network from different initial graphs\n"
        f"(I₀={cfg.I0}, γ={cfg.gamma}, T={cfg.n_steps * cfg.dt:.0f})",
        fontsize=12,
    )
    plt.tight_layout()
    save_path = FIG_DIR / "hypothesis1_initial_topology.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {save_path}")


if __name__ == "__main__":
    main()
