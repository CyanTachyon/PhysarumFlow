"""
Hypothesis 3: Physarum networks are more robust than MST under both random
failures AND targeted (highest-flow) attacks.

Motivation
----------
The TL-MD-FT metrics give a static snapshot of network quality. But real
infrastructure must survive progressive degradation: edges fail one at a
time, either at random (wear-and-tear) or by importance (targeted attacks
on the busiest segments). A truly well-designed network should retain
connectivity for as long as possible under both modes.

The paper's "FT" metric counts only single-edge failures. Here we extend
this idea by progressively removing edges and measuring the *fraction of
node pairs that remain mutually reachable* as a function of how many edges
have been removed. This produces a robustness curve that distinguishes
networks which only LOOK robust from those that degrade gracefully.

Protocol
--------
For each of {MST, Delaunay, Physarum (I_0=2, γ=1.8)}:
    Mode "random"   - remove edges in a random order
    Mode "targeted" - always remove the edge with the highest current
                      betweenness centrality (i.e. the edge whose loss
                      does the most damage right now)
After each removal, compute the fraction of unordered node pairs that are
still in the same connected component.

Result is averaged over 20 random seeds for the random-removal mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.physarum import (
    PhysarumConfig, run_simulation, final_network, build_delaunay_edges,
)
from src.tokyo_data import tokyo_coords
from src.metrics import build_graph, minimum_spanning_tree, delaubay_graph
from src.viz import (
    COLOR_PHYSARUM, COLOR_MST, COLOR_DELAUNAY,
)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def connected_pair_fraction(G: nx.Graph) -> float:
    """Fraction of unordered node pairs that share a connected component."""
    n = G.number_of_nodes()
    if n < 2:
        return 1.0
    components = list(nx.connected_components(G))
    total_pairs = n * (n - 1) / 2
    reachable = sum(len(c) * (len(c) - 1) / 2 for c in components)
    return reachable / total_pairs


def random_failure_curve(G: nx.Graph, n_seeds: int = 20) -> np.ndarray:
    """Average connectivity-vs-removals curve under random edge removal."""
    edges = list(G.edges())
    m = len(edges)
    curves = np.zeros((n_seeds, m + 1))
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        order = rng.permutation(m)
        H = G.copy()
        curves[s, 0] = connected_pair_fraction(H)
        for k, idx in enumerate(order, start=1):
            u, v = edges[idx]
            if H.has_edge(u, v):
                H.remove_edge(u, v)
            curves[s, k] = connected_pair_fraction(H)
    return curves.mean(axis=0)


def targeted_failure_curve(G: nx.Graph) -> np.ndarray:
    """Connectivity curve when always removing the edge with highest
    betweenness centrality (most damage-per-edge)."""
    H = G.copy()
    m = H.number_of_edges()
    curve = np.zeros(m + 1)
    curve[0] = connected_pair_fraction(H)
    for k in range(1, m + 1):
        if H.number_of_edges() == 0:
            curve[k] = 0.0
            continue
        bc = nx.edge_betweenness_centrality(H, weight="length")
        target_edge = max(bc.items(), key=lambda kv: kv[1])[0]
        H.remove_edge(*target_edge)
        curve[k] = connected_pair_fraction(H)
    return curve


def area_under_curve(curve: np.ndarray) -> float:
    """Normalised area under a robustness curve (in [0, 1])."""
    m = len(curve) - 1
    if m == 0:
        return 0.0
    return float(np.trapezoid(curve, dx=1) / m)


def main():
    print("=" * 70)
    print("Hypothesis 3: progressive failure comparison")
    print("=" * 70)

    coords, _ = tokyo_coords()

    cfg = PhysarumConfig(I0=2.0, gamma=1.8, dt=0.05, n_steps=5000, seed=0)
    state = run_simulation(coords, config=cfg)
    net = final_network(state, threshold=cfg.prune_threshold)

    phys_G = build_graph(coords, net["edge_index"], net["edge_length"])
    mst_G = minimum_spanning_tree(coords)
    del_G = delaubay_graph(coords)

    networks = {
        "MST (tree)":      mst_G,
        "Physarum":        phys_G,
        "Delaunay (dense)": del_G,
    }
    colors = {
        "MST (tree)":      COLOR_MST,
        "Physarum":        COLOR_PHYSARUM,
        "Delaunay (dense)": COLOR_DELAUNAY,
    }

    print(f"\n{'Network':<18} {'edges':>6} {'AU-random':>12} {'AU-targeted':>14}")
    print("-" * 52)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    summary = {}

    for label, G in networks.items():
        rand_curve = random_failure_curve(G, n_seeds=20)
        targ_curve = targeted_failure_curve(G)
        au_rand = area_under_curve(rand_curve)
        au_targ = area_under_curve(targ_curve)
        summary[label] = {
            "n_edges": G.number_of_edges(),
            "rand_curve": rand_curve,
            "targ_curve": targ_curve,
            "AU_random": au_rand,
            "AU_targeted": au_targ,
        }
        print(f"{label:<18} {G.number_of_edges():>6} "
              f"{au_rand:>12.3f} {au_targ:>14.3f}")

        xs = np.arange(len(rand_curve))
        norm_x = xs / xs[-1]
        axes[0].plot(norm_x, rand_curve, linewidth=2.2,
                     color=colors[label], label=label)
        axes[1].plot(np.arange(len(targ_curve)) / (len(targ_curve) - 1),
                     targ_curve, linewidth=2.2,
                     color=colors[label], label=label)

    axes[0].set_xlabel("fraction of edges removed (random)", fontsize=11)
    axes[0].set_ylabel("fraction of node pairs still connected", fontsize=11)
    axes[0].set_title("Random failures", fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.02, 1.04)
    axes[0].legend(loc="upper right", fontsize=10)

    axes[1].set_xlabel("fraction of edges removed (targeted)", fontsize=11)
    axes[1].set_ylabel("fraction of node pairs still connected", fontsize=11)
    axes[1].set_title("Targeted attacks (highest betweenness first)", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(-0.02, 1.04)
    axes[1].legend(loc="upper right", fontsize=10)

    fig.suptitle("Hypothesis 3: progressive edge-failure robustness on Tokyo",
                 fontsize=13)
    plt.tight_layout()
    save_path = FIG_DIR / "hypothesis3_robustness.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {save_path}")

    print("\n--- Interpretation ---")
    p = summary["Physarum"]
    m = summary["MST (tree)"]
    print(f"Physarum / MST ratio of robust area:")
    print(f"  random mode:   {p['AU_random'] / max(m['AU_random'], 1e-9):.2f}x")
    print(f"  targeted mode: {p['AU_targeted'] / max(m['AU_targeted'], 1e-9):.2f}x")


if __name__ == "__main__":
    main()
