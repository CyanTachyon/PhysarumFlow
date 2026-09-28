"""
Reproduction of Tero et al. (2010) Figs. 3 and 4.

Two outputs:

  1. Network evolution snapshots (paper Fig. 4 style): shows the plasmodium
     starting from a uniform mesh and pruning itself into a Tokyo-rail-like
     network.

  2. Parameter sweep over I_0 (paper Fig. 3 style): each I_0 produces one
     network; we plot TL_mst vs MD_mst and FT vs I_0. The result should show
     that increasing I_0 produces networks with more cross-links: better
     transport efficiency (lower MD) and better fault tolerance, but at
     higher cost (larger TL).
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
)
from src.tokyo_data import tokyo_coords
from src.metrics import (
    build_graph,
    minimum_spanning_tree,
    delaubay_graph,
    all_metrics,
)
from src.viz import (
    plot_network,
    plot_overlay,
    plot_evolution,
    plot_pareto,
    COLOR_PHYSARUM,
    COLOR_MST,
    COLOR_DELAUNAY,
)

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def experiment_evolution():
    """Run one Physarum simulation on Tokyo and save evolution snapshots."""
    print("=" * 60)
    print("Experiment 1: network evolution (paper Fig. 4 reproduction)")
    print("=" * 60)

    coords, names = tokyo_coords()
    print(f"Tokyo dataset: {len(coords)} nodes")

    # Delaunay initial mesh
    edge_index, edge_length = build_delaunay_edges(coords)
    print(f"Initial Delaunay mesh: {len(edge_index)} edges")

    # Take snapshots at steps 0, 200, 500, 1000, 2000, 4000
    cfg = PhysarumConfig(
        I0=2.0,
        gamma=1.8,
        dt=0.05,
        n_steps=5000,
        seed=0,
    )
    state = run_simulation(
        coords,
        edge_index=edge_index,
        edge_length=edge_length,
        config=cfg,
        snapshot_every=500,
    )
    print(f"Snapshots captured: {len(state.history)}")

    # Pick six evenly-spaced snapshots for the figure.
    snap_idx = [0, 1, 2, 4, 6, 9]
    snap_idx = [i for i in snap_idx if i < len(state.history)]
    save_path = FIG_DIR / "tokyo_evolution.png"
    plot_evolution(
        state.history,
        coords,
        edge_index,
        snap_idx,
        title=f"Physarum network evolution on Tokyo  (I₀={cfg.I0}, γ={cfg.gamma})",
        save_path=save_path,
        prune_threshold=cfg.prune_threshold,
    )
    print(f"Saved {save_path}")

    # Final network + MST overlay
    net = final_network(state, threshold=cfg.prune_threshold)
    mst = minimum_spanning_tree(coords)
    mst_edges = np.array([(u, v) for u, v in mst.edges()], dtype=np.int64)
    save_path = FIG_DIR / "tokyo_final_overlay.png"
    plot_overlay(
        coords,
        net["edge_index"],
        mst_edges=mst_edges,
        title=f"Final Physarum solution ({len(net['edge_index'])} edges) vs MST ({len(mst_edges)} edges)",
        save_path=save_path,
    )
    print(f"Saved {save_path}")

    # Final metrics
    phys_G = build_graph(coords, net["edge_index"], net["edge_length"])
    phys_m = all_metrics(phys_G)
    mst_m = all_metrics(mst)
    del_m = all_metrics(delaubay_graph(coords))

    print(f"\n{'Network':<15} {'TL':>10} {'MD':>10} {'FT':>8}")
    print(f"{'MST':<15} {mst_m['TL']:>10.3f} {mst_m['MD']:>10.3f} {mst_m['FT']:>8.3f}")
    print(f"{'Delaunay':<15} {del_m['TL']:>10.3f} {del_m['MD']:>10.3f} {del_m['FT']:>8.3f}")
    print(f"{'Physarum':<15} {phys_m['TL']:>10.3f} {phys_m['MD']:>10.3f} {phys_m['FT']:>8.3f}")
    print(f"\nNormalised to MST:")
    print(f"  TL/TL_MST = {phys_m['TL']/mst_m['TL']:.3f}  (paper: ~1.75 ± 0.30)")
    print(f"  MD/MD_MST = {phys_m['MD']/mst_m['MD']:.3f}  (paper: ~0.85 ± 0.04)")
    print(f"  FT        = {phys_m['FT']:.3f}  (paper: 0-1 range; MST = 0)")

    return state, coords, mst_m, del_m


def experiment_parameter_sweep(mst_m, del_m):
    """Sweep I_0 from 0.5 to 8.0; each value yields one converged network."""
    print("\n" + "=" * 60)
    print("Experiment 2: parameter sweep over I_0 (paper Fig. 3)")
    print("=" * 60)

    coords, _ = tokyo_coords()
    edge_index, edge_length = build_delaunay_edges(coords)

    I0_values = [0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.0]
    rows = []

    for I0 in I0_values:
        cfg = PhysarumConfig(
            I0=I0, gamma=1.8, dt=0.05, n_steps=4000, seed=0,
        )
        state = run_simulation(
            coords,
            edge_index=edge_index,
            edge_length=edge_length,
            config=cfg,
        )
        net = final_network(state, threshold=cfg.prune_threshold)
        G = build_graph(coords, net["edge_index"], net["edge_length"])
        m = all_metrics(G)
        TL_mst = m["TL"] / mst_m["TL"]
        MD_mst = m["MD"] / mst_m["MD"]
        FT = m["FT"]
        n_edges = len(net["edge_index"])
        conn_frac = m.get("largest_cc_frac", 1.0)
        print(f"I0={I0:>4.1f}: edges={n_edges:>3}, "
              f"TL_mst={TL_mst:.3f}, MD_mst={MD_mst:.3f}, FT={FT:.3f}, "
              f"connected={conn_frac:.2f}")
        rows.append({
            "I0": I0,
            "TL_mst": TL_mst,
            "MD_mst": MD_mst,
            "FT": FT,
            "n_edges": n_edges,
            "conn_frac": conn_frac,
            "label": f"Physarum I₀ sweep",
            "color": COLOR_PHYSARUM,
            "marker": "o",
        })

    plot_pareto_and_curves(rows, mst_m, del_m, coords, suffix="I0_sweep",
                           xlabel="I₀")
    return rows


def experiment_gamma_sweep(mst_m, del_m):
    """Sweep γ at fixed I_0. Paper: low γ -> more cross-linking, Steiner pts."""
    print("\n" + "=" * 60)
    print("Experiment 3: parameter sweep over γ at fixed I_0=2")
    print("=" * 60)

    coords, _ = tokyo_coords()
    edge_index, edge_length = build_delaunay_edges(coords)

    gamma_values = [0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 5.0]
    rows = []

    for gamma in gamma_values:
        cfg = PhysarumConfig(
            I0=2.0, gamma=gamma, dt=0.05, n_steps=4000, seed=0,
        )
        state = run_simulation(
            coords,
            edge_index=edge_index,
            edge_length=edge_length,
            config=cfg,
        )
        net = final_network(state, threshold=cfg.prune_threshold)
        G = build_graph(coords, net["edge_index"], net["edge_length"])
        m = all_metrics(G)
        TL_mst = m["TL"] / mst_m["TL"]
        MD_mst = m["MD"] / mst_m["MD"]
        FT = m["FT"]
        n_edges = len(net["edge_index"])
        conn_frac = m.get("largest_cc_frac", 1.0)
        print(f"γ={gamma:>4.1f}: edges={n_edges:>3}, "
              f"TL_mst={TL_mst:.3f}, MD_mst={MD_mst:.3f}, FT={FT:.3f}, "
              f"connected={conn_frac:.2f}")
        rows.append({
            "gamma": gamma,
            "TL_mst": TL_mst,
            "MD_mst": MD_mst,
            "FT": FT,
            "n_edges": n_edges,
            "conn_frac": conn_frac,
            "label": "Physarum γ sweep",
            "color": COLOR_PHYSARUM,
            "marker": "s",
        })

    plot_pareto_and_curves(rows, mst_m, del_m, coords, suffix="gamma_sweep",
                           xlabel="γ")
    return rows


def plot_pareto_and_curves(rows, mst_m, del_m, coords, *, suffix, xlabel):
    pareto_results = [
        {
            "TL_mst": r["TL_mst"],
            "MD_mst": r["MD_mst"],
            "label": f"Physarum ({suffix})",
            "color": COLOR_PHYSARUM,
            "marker": "o" if "I0" in suffix else "s",
        }
        for r in rows
    ]
    reference_points = {
        "Delaunay": (
            del_m["TL"] / mst_m["TL"],
            del_m["MD"] / mst_m["MD"],
            COLOR_DELAUNAY,
            "^",
        ),
    }
    save_path = FIG_DIR / f"pareto_{suffix}.png"
    plot_pareto(
        pareto_results,
        reference=reference_points,
        save_path=save_path,
        title=f"Cost vs transport efficiency ({suffix.replace('_', ' ')})",
    )
    print(f"Saved {save_path}")

    x_values = [r.get("I0", r.get("gamma")) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(x_values, [r["FT"] for r in rows], "o-",
                 color=COLOR_PHYSARUM, linewidth=2, markersize=8)
    axes[0].axhline(0.0, color=COLOR_MST, linestyle="--", linewidth=1.5,
                    label="MST = 0")
    axes[0].axhline(del_m["FT"], color=COLOR_DELAUNAY, linestyle="--",
                    linewidth=1.5, label=f"Delaunay = {del_m['FT']:.2f}")
    axes[0].set_xlabel(xlabel, fontsize=11)
    axes[0].set_ylabel("Fault Tolerance FT", fontsize=11)
    axes[0].set_title(f"FT vs {xlabel}", fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)

    axes[1].plot(x_values, [r["n_edges"] for r in rows], "o-",
                 color=COLOR_PHYSARUM, linewidth=2, markersize=8)
    axes[1].axhline(len(coords) - 1, color=COLOR_MST, linestyle="--",
                    linewidth=1.5, label=f"MST = {len(coords)-1}")
    delaunay_n = delaubay_graph(coords).number_of_edges()
    axes[1].axhline(delaunay_n, color=COLOR_DELAUNAY, linestyle="--",
                    linewidth=1.5, label=f"Delaunay = {delaunay_n}")
    axes[1].set_xlabel(xlabel, fontsize=11)
    axes[1].set_ylabel("# edges", fontsize=11)
    axes[1].set_title(f"Network density vs {xlabel}", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=9)

    axes[2].plot(x_values, [r["MD_mst"] for r in rows], "o-",
                 color=COLOR_PHYSARUM, linewidth=2, markersize=8,
                 label="MD_mst (efficiency)")
    axes[2].plot(x_values, [r["TL_mst"] for r in rows], "s-",
                 color="#9467bd", linewidth=2, markersize=8,
                 label="TL_mst (cost)")
    axes[2].axhline(1.0, color=COLOR_MST, linestyle="--", linewidth=1.5,
                    label="MST baseline = 1.0")
    axes[2].set_xlabel(xlabel, fontsize=11)
    axes[2].set_ylabel("normalised metric", fontsize=11)
    axes[2].set_title(f"Cost & efficiency vs {xlabel}", fontsize=12)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="center right", fontsize=9)

    plt.tight_layout()
    save_path = FIG_DIR / f"metrics_vs_{suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    state, coords, mst_m, del_m = experiment_evolution()
    experiment_parameter_sweep(mst_m, del_m)
    experiment_gamma_sweep(mst_m, del_m)


if __name__ == "__main__":
    main()
