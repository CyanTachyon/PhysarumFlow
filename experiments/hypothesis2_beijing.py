"""
Hypothesis 2: The Physarum model produces networks that resemble the real
Beijing subway, even though the model has zero knowledge of Beijing.

This is the most compelling demonstration of biological universality: the
same local rule (flux strengthens tubes, no flux kills them) applied to
ANY city's geography should produce a network that looks like that city's
real transport network, because both are solutions to the same underlying
trade-off between cost, efficiency, and robustness.

Protocol
--------
1. Apply the model (with paper-tuned I_0=2, γ=1.8) to 36 Beijing locations.
2. Compare the resulting network to:
   - The Beijing MST baseline
   - The Delaunay triangulation baseline
   - A simplified hand-coded representation of the actual Beijing subway
     (loop lines + major radials)
3. Compute the three quality metrics for each.
4. Visualise side-by-side.
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
from src.beijing_data import beijing_coords
from src.metrics import (
    build_graph, minimum_spanning_tree, delaubay_graph, all_metrics,
)
from src.viz import plot_network, plot_overlay

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def beijing_subway_skeleton(coords: np.ndarray, names: list[str]) -> np.ndarray:
    """Hand-coded edges of the Beijing subway (simplified, lines 1/2/4/5/10/13).

    Edges connect major stations that share a real subway line. The set is
    intentionally sparse -- it represents the *trunk* of the network, not
    every stop. Station pairs are matched to the nearest nodes in our 36-node
    set by name.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}

    edges_raw = [
        ("Line 1 (east-west spine)"),
        ("Shijingshan", "Wukesong"),
        ("Wukesong", "Gongzhufen"),
        ("Gongzhufen", "Financial St"),
        ("Financial St", "Fuxingmen"),
        ("Fuxingmen", "Tiananmen"),
        ("Tiananmen", "Wangfujing"),
        ("Wangfujing", "Jianguomen"),
        ("Jianguomen", "Guomao"),
        ("Guomao", "Sihui"),
        ("Sihui", "Tongzhou"),

        ("Line 2 (inner loop)"),
        ("Fuxingmen", "Beijing North"),
        ("Beijing North", "Xizhimen"),
        ("Xizhimen", "Andingmen"),
        ("Andingmen", "Yonghegong"),
        ("Yonghegong", "Dongzhimen"),
        ("Dongzhimen", "Jianguomen"),
        ("Jianguomen", "Beijing Station"),
        ("Beijing Station", "Qianmen"),
        ("Qianmen", "Fuxingmen"),

        ("Line 4 (north-south)"),
        ("Fragrant Hills", "Shijingshan"),
        ("Shijingshan", "Gongzhufen"),
        ("Gongzhufen", "Financial St"),
        ("Financial St", "Fuxingmen"),
        ("Fuxingmen", "Xizhimen"),
        ("Xizhimen", "Beijing North"),
        ("Beijing North", "Olympic Park"),
        ("Olympic Park", "Peking Univ"),
        ("Peking Univ", "Tsinghua"),

        ("Line 5 (north-south east)"),
        ("Yonghegong", "Andingmen"),
        ("Andingmen", "Hepingli"),
        ("Hepingli", "Huixinxijie"),
        ("Yonghegong", "Wangfujing"),
        ("Wangfujing", "Qianmen"),
        ("Qianmen", "Tiantan"),

        ("Line 10 (outer ring)"),
        ("Hepingli", "Shaoyaoju"),
        ("Shaoyaoju", "Wangjing"),
        ("Wangjing", "Guomao"),
        ("Guomao", "Beijing South"),
        ("Beijing South", "Lize"),
        ("Lize", "Gongzhufen"),
        ("Gongzhufen", "Tsinghua"),
        ("Tsinghua", "Huixinxijie"),
        ("Huixinxijie", "Hepingli"),

        ("Line 13 (northern loop)"),
        ("Xizhimen", "Wangjing"),
        ("Wangjing", "Dongzhimen"),
        ("Dongzhimen", "Xizhimen"),

        ("Yizhuang line"),
        ("Beijing South", "Yizhuang"),

        ("Daxing line"),
        ("Qianmen", "Daxing"),

        ("Airport express"),
        ("Dongzhimen", "Capital Airport"),

        ("Line 8 (Olympic spine)"),
        ("Nanluoguxiang", "Yonghegong"),
        ("Yonghegong", "Olympic Park"),
    ]

    edges = []
    for entry in edges_raw:
        if not isinstance(entry, tuple) or len(entry) != 2:
            continue
        a, b = entry
        if a in name_to_idx and b in name_to_idx:
            edges.append(sorted((name_to_idx[a], name_to_idx[b])))
    seen: set = set()
    out = []
    for e in edges:
        t = tuple(e)
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return np.array(out, dtype=np.int64)


def main():
    print("=" * 70)
    print("Hypothesis 2: Physarum solution vs real Beijing subway")
    print("=" * 70)

    coords, names = beijing_coords()
    edge_index, edge_length = build_delaunay_edges(coords)

    cfg = PhysarumConfig(I0=2.0, gamma=1.8, dt=0.05, n_steps=5000, seed=0)
    state = run_simulation(
        coords, edge_index=edge_index, edge_length=edge_length, config=cfg,
    )
    net = final_network(state, threshold=cfg.prune_threshold)

    phys_G = build_graph(coords, net["edge_index"], net["edge_length"])
    mst = minimum_spanning_tree(coords)
    mst_idx = np.array([(u, v) for u, v in mst.edges()], dtype=np.int64)
    subway_idx = beijing_subway_skeleton(coords, names)
    subway_G = build_graph(coords, subway_idx)

    phys_m = all_metrics(phys_G)
    mst_m = all_metrics(mst)
    sub_m = all_metrics(subway_G)

    print(f"\n{'Network':<18} {'TL':>10} {'MD':>10} {'FT':>8} {'edges':>7}")
    print(f"{'MST':<18} {mst_m['TL']:>10.3f} {mst_m['MD']:>10.3f} "
          f"{mst_m['FT']:>8.3f} {mst.number_of_edges():>7}")
    print(f"{'Real Beijing':<18} {sub_m['TL']:>10.3f} {sub_m['MD']:>10.3f} "
          f"{sub_m['FT']:>8.3f} {subway_G.number_of_edges():>7}")
    print(f"{'Physarum':<18} {phys_m['TL']:>10.3f} {phys_m['MD']:>10.3f} "
          f"{phys_m['FT']:>8.3f} {phys_G.number_of_edges():>7}")
    print(f"\nNormalised to MST:")
    print(f"  Real Beijing: TL/TL_MST={sub_m['TL']/mst_m['TL']:.3f}, "
          f"MD/MD_MST={sub_m['MD']/mst_m['MD']:.3f}, FT={sub_m['FT']:.3f}")
    print(f"  Physarum:     TL/TL_MST={phys_m['TL']/mst_m['TL']:.3f}, "
          f"MD/MD_MST={phys_m['MD']/mst_m['MD']:.3f}, FT={phys_m['FT']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_network(
        coords, mst_idx,
        ax=axes[0], color="#1f77b4",
        title=f"MST  ({mst.number_of_edges()} edges, FT=0)",
        linewidth_scale=1.5,
    )
    plot_network(
        coords, subway_idx,
        ax=axes[1], color="#2ca02c",
        title=f"Beijing subway (simplified, {subway_G.number_of_edges()} edges)",
        linewidth_scale=1.5,
    )
    plot_network(
        coords, net["edge_index"], edge_weight=net["D"],
        ax=axes[2], color="#d62728",
        title=f"Physarum  ({len(net['edge_index'])} edges, FT={phys_m['FT']:.2f})",
        linewidth_scale=3.5,
    )
    fig.suptitle(
        "Hypothesis 2: Physarum on Beijing vs real subway vs MST baseline",
        fontsize=13,
    )
    plt.tight_layout()
    save_path = FIG_DIR / "hypothesis2_beijing.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {save_path}")

    overlap = len(
        {tuple(sorted(int(x) for x in e)) for e in net["edge_index"]}
        & {tuple(sorted(int(x) for x in e)) for e in subway_idx}
    )
    union = len(
        {tuple(sorted(int(x) for x in e)) for e in net["edge_index"]}
        | {tuple(sorted(int(x) for x in e)) for e in subway_idx}
    )
    print(f"\nEdge overlap Physarum <-> real Beijing subway: "
          f"{overlap}/{union} = {overlap/max(union,1):.1%} (Jaccard)")


if __name__ == "__main__":
    main()
