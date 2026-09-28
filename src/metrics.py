"""
Network quality metrics for the Physarum experiment.

Three core metrics from Tero et al. (2010):

  TL  (Total Length)     : sum of all edge lengths -- the "cost" of the network.
  MD  (Mean Distance)    : average shortest-path distance over all node pairs
                           -- the "transport efficiency" (lower = better).
  FT  (Fault Tolerance)  : fraction of single-edge failures that do NOT
                           disconnect the network (0 for any tree, 1 for a
                           graph whose every edge lies on a cycle).

Each metric is also reported normalised to the value for the Minimum Spanning
Tree (MST) of the same point set, following the paper's convention:
    TL_MST  = TL / TL_MST   (>= 1)
    MD_MST  = MD / MD_MST   (<= 1, lower means shorter paths than the MST)
    FT_MST  = FT            (already on [0,1]; MST itself has FT = 0)
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np


def build_graph(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_length: np.ndarray | None = None,
) -> nx.Graph:
    """Build a weighted networkx Graph from arrays."""
    if edge_length is None:
        diff = coords[edge_index[:, 0]] - coords[edge_index[:, 1]]
        edge_length = np.linalg.norm(diff, axis=1)
    edge_length_arr: np.ndarray = np.asarray(edge_length)
    G = nx.Graph()
    for i, (x, y) in enumerate(coords):
        G.add_node(i, pos=(float(x), float(y)))
    for (a, b), L in zip(edge_index, edge_length_arr):
        G.add_edge(int(a), int(b), length=float(L))
    return G


def total_length(G: nx.Graph) -> float:
    """Sum of all edge lengths."""
    return sum(d["length"] for _, _, d in G.edges(data=True))


def mean_pairwise_distance(G: nx.Graph) -> float:
    """Average shortest-path distance over all unordered node pairs.

    If the graph is disconnected, missing pairs are scored as +inf, which
    propagates to a mean of +inf. Callers should check connectivity.
    """
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    n = G.number_of_nodes()
    total = 0.0
    count = 0
    for i in lengths:
        for j, d in lengths[i].items():
            if i < j:
                total += d
                count += 1
    if count == 0:
        return float("inf")
    return total / count


def fault_tolerance(G: nx.Graph) -> float:
    """Fraction of edges whose removal leaves the graph connected.

    Equivalent to (E - #bridges) / E. A tree has FT = 0; a graph where every
    edge lies on at least one cycle has FT = 1.
    """
    m = G.number_of_edges()
    if m == 0:
        return 0.0
    n_bridges = sum(1 for _ in nx.bridges(G))
    return (m - n_bridges) / m


def minimum_spanning_tree(coords: np.ndarray) -> nx.Graph:
    """Euclidean MST of the point set."""
    n = len(coords)
    complete = nx.complete_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            complete[i][j]["length"] = d
    return nx.minimum_spanning_tree(complete, weight="length")


def delaubay_graph(coords: np.ndarray) -> nx.Graph:
    """Euclidean Delaunay triangulation graph."""
    from scipy.spatial import Delaunay

    tri = Delaunay(coords)
    pairs: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for i, j in ((a, b), (b, c), (a, c)):
            pairs.add((min(i, j), max(i, j)))
    edge_index = np.array(sorted(pairs), dtype=np.int64)
    return build_graph(coords, edge_index)


def all_metrics(G: nx.Graph) -> dict[str, float]:
    """Return TL, MD, FT for a single graph.

    If the graph is disconnected, MD is computed on the largest connected
    component so the metric stays finite and meaningful. The 'connected'
    flag in the returned dict records this.
    """
    connected = nx.is_connected(G)
    if connected:
        md = mean_pairwise_distance(G)
    else:
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        sub = G.subgraph(largest).copy()
        md = mean_pairwise_distance(sub)
    return {
        "TL": total_length(G),
        "MD": md,
        "FT": fault_tolerance(G),
        "connected": 1.0 if connected else 0.0,
        "largest_cc_frac": (
            len(max(nx.connected_components(G), key=len)) / G.number_of_nodes()
        ),
    }


def normalised_to_mst(
    G: nx.Graph, mst_metrics: dict[str, float]
) -> dict[str, float]:
    """Express TL, MD, FT normalised to the MST baseline."""
    m = all_metrics(G)
    return {
        "TL_mst": m["TL"] / mst_metrics["TL"],
        "MD_mst": m["MD"] / mst_metrics["MD"],
        "FT_mst": m["FT"],   # MST itself is 0, so we keep FT raw
    }


def edge_subgraph_metrics(
    coords: np.ndarray,
    edge_index: np.ndarray,
    edge_length: np.ndarray | None = None,
) -> dict[str, float]:
    """Convenience: metrics straight from coords + edge arrays."""
    G = build_graph(coords, edge_index, edge_length)
    return all_metrics(G)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((10, 2))

    mst = minimum_spanning_tree(coords)
    del_g = delaubay_graph(coords)
    print(f"MST: edges={mst.number_of_edges()}, "
          f"TL={total_length(mst):.3f}, "
          f"MD={mean_pairwise_distance(mst):.3f}, "
          f"FT={fault_tolerance(mst):.3f}")
    print(f"Delaunay: edges={del_g.number_of_edges()}, "
          f"TL={total_length(del_g):.3f}, "
          f"MD={mean_pairwise_distance(del_g):.3f}, "
          f"FT={fault_tolerance(del_g):.3f}")
