"""
Core Physarum polycephalum adaptive network solver.

Faithful reproduction of the mathematical model in:
    Tero et al. (2010) "Rules for Biologically Inspired Adaptive Network Design"
    Science 327(5964): 439-432.

The model has three ingredients:

  (1) Hagen-Poiseuille flux through a tube of length L_ij and conductivity D_ij:
          Q_ij = (D_ij / L_ij) * (p_i - p_j)

  (2) Kirchhoff conservation at every node, with one source (flow +I0) and
      one sink (flow -I0). Solving this linear system gives the node pressures p_i
      and therefore every edge flux Q_ij.

  (3) Conductivity adaptation (the key biological feedback rule):
          dD_ij/dt = f(|Q_ij|) - D_ij,    f(|Q|) = |Q|^gamma / (1 + |Q|^gamma)

At every simulation step, a new (source, sink) pair is chosen, fluxes are
computed, and conductivities are updated by an Euler step. Edges whose
conductivity drops below a small threshold are pruned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class PhysarumConfig:
    """Hyperparameters of the Physarum model.

    Defaults follow the "typical simulation" reported in the paper:
        I_0 = 2, gamma = 1.8
    """

    I0: float = 2.0          # source/sink flux magnitude
    gamma: float = 1.8       # nonlinearity of the feedback sigmoid
    dt: float = 1.0e-3       # Euler step for D_ij update
    prune_threshold: float = 1.0e-3   # D_ij below this -> edge removed
    source_selection: str = "random_pair"   # "random_pair" or "uniform"
    n_steps: int = 8000      # number of simulation steps
    seed: int = 0


@dataclass
class PhysarumState:
    """Mutable state of one simulation run."""

    coords: np.ndarray            # (N, 2) node positions
    edge_index: np.ndarray        # (E, 2) endpoints of each undirected edge
    edge_length: np.ndarray       # (E,) L_ij (Euclidean, can be overridden)
    D: np.ndarray                 # (E,) conductivity per edge
    history: list = field(default_factory=list)   # snapshots for visualization


def build_delaunay_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the initial graph as the Delaunay triangulation.

    This mirrors the paper, which uses a triangulated mesh as the initial
    network that the plasmodium then prunes.
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(coords)
    pairs: set[tuple[int, int]] = set()
    for simplex in tri.simplices:
        a, b, c = simplex
        for i, j in ((a, b), (b, c), (a, c)):
            pairs.add((min(i, j), max(i, j)))
    edge_index = np.array(sorted(pairs), dtype=np.int64)
    diff = coords[edge_index[:, 0]] - coords[edge_index[:, 1]]
    edge_length = np.linalg.norm(diff, axis=1)
    return edge_index, edge_length


def build_complete_edges(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All (i<j) edges; used as a baseline initial graph in some experiments."""
    n = len(coords)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = np.array(pairs, dtype=np.int64)
    diff = coords[edge_index[:, 0]] - coords[edge_index[:, 1]]
    edge_length = np.linalg.norm(diff, axis=1)
    return edge_index, edge_length


def build_grid_edges(coords: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """k-nearest-neighbour graph as another alternative initial topology."""
    from scipy.spatial import cKDTree

    n = len(coords)
    tree = cKDTree(coords)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        _, idx = tree.query(coords[i], k=k + 1)
        for j in idx[1:]:
            pairs.add((min(i, j), max(i, j)))
    edge_index = np.array(sorted(pairs), dtype=np.int64)
    diff = coords[edge_index[:, 0]] - coords[edge_index[:, 1]]
    edge_length = np.linalg.norm(diff, axis=1)
    return edge_index, edge_length


def _solve_fluxes(
    edge_index: np.ndarray,
    edge_length: np.ndarray,
    D: np.ndarray,
    n_nodes: int,
    source: int,
    sink: int,
    I0: float,
) -> np.ndarray:
    """Solve for edge fluxes Q_ij given current conductivities.

    Builds the sparse weighted Laplacian L where
        L[i, j] = -D_ij / L_ij   for i != j (off-diagonal)
        L[i, i] = sum_j D_ij / L_ij  (diagonal = row sum)
    Then solves L p = b, where b[source] = +I0, b[sink] = -I0.
    Edge fluxes are Q_ij = (D_ij / L_ij) * (p_i - p_j).

    A tiny regularisation is added to the diagonal so the linear solve never
    breaks down when the graph becomes disconnected during pruning.
    """
    ei, ej = edge_index[:, 0], edge_index[:, 1]
    w = D / np.maximum(edge_length, 1e-12)

    rows = np.concatenate([ei, ej, ei, ej])
    cols = np.concatenate([ej, ei, ei, ej])
    data = np.concatenate([-w, -w, w, w])

    L = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    b = np.zeros(n_nodes)
    b[source] = I0
    b[sink] = -I0

    # Regularise diagonal to handle isolated nodes robustly.
    L = L + sp.diags(np.full(n_nodes, 1e-9))

    p = spla.spsolve(L.tocsc(), b)

    Q = w * (p[ei] - p[ej])
    return Q


def run_simulation(
    coords: np.ndarray,
    *,
    edge_index: np.ndarray | None = None,
    edge_length: np.ndarray | None = None,
    config: PhysarumConfig | None = None,
    initial_D: np.ndarray | None = None,
    snapshot_every: int = 0,
) -> PhysarumState:
    """Run a Physarum simulation and return the final state.

    Parameters
    ----------
    coords
        (N, 2) array of node positions.
    edge_index, edge_length
        Optional initial graph. Defaults to Delaunay triangulation.
    config
        Hyperparameters; uses PhysarumConfig() defaults if None.
    initial_D
        Optional initial conductivities (default: uniform 1.0).
    snapshot_every
        If > 0, save snapshots of (alive edges, D) every N steps for animation.
    """
    if config is None:
        config = PhysarumConfig()

    coords = np.asarray(coords, dtype=np.float64)
    n_nodes = len(coords)

    if edge_index is None or edge_length is None:
        edge_index, edge_length = build_delaunay_edges(coords)
    else:
        edge_index = np.asarray(edge_index, dtype=np.int64)
        edge_length = np.asarray(edge_length, dtype=np.float64)

    if initial_D is None:
        # Paper: "initial random fluctuations of D_ij".
        rng = np.random.default_rng(config.seed)
        D = 1.0 + 0.1 * rng.standard_normal(len(edge_index))
        D = np.clip(D, 0.5, 1.5)
    else:
        D = np.asarray(initial_D, dtype=np.float64).copy()

    state = PhysarumState(
        coords=coords,
        edge_index=edge_index,
        edge_length=edge_length,
        D=D,
    )

    rng = np.random.default_rng(config.seed)

    for step in range(config.n_steps):
        # Pick source/sink pair. The paper cycles through pairs; random sampling
        # is equivalent in expectation and converges just as well.
        if config.source_selection == "random_pair":
            source, sink = rng.choice(n_nodes, size=2, replace=False)
        else:  # cycle through ordered pairs deterministically
            source = step % n_nodes
            sink = (step + 1 + (step // n_nodes)) % n_nodes
            if sink == source:
                sink = (sink + 1) % n_nodes

        Q = _solve_fluxes(
            edge_index, edge_length, D, n_nodes, source, sink, config.I0
        )

        # Conductivity update:  dD/dt = f(|Q|) - D
        abs_Q = np.abs(Q)
        f_Q = (abs_Q ** config.gamma) / (1.0 + abs_Q ** config.gamma)
        D = D + config.dt * (f_Q - D)

        # Soft prune: do not remove edges here, let them decay naturally.
        # Hard prune only at the end / on snapshots so the linear solve stays
        # well-conditioned during dynamics.

        if snapshot_every and (step + 1) % snapshot_every == 0:
            state.history.append(
                {
                    "step": step + 1,
                    "D": D.copy(),
                    "Q": Q.copy(),
                    "alive": D > config.prune_threshold,
                }
            )

    state.D = D
    return state


def final_network(state: PhysarumState, threshold: float | None = None) -> dict:
    """Return the alive edges (D > threshold) of a final state.

    Returns dict with edge_index, edge_length, D, and node coords.
    """
    if threshold is None:
        threshold = PhysarumConfig.prune_threshold
    alive = state.D > threshold
    return {
        "coords": state.coords,
        "edge_index": state.edge_index[alive],
        "edge_length": state.edge_length[alive],
        "D": state.D[alive],
        "alive_mask": alive,
    }


if __name__ == "__main__":
    # Smoke test on a small random cluster.
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((12, 2))
    cfg = PhysarumConfig(n_steps=2000, seed=0)
    state = run_simulation(coords, config=cfg)
    net = final_network(state)
    print(f"Initial edges: {len(state.edge_index)}")
    print(f"Final alive edges (D > {cfg.prune_threshold}): {len(net['edge_index'])}")
    print(f"D range: [{state.D.min():.4f}, {state.D.max():.4f}]")
