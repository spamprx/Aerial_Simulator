"""Graph layer: disk graphs, packet drops, Laplacians and spectral utilities.

Convention used everywhere in this code base
--------------------------------------------
``A`` is an (N, N) boolean matrix with ``A[i, j] == True``  <=>  agent ``i`` *receives*
agent ``j``'s message  (directed edge j -> i).  Row i is therefore agent i's in-neighbour
set, and the consensus update  x_i += sum_j A[i, j] (x_j - x_i)  is simply  -L x  with
L = D_in - A.  The diagonal is always False.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

FIXED_TOPOLOGIES = ("ring", "line", "complete", "star", "grid", "erdos_renyi", "custom")


# ----------------------------------------------------------------------------------
# building graphs
# ----------------------------------------------------------------------------------
def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    diff = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def disk_adjacency(pos: np.ndarray, radius: float) -> np.ndarray:
    """Distance-bounded (unit-disk) graph: i hears j iff ||p_i - p_j|| <= radius."""
    A = pairwise_distances(pos) <= radius
    np.fill_diagonal(A, False)
    return A


def packet_keep_mask(
    shape, drop_prob: float, rng: np.random.Generator, symmetric: bool = False
) -> np.ndarray:
    """Raw Bernoulli keep mask, before it meets any graph.  Draws nothing if ``drop_prob <= 0``."""
    if drop_prob <= 0.0:
        return np.ones(shape, dtype=bool)
    keep = rng.random(shape) >= drop_prob
    if symmetric:
        upper = np.triu(keep, 1)
        keep = upper | upper.T
    return keep


def apply_packet_drops(
    A: np.ndarray, drop_prob: float, rng: np.random.Generator, symmetric: bool = False
) -> np.ndarray:
    """Independent Bernoulli(drop_prob) loss on every link, redrawn each call (= each step)."""
    return A & packet_keep_mask(A.shape, drop_prob, rng, symmetric)


def make_topology(
    name: str, n: int, rng: Optional[np.random.Generator] = None, **params
) -> np.ndarray:
    """Fixed (position-independent) undirected topologies, returned as a boolean matrix.

    Used for the convergence-speed vs. algebraic-connectivity study and for the
    Erdos-Renyi robustness experiments.  ``erdos_renyi`` takes ``p`` (edge prob.) and
    ``ensure_connected`` (default True, resamples until connected).  ``custom`` takes
    ``adjacency``, an (n, n) list, and may be directed, so one exact graph can live in a config.
    """
    if name not in FIXED_TOPOLOGIES:
        raise ValueError(f"unknown topology '{name}', choose from {FIXED_TOPOLOGIES}")
    if name == "custom":
        A = np.array(params["adjacency"], dtype=bool)
        if A.shape != (n, n):
            raise ValueError(f"a custom adjacency must have shape ({n}, {n}), got {A.shape}")
        if A.diagonal().any():
            raise ValueError("a custom adjacency must not have self-loops")
        return A
    if name == "ring":
        G = nx.cycle_graph(n) if n >= 3 else nx.path_graph(n)
    elif name == "line":
        G = nx.path_graph(n)
    elif name == "complete":
        G = nx.complete_graph(n)
    elif name == "star":
        G = nx.star_graph(max(n - 1, 0))
    elif name == "grid":
        rows = int(params.get("rows", max(1, int(np.floor(np.sqrt(n))))))
        cols = int(np.ceil(n / rows))
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols), ordering="sorted")
        G = G.subgraph(range(n)).copy()  # first n cells in row-major order stay connected
    else:  # erdos_renyi
        rng = rng if rng is not None else np.random.default_rng()
        p = float(params.get("p", 0.3))
        ensure = bool(params.get("ensure_connected", True))
        for _ in range(1000):
            G = nx.gnp_random_graph(n, p, seed=int(rng.integers(2**31 - 1)))
            if not ensure or n < 2 or nx.is_connected(G):
                break
        else:
            raise RuntimeError(f"could not sample a connected G({n}, {p}); increase p")
    A = nx.to_numpy_array(G, nodelist=range(n)) > 0
    np.fill_diagonal(A, False)
    return A


# ----------------------------------------------------------------------------------
# linear algebra / spectral utilities
# ----------------------------------------------------------------------------------
def laplacian(A: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """In-degree Laplacian  L = D_in - W  (rows sum to zero).  ``x' = -L x`` is consensus."""
    W = A.astype(float) if weights is None else np.where(A, weights, 0.0)
    return np.diag(W.sum(axis=1)) - W


def algebraic_connectivity(A: np.ndarray) -> float:
    """lambda_2(L) of the underlying undirected graph (edge if either direction exists).

    Zero iff the (underlying) graph is disconnected; larger means faster consensus.
    """
    n = A.shape[0]
    if n < 2:
        return 0.0
    S = (A | A.T).astype(float)
    L = np.diag(S.sum(axis=1)) - S
    lam2 = float(np.linalg.eigvalsh(L)[1])
    return max(lam2, 0.0)


def to_networkx(A: np.ndarray, directed: bool = True) -> nx.Graph:
    """Edge j -> i for every A[i, j] (information flows from sender to receiver)."""
    if directed:
        return nx.from_numpy_array(A.T.astype(int), create_using=nx.DiGraph)
    return nx.from_numpy_array((A | A.T).astype(int))


def is_connected(A: np.ndarray, strong: bool = False) -> bool:
    """Connectivity of the underlying undirected graph, or strong connectivity if asked."""
    if A.shape[0] <= 1:
        return True
    if strong:
        return nx.is_strongly_connected(to_networkx(A, directed=True))
    return nx.is_connected(to_networkx(A, directed=False))


def degrees(A: np.ndarray) -> np.ndarray:
    """In-degree of every agent (number of messages received)."""
    return A.sum(axis=1)
