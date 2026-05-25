import jax.numpy as jnp
import networkx as nx
import numpy as np
from jaxtyping import Array
import equinox as eqx


class IsingModel(eqx.Module):
    edge_src: Array   # source node of each edge, shape [n_edges]
    edge_dst: Array   # destination node of each edge, shape [n_edges]
    J: Array          # coupling strength per edge, shape [n_edges]
    beta: Array       # inverse temperature, scalar
    n_nodes: int      # total number of nodes


def ising_energy(model: IsingModel, spins: Array) -> Array:
    """Energy of a spin configuration. Low energy = high probability."""
    return -model.beta * jnp.sum(model.J * spins[model.edge_src] * spins[model.edge_dst])


def maxcut_value(model: IsingModel, spins: Array) -> Array:
    """Number of edges cut by a spin configuration (edge cut when endpoints differ)."""
    edge_products = spins[model.edge_src] * spins[model.edge_dst]
    return jnp.sum(jnp.abs(model.J) * (1 - edge_products) / 2)


# --- Graph constructors ---

def _graph_to_model(G: nx.Graph, J_vals: np.ndarray, beta: float) -> IsingModel:
    """Convert a NetworkX graph + edge weights into an IsingModel."""
    G = nx.convert_node_labels_to_integers(G)
    edges = list(G.edges())
    src = np.array([e[0] for e in edges], dtype=np.int32)
    dst = np.array([e[1] for e in edges], dtype=np.int32)
    return IsingModel(
        edge_src=jnp.array(src),
        edge_dst=jnp.array(dst),
        J=jnp.array(J_vals, dtype=jnp.float32),
        beta=jnp.array(beta, dtype=jnp.float32),
        n_nodes=G.number_of_nodes(),
    )


def make_grid_model(side: int, J: float, beta: float) -> IsingModel:
    """2D square lattice Ising model.

    J > 0: ferromagnetic, J < 0: antiferromagnetic.
    """
    G = nx.grid_2d_graph(side, side)
    n_edges = G.number_of_edges()
    return _graph_to_model(G, np.full(n_edges, J, dtype=np.float32), beta)


def make_random_model(n_nodes: int, edge_prob: float, J: float, beta: float, seed: int = 0) -> IsingModel:
    """Erdos-Renyi random graph Ising model."""
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    n_edges = G.number_of_edges()
    return _graph_to_model(G, np.full(n_edges, J, dtype=np.float32), beta)


def make_random_regular_model(n_nodes: int, degree: int, J: float, beta: float, seed: int = 0) -> IsingModel:
    """Random regular graph Ising model. Every node has exactly `degree` neighbours."""
    G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    n_edges = G.number_of_edges()
    return _graph_to_model(G, np.full(n_edges, J, dtype=np.float32), beta)


def make_maxcut_model(n_nodes: int, edge_prob: float, beta: float, seed: int = 0) -> IsingModel:
    """MaxCut on an Erdos-Renyi graph. Sets J = -1 so minimising energy = maximising the cut."""
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    n_edges = G.number_of_edges()
    return _graph_to_model(G, np.full(n_edges, -1.0, dtype=np.float32), beta)
