import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
from jaxtyping import Array, Key

from thrml import Block, SamplingSchedule, sample_states, SpinNode
from thrml.models.ising import IsingEBM, IsingSamplingProgram

from models import IsingModel

def _build_program(model: IsingModel):
    """Convert IsingModel to a THRML sampling program.

    Returns:
        program:      IsingSamplingProgram ready for sample_states
        free_blocks:  list of Blocks (one per color group)
        nodes:        list of SpinNode objects, nodes[i] corresponds to integer node i
        color_groups: list of lists of integer node indices, one per color group
    """
    n = model.n_nodes
    src = np.array(model.edge_src)
    dst = np.array(model.edge_dst)

    # create one SpinNode per integer node, preserving index correspondence
    nodes = [SpinNode() for _ in range(n)]

    # reconstruct networkx graph for coloring
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src, dst))

    # colour the graph - non-adjacent nodes share a colour and can be updated in parallel
    coloring = nx.coloring.greedy_color(G, strategy="DSATUR")
    n_colors = max(coloring.values()) + 1
    color_groups = [[] for _ in range(n_colors)]
    for node_idx, color in coloring.items():
        color_groups[color].append(node_idx)

    free_blocks = [Block([nodes[i] for i in group]) for group in color_groups]

    edges = [(nodes[s], nodes[d]) for s, d in zip(src, dst)]
    biases = jnp.zeros(n, dtype=jnp.float32)

    ising_ebm = IsingEBM(nodes, edges, biases, model.J, model.beta)
    program = IsingSamplingProgram(ising_ebm, free_blocks, clamped_blocks=[])

    return program, free_blocks, nodes, color_groups

def run_thrml(
    key: Key,
    model: IsingModel,
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    init_spins: Array,
) -> Array:
    """Run THRML block Gibbs sampling on an Ising model.

    Non-adjacent nodes (same colour group) are updated in parallel within each step.

    Args:
        key:              JAX PRNG key
        model:            IsingModel instance
        n_warmup:         number of steps before collecting samples
        n_samples:        number of samples to collect
        steps_per_sample: steps between each recorded sample
        init_spins:       initial spin configuration, shape [n_nodes], values in {-1, +1}

    Returns:
        samples of shape [n_samples, n_nodes]
    """
    program, free_blocks, nodes, color_groups = _build_program(model)

    # convert {-1, +1} floats to booleans (THRML represents spins as bool)
    # True = +1, False = -1
    init_bool = init_spins > 0
    init_state = [init_bool[jnp.array(group, dtype=jnp.int32)] for group in color_groups]

    schedule = SamplingSchedule(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
    )

    # request samples for all nodes in original index order
    samples_bool = sample_states(key, program, schedule, init_state, [], [Block(nodes)])

    # samples_bool is a list with one element of shape [n_samples, n_nodes]
    # convert bool back to {-1, +1} float
    return jnp.where(samples_bool[0], 1.0, -1.0)
