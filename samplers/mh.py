import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from models import IsingModel


def _build_neighbour_list(model: IsingModel) -> tuple[Array, Array, Array]:
    """Build padded neighbour arrays from edge list.

    Returns:
        neighbours:  [n_nodes, max_degree]  - neighbour indices
        nbr_J:       [n_nodes, max_degree]  - coupling to each neighbour
        mask:        [n_nodes, max_degree]  - True for real neighbours, False for padding
    """
    n = model.n_nodes
    src = np.array(model.edge_src)
    dst = np.array(model.edge_dst)
    J = np.array(model.J)

    degree = np.zeros(n, dtype=int)
    for i in src: degree[i] += 1
    for i in dst: degree[i] += 1
    max_degree = int(degree.max())

    neighbours = np.zeros((n, max_degree), dtype=np.int32)
    nbr_J = np.zeros((n, max_degree), dtype=np.float32)
    mask = np.zeros((n, max_degree), dtype=bool)
    count = np.zeros(n, dtype=int)

    for e, (s, d) in enumerate(zip(src, dst)):
        neighbours[s, count[s]] = d;  nbr_J[s, count[s]] = J[e];  mask[s, count[s]] = True;  count[s] += 1
        neighbours[d, count[d]] = s;  nbr_J[d, count[d]] = J[e];  mask[d, count[d]] = True;  count[d] += 1

    return jnp.array(neighbours), jnp.array(nbr_J), jnp.array(mask)


def run_mh(
    key: Key,
    model: IsingModel,
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    init_spins: Array,
) -> Array:
    """Run Metropolis-Hastings sampling on an Ising model.

    One step = one single-spin flip proposal.
    One sweep = n_nodes steps (comparable to one Gibbs sweep).

    Args:
        key:              JAX PRNG key
        model:            IsingModel instance
        n_warmup:         number of sweeps before collecting samples
        n_samples:        number of samples to collect
        steps_per_sample: sweeps between each recorded sample
        init_spins:       initial spin configuration, shape [n_nodes], values in {-1, +1}

    Returns:
        samples of shape [n_samples, n_nodes]
    """
    neighbours, nbr_J, mask = _build_neighbour_list(model)
    n = model.n_nodes
    beta = model.beta

    def mh_step(spins: Array, key: Key) -> tuple[Array, None]:
        k1, k2 = jax.random.split(key)
        node = jax.random.randint(k1, shape=(), minval=0, maxval=n)
        local_field = jnp.sum(nbr_J[node] * mask[node] * spins[neighbours[node]])
        dE = 2.0 * beta * spins[node] * local_field
        accept = jax.random.uniform(k2) < jnp.exp(-dE)
        spins = spins.at[node].mul(jnp.where(accept, -1.0, 1.0))
        return spins, None

    def sweep(spins: Array, step_keys: Array) -> tuple[Array, None]:
        # step_keys: [n_nodes]
        spins, _ = jax.lax.scan(mh_step, spins, step_keys)
        return spins, None

    def sample_step(spins: Array, sweep_keys: Array) -> tuple[Array, Array]:
        # sweep_keys: [steps_per_sample, n_nodes]
        spins, _ = jax.lax.scan(sweep, spins, sweep_keys)
        return spins, spins

    total_sweeps = n_warmup + n_samples * steps_per_sample
    all_keys = jax.random.split(key, total_sweeps * n).reshape(total_sweeps, n)

    spins = init_spins

    if n_warmup > 0:
        spins, _ = jax.lax.scan(sweep, spins, all_keys[:n_warmup])

    sample_keys = all_keys[n_warmup:].reshape(n_samples, steps_per_sample, n)
    _, samples = jax.lax.scan(sample_step, spins, sample_keys)

    return samples  # [n_samples, n_nodes]
