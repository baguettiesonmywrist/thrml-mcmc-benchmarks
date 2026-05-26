import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key

from models import IsingModel
from samplers.mh import _build_neighbour_list


def run_gibbs(
    key: Key,
    model: IsingModel,
    n_warmup: int,
    n_samples: int,
    steps_per_sample: int,
    init_spins: Array,
) -> Array:
    """Run single-site Gibbs sampling on an Ising model.

    Each step samples spin i exactly from its conditional distribution.
    No accept/reject — unlike MH, every proposal is always accepted.
    One sweep = n_nodes steps.

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

    def gibbs_step(spins: Array, key: Key) -> tuple[Array, None]:
        k1, k2 = jax.random.split(key)
        node = jax.random.randint(k1, shape=(), minval=0, maxval=n)
        local_field = jnp.sum(nbr_J[node] * mask[node] * spins[neighbours[node]])
        # probability that spin i = +1 given its neighbours
        p_up = jax.nn.sigmoid(2.0 * beta * local_field)
        new_spin = jnp.where(jax.random.uniform(k2) < p_up, 1.0, -1.0)
        spins = spins.at[node].set(new_spin)
        return spins, None

    def sweep(spins: Array, step_keys: Array) -> tuple[Array, None]:
        spins, _ = jax.lax.scan(gibbs_step, spins, step_keys)
        return spins, None

    def sample_step(spins: Array, sweep_keys: Array) -> tuple[Array, Array]:
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
