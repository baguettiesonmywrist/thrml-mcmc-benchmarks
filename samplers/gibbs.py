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

    warmup_key, sample_key = jax.random.split(key)

    def do_sweep(spins: Array, sweep_key: Key) -> Array:
        """Run one sweep of n single-site Gibbs updates."""
        def gibbs_step(spins: Array, step_idx: int) -> tuple[Array, None]:
            k = jax.random.fold_in(sweep_key, step_idx)
            k1, k2 = jax.random.split(k)
            node = jax.random.randint(k1, shape=(), minval=0, maxval=n)
            local_field = jnp.sum(nbr_J[node] * mask[node] * spins[neighbours[node]])
            p_up = jax.nn.sigmoid(2.0 * beta * local_field)
            new_spin = jnp.where(jax.random.uniform(k2) < p_up, 1.0, -1.0)
            spins = spins.at[node].set(new_spin)
            return spins, None
        spins, _ = jax.lax.scan(gibbs_step, spins, jnp.arange(n))
        return spins

    def warmup_sweep(spins: Array, sweep_idx: int) -> tuple[Array, None]:
        sweep_key = jax.random.fold_in(warmup_key, sweep_idx)
        return do_sweep(spins, sweep_key), None

    def sample_step(spins: Array, sample_idx: int) -> tuple[Array, Array]:
        def sub_sweep(spins: Array, sub_idx: int) -> tuple[Array, None]:
            sweep_key = jax.random.fold_in(jax.random.fold_in(sample_key, sample_idx), sub_idx)
            return do_sweep(spins, sweep_key), None
        spins, _ = jax.lax.scan(sub_sweep, spins, jnp.arange(steps_per_sample))
        return spins, spins

    spins = init_spins

    if n_warmup > 0:
        spins, _ = jax.lax.scan(warmup_sweep, spins, jnp.arange(n_warmup))

    _, samples = jax.lax.scan(sample_step, spins, jnp.arange(n_samples))

    return samples  # [n_samples, n_nodes]
