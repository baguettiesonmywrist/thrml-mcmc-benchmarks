import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from models import make_grid_model, ising_energy
from samplers.thrml_gibbs import run_thrml
from samplers.mh import run_mh

key = jax.random.key(0)

model = make_grid_model(side=10, J=1.0, beta=0.1)
init_spins = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(model.n_nodes,))

# --- check 1: output shape ---
samples = run_thrml(key, model, n_warmup=100, n_samples=200, steps_per_sample=5, init_spins=init_spins)
assert samples.shape == (200, model.n_nodes), f"unexpected shape {samples.shape}"
print(f"Output shape: {samples.shape}  (expect (200, {model.n_nodes}))")

# --- check 2: spins remain in {-1, +1} ---
assert jnp.all((samples == 1.0) | (samples == -1.0)), "spins left {-1, +1}"
print(f"Spin values: {jnp.unique(samples)}  (expect [-1, 1])")

# --- check 3: temperature dependence ---
model_hot  = make_grid_model(side=10, J=1.0, beta=0.01)
model_cold = make_grid_model(side=10, J=1.0, beta=10.0)

samples_hot  = run_thrml(key, model_hot,  n_warmup=500, n_samples=100, steps_per_sample=10, init_spins=init_spins)
samples_cold = run_thrml(key, model_cold, n_warmup=500, n_samples=100, steps_per_sample=10, init_spins=init_spins)

energy_hot  = jnp.mean(jax.vmap(lambda s: ising_energy(model_hot,  s))(samples_hot))
energy_cold = jnp.mean(jax.vmap(lambda s: ising_energy(model_cold, s))(samples_cold))

print(f"Mean energy (hot  beta=0.01): {energy_hot:.2f}  (expect near 0)")
print(f"Mean energy (cold beta=10.0): {energy_cold:.2f}  (expect very negative)")
assert energy_cold < energy_hot, "cold chain should have lower energy than hot chain"

# --- check 4: thrml and MH agree at same temperature ---
model_cmp = make_grid_model(side=10, J=1.0, beta=1.0)
samples_thrml = run_thrml(key, model_cmp, n_warmup=500, n_samples=500, steps_per_sample=10, init_spins=init_spins)
samples_mh    = run_mh(key,   model_cmp, n_warmup=500, n_samples=500, steps_per_sample=10, init_spins=init_spins)

energy_thrml = jnp.mean(jax.vmap(lambda s: ising_energy(model_cmp, s))(samples_thrml))
energy_mh    = jnp.mean(jax.vmap(lambda s: ising_energy(model_cmp, s))(samples_mh))

print(f"Mean energy THRML: {energy_thrml:.2f}, MH: {energy_mh:.2f}  (expect close)")
assert abs(energy_thrml - energy_mh) < 5.0, "THRML and MH should converge to similar energies"

# --- check 5: vmap over chains works ---
n_chains = 10
keys = jax.random.split(key, n_chains)
init_spins_batch = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(n_chains, model.n_nodes))
batch_samples = jax.vmap(
    lambda k, s: run_thrml(k, model, n_warmup=50, n_samples=20, steps_per_sample=5, init_spins=s)
)(keys, init_spins_batch)
assert batch_samples.shape == (n_chains, 20, model.n_nodes), f"unexpected batch shape {batch_samples.shape}"
print(f"Batched output shape: {batch_samples.shape}  (expect ({n_chains}, 20, {model.n_nodes}))")

print("\nall checks passed")
