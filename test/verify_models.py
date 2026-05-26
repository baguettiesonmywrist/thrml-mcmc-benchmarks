import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
from models import make_grid_model, make_random_model, make_random_regular_model, make_maxcut_model, ising_energy, maxcut_value

key = jax.random.key(0)

# --- grid model ---
model = make_grid_model(side=4, J=1.0, beta=1.0)
print(f"Grid: {model.n_nodes} nodes, {len(model.edge_src)} edges")

spins = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(model.n_nodes,))
e = ising_energy(model, spins)
print(f"Grid energy (random spins): {e:.4f}")

# ferromagnetic: all-aligned spins should have equal and negative energy
all_up = jnp.ones(model.n_nodes)
all_down = -jnp.ones(model.n_nodes)
e_up = ising_energy(model, all_up)
e_down = ising_energy(model, all_down)
print(f"All-up: {e_up:.4f}, all-down: {e_down:.4f}  (should be equal and negative)")
assert e_up == e_down, "all-up and all-down should have equal energy"
assert e_up < 0, "ground state energy should be negative for ferromagnet"

# antiferromagnetic: checkerboard should be lowest energy on grid
model_af = make_grid_model(side=4, J=-1.0, beta=1.0)
checkerboard = jnp.array([1.0 if (i // 4 + i % 4) % 2 == 0 else -1.0 for i in range(16)])
e_checker = ising_energy(model_af, checkerboard)
e_random = ising_energy(model_af, spins)
print(f"Antiferro checkerboard: {e_checker:.4f}, random: {e_random:.4f}  (checkerboard should be lower)")
assert e_checker < e_random, "checkerboard should be lower energy than random for antiferromagnet"

# --- random regular model ---
model_rr = make_random_regular_model(n_nodes=100, degree=3, J=1.0, beta=1.0)
print(f"Random regular: {model_rr.n_nodes} nodes, {len(model_rr.edge_src)} edges (expect 150)")
assert model_rr.n_nodes == 100
assert len(model_rr.edge_src) == 150, f"expected 150 edges, got {len(model_rr.edge_src)}"

# --- maxcut ---
model_mc = make_maxcut_model(n_nodes=20, edge_prob=0.4, beta=1.0)
spins_mc = jax.random.choice(key, jnp.array([-1.0, 1.0]), shape=(model_mc.n_nodes,))
cut = maxcut_value(model_mc, spins_mc)
print(f"MaxCut value: {cut:.0f} (out of {len(model_mc.edge_src)} edges)")
assert 0 <= cut <= len(model_mc.edge_src), "cut value should be between 0 and n_edges"

print("\nall checks passed")
