import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np
from models import make_grid_model
from benchmark import BenchmarkConfig, run_benchmark, RESULTS_DIR

key = jax.random.key(0)

# small config just to verify the pipeline runs end to end
config = BenchmarkConfig(
    name="smoke_test",
    model=make_grid_model(side=5, J=1.0, beta=1.0),
    problem="ferro",
    n_chains=4,
    n_warmup=10,
    n_samples=20,
    steps_per_sample=2,
)

run_benchmark(key, config)

# verify all three result files were saved
for sampler in ["mh", "gibbs", "thrml"]:
    path = os.path.join(RESULTS_DIR, f"smoke_test__{sampler}.npz")
    assert os.path.exists(path), f"missing result file: {path}"

    data = np.load(path)
    assert data["energy_trajectory"].shape == (4, 20), f"unexpected shape {data['energy_trajectory'].shape}"
    assert data["best_energy"].shape == (4,), f"unexpected shape {data['best_energy'].shape}"
    assert float(data["wall_time"]) > 0, "wall time should be positive"

    print(f"[{sampler}] energy_trajectory {data['energy_trajectory'].shape}  "
          f"mean_best={data['best_energy'].mean():.2f}  "
          f"time={float(data['wall_time']):.2f}s")

print("\nsmoke test passed")
