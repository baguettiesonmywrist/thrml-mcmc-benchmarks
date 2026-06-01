import os
import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from models import (
    IsingModel,
    ising_energy,
    maxcut_value,
    make_grid_model,
    make_random_regular_model,
    make_maxcut_rrg_model,
)
from samplers.mh import run_mh
from samplers.gibbs import run_gibbs
from samplers.thrml_gibbs import run_thrml

RESULTS_DIR = "results"

@dataclass
class BenchmarkConfig:
    name: str               # unique identifier for this config, used in filenames
    model: IsingModel       # the problem instance
    problem: str            # "ferro", "antiferro", or "maxcut"
    n_chains: int           # number of parallel chains (vmap)
    n_warmup: int           # sweeps before collecting samples
    n_samples: int          # samples to collect per chain
    steps_per_sample: int

@dataclass
class BenchmarkResult:
    config_name: str
    sampler: str                   # "mh", "gibbs", or "thrml"
    energy_trajectory: np.ndarray  # [n_chains, n_samples] energy at each sample
    best_energy: np.ndarray        # [n_chains] minimum energy found per chain
    wall_time: float               # seconds, excludes jit compilation
    output_memory_bytes: float     # output array footprint: n_chains × n_samples × n_nodes × 4 bytes

def _run_sampler(
    key: jax.Array,
    config: BenchmarkConfig,
    run_fn: Callable,
    sampler_name: str,
) -> BenchmarkResult:
    n = config.model.n_nodes

    # initialise all chains with random spins
    key, k_init = jax.random.split(key)
    init_spins = jax.random.choice(
        k_init, jnp.array([-1.0, 1.0]), shape=(config.n_chains, n)
    )
    keys = jax.random.split(key, config.n_chains)

    # build the vmapped sampler
    batched = jax.jit(
        jax.vmap(
            lambda k, s: run_fn(k, config.model, config.n_warmup, config.n_samples, config.steps_per_sample, s)
        )
    )

    # warm up: compile without timing
    print(f"  [{sampler_name}] compiling...", flush=True)
    _ = jax.block_until_ready(batched(keys, init_spins))

    # timed run
    print(f"  [{sampler_name}] running...", flush=True)
    t0 = time.perf_counter()
    samples = jax.block_until_ready(batched(keys, init_spins))
    wall_time = time.perf_counter() - t0
    output_memory = config.n_chains * config.n_samples * config.model.n_nodes * 4

    # samples: [n_chains, n_samples, n_nodes]
    # compute energy or cut value one chain at a time to avoid OOM on dense graphs
    # double vmap would materialise [n_chains, n_samples, n_edges] intermediates
    if config.problem == "maxcut":
        metric_fn = lambda s: maxcut_value(config.model, s)
    else:
        metric_fn = lambda s: ising_energy(config.model, s)

    trajectory = np.stack([
        np.array(jax.vmap(metric_fn)(samples[i]))
        for i in range(config.n_chains)
    ])  # [n_chains, n_samples]
    # for Ising lower is better (energy), for MaxCut higher is better (cut value)
    best = trajectory.max(axis=1) if config.problem == "maxcut" else trajectory.min(axis=1)

    print(f"  [{sampler_name}] done in {wall_time:.2f}s  mean_best={best.mean():.2f}  output_mem={output_memory/1e6:.1f}MB", flush=True)

    return BenchmarkResult(
        config_name=config.name,
        sampler=sampler_name,
        energy_trajectory=trajectory,
        best_energy=best,
        wall_time=wall_time,
        output_memory_bytes=output_memory,
    )

N_RUNS = 3

def save_result(result: BenchmarkResult, run_idx: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{result.config_name}__{result.sampler}__run{run_idx}.npz")
    np.savez(
        path,
        energy_trajectory=result.energy_trajectory,
        best_energy=result.best_energy,
        wall_time=np.array(result.wall_time),
        output_memory_bytes=np.array(result.output_memory_bytes),
    )
    print(f"  saved -> {path}")

def run_benchmark(key: jax.Array, config: BenchmarkConfig) -> None:
    print(f"\n=== {config.name} ===")
    for run_idx in range(N_RUNS):
        print(f"  --- run {run_idx + 1}/{N_RUNS} ---")
        for sampler_name, run_fn in [("mh", run_mh), ("gibbs", run_gibbs), ("thrml", run_thrml)]:
            key, subkey = jax.random.split(key)
            result = _run_sampler(subkey, config, run_fn, sampler_name)
            save_result(result, run_idx)

# --- experiment configs ---

def make_configs() -> list[BenchmarkConfig]:
    configs = []

    sampling_kwargs = dict(n_chains=50, n_warmup=200, n_samples=500, steps_per_sample=5)

    # sweep over system sizes: 2500, 4900, 10000 nodes
    for side in [50, 70, 100]:
        configs.append(BenchmarkConfig(
            name=f"ferro_grid_{side}x{side}",
            model=make_grid_model(side=side, J=1.0, beta=1.0),
            problem="ferro",
            **sampling_kwargs,
        ))
        configs.append(BenchmarkConfig(
            name=f"antiferro_grid_{side}x{side}",
            model=make_grid_model(side=side, J=-1.0, beta=1.0),
            problem="antiferro",
            **sampling_kwargs,
        ))

    # sweep over temperatures on a mid-size grid
    for beta in [0.1, 0.5, 1.0, 2.0]:
        configs.append(BenchmarkConfig(
            name=f"ferro_grid_70x70_beta{beta}",
            model=make_grid_model(side=70, J=1.0, beta=beta),
            problem="ferro",
            **sampling_kwargs,
        ))

    # random regular graph
    for n_nodes in [900, 1600, 2500]:
        configs.append(BenchmarkConfig(
            name=f"ferro_rrg_{n_nodes}",
            model=make_random_regular_model(n_nodes=n_nodes, degree=3, J=1.0, beta=1.0),
            problem="ferro",
            **sampling_kwargs,
        ))

    # maxcut on random regular graph - degree sweep at same sizes as RRG
    for degree in [3, 5]:
        for n_nodes in [900, 1600, 2500]:
            configs.append(BenchmarkConfig(
                name=f"maxcut_d{degree}_{n_nodes}",
                model=make_maxcut_rrg_model(n_nodes=n_nodes, degree=degree, beta=1.0),
                problem="maxcut",
                **sampling_kwargs,
            ))

    return configs

if __name__ == "__main__":
    key = jax.random.key(0)
    configs = make_configs()
    print(f"Running {len(configs)} configs x 3 samplers")
    for config in configs:
        key, subkey = jax.random.split(key)
        run_benchmark(subkey, config)
    print("\nAll benchmarks complete.")
