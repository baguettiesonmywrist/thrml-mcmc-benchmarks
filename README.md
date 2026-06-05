# Thermodynamic Block Gibbs Benchmark

Benchmarking Extropic's **THRML block Gibbs** sampler against **Metropolis–Hastings** and **single-site Gibbs** on Ising ground-state estimation and MaxCut, all running on a single GPU via JAX.

## Samplers

| File | Sampler |
|------|---------|
| `samplers/mh.py` | Metropolis–Hastings (single-site, rejection) |
| `samplers/gibbs.py` | Single-site Gibbs (rejection-free) |
| `samplers/thrml_gibbs.py` | Block Gibbs via THRML (parallel colour classes) |

## Layout

- `models.py` - Ising model + graph builders (2D grid, random regular graph, MaxCut) and energy/cut functions
- `benchmark.py` - runs all samplers on every configuration; writes `results/*.npz`
- `analysis.py` - convergence (R-hat), ESS / ESS-per-second, plots; writes `plots/` and `results_summary.txt`
- `significance.py` - Mann–Whitney U tests on per-chain best values; writes `significance.txt`
- `plot.py` - per-configuration detail plots
- `test/` - verification scripts for each sampler and the models

## Usage

Run from the repo root using a virtualenv:

```bash
.venv/bin/python benchmark.py     # run the benchmark -> results/
.venv/bin/python analysis.py      # diagnostics + plots -> plots/, results_summary.txt
.venv/bin/python significance.py  # statistical tests -> significance.txt
.venv/bin/python plot.py          # per-config detail plots -> plots/
```

Requires Python with `jax`, `thrml`, `networkx`, `numpy`, `equinox`, `matplotlib`, and `scipy`.
