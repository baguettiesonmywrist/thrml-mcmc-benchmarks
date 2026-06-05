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
- `significance.py` - Mann-Whitney U tests on per-chain best values; writes `significance.txt`
- `plot.py` - per-configuration detail plots
- `test/` - verification scripts for each sampler and the models

## Setup

Requires Python 3.10+ and a CUDA-capable GPU (JAX is configured for CUDA 13).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** JAX GPU support requires matching CUDA/cuDNN drivers. The pinned versions target CUDA 13. If your environment differs, install `jax[cuda13]` separately following the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

## Usage

Run from the repo root with the virtualenv active:

```bash
python benchmark.py     # run all samplers on every config  ->  results/
python analysis.py      # convergence, ESS, scaling plots   ->  plots/, results_summary.txt
python significance.py  # Mann-Whitney U tests              ->  significance.txt
python plot.py          # per-config detail plots           ->  plots/
```

Verification scripts are in `test/` and can be run individually, e.g. `python test/verify_gibbs.py`.
