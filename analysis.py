import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SAMPLERS = ["mh", "gibbs", "thrml"]
COLORS = {"mh": "#e41a1c", "gibbs": "#377eb8", "thrml": "#4daf4a"}
LABELS = {"mh": "Metropolis-Hastings", "gibbs": "Standard Gibbs", "thrml": "Block Gibbs (THRML)"}


# --- loading ---

def load_result(config_name: str, sampler: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"{config_name}__{sampler}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return {
        "energy_trajectory": data["energy_trajectory"],  # [n_chains, n_samples]
        "best_energy": data["best_energy"],               # [n_chains]
        "wall_time": float(data["wall_time"]),
    }


def load_all_results() -> dict:
    """Load all results into a nested dict: results[config_name][sampler] = data."""
    results = defaultdict(dict)
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".npz"):
            continue
        config_name, sampler = fname[:-4].rsplit("__", 1)
        if sampler in SAMPLERS:
            results[config_name][sampler] = load_result(config_name, sampler)
    return dict(results)


# --- plot helpers ---

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def _mean_trajectory(data: dict) -> np.ndarray:
    """Mean energy over chains at each sample step."""
    return data["energy_trajectory"].mean(axis=0)  # [n_samples]


# --- plots ---

def plot_convergence(results: dict, config_name: str) -> None:
    """Energy trajectory over sample steps for all three samplers."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for sampler in SAMPLERS:
        data = results.get(config_name, {}).get(sampler)
        if data is None:
            continue
        traj = _mean_trajectory(data)
        ax.plot(traj, color=COLORS[sampler], label=LABELS[sampler])
        # shaded band = std across chains
        std = data["energy_trajectory"].std(axis=0)
        ax.fill_between(range(len(traj)), traj - std, traj + std,
                        color=COLORS[sampler], alpha=0.15)

    ax.set_xlabel("Sample step")
    ax.set_ylabel("Mean energy")
    ax.set_title(f"Convergence — {config_name}")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"convergence__{config_name}")


def plot_time_comparison(results: dict, config_names: list[str], title: str, fname: str) -> None:
    """Bar chart of wall-clock times across configs."""
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(config_names) * 1.5), 4))

    for i, sampler in enumerate(SAMPLERS):
        times = [results.get(c, {}).get(sampler, {}).get("wall_time", 0) for c in config_names]
        ax.bar(x + i * width, times, width, label=LABELS[sampler], color=COLORS[sampler])

    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_scaling(results: dict, config_names: list[str], sizes: list[int], title: str, fname: str) -> None:
    """Wall-clock time vs system size for each sampler."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for sampler in SAMPLERS:
        times = [results.get(c, {}).get(sampler, {}).get("wall_time", np.nan) for c in config_names]
        ax.plot(sizes, times, marker="o", color=COLORS[sampler], label=LABELS[sampler])

    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_quality_comparison(results: dict, config_names: list[str], title: str, fname: str) -> None:
    """Mean best energy per sampler across configs (lower = better for Ising)."""
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(config_names) * 1.5), 4))

    for i, sampler in enumerate(SAMPLERS):
        means = [results.get(c, {}).get(sampler, {}).get("best_energy", np.array([np.nan])).mean()
                 for c in config_names]
        ax.bar(x + i * width, means, width, label=LABELS[sampler], color=COLORS[sampler])

    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean best energy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_temperature_sweep(results: dict) -> None:
    """Final mean energy vs beta for each sampler."""
    betas = [0.1, 0.5, 1.0, 2.0]
    config_names = [f"ferro_grid_10x10_beta{b}" for b in betas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, ylabel in zip(
        axes,
        ["best_energy", "wall_time"],
        ["Mean best energy", "Wall-clock time (s)"],
    ):
        for sampler in SAMPLERS:
            vals = []
            for c in config_names:
                data = results.get(c, {}).get(sampler)
                if data is None:
                    vals.append(np.nan)
                elif metric == "best_energy":
                    vals.append(data["best_energy"].mean())
                else:
                    vals.append(data["wall_time"])
            ax.plot(betas, vals, marker="o", color=COLORS[sampler], label=LABELS[sampler])

        ax.set_xlabel("Beta (inverse temperature)")
        ax.set_ylabel(ylabel)
        ax.legend()

    axes[0].set_title("Solution quality vs temperature")
    axes[1].set_title("Speed vs temperature")
    fig.tight_layout()
    _save(fig, "temperature_sweep")


# --- main ---

if __name__ == "__main__":
    results = load_all_results()
    print(f"Loaded results for {len(results)} configs")

    # convergence curves for key configs
    for config_name in results:
        if "smoke_test" in config_name:
            continue
        plot_convergence(results, config_name)

    # size scaling
    grid_sizes = [10*10, 20*20, 30*30]
    plot_scaling(
        results,
        [f"ferro_grid_{s}x{s}" for s in [10, 20, 30]],
        grid_sizes,
        title="Grid — wall-clock time vs system size (ferro)",
        fname="scaling_grid_ferro",
    )
    plot_scaling(
        results,
        [f"ferro_rrg_{n}" for n in [100, 400, 900]],
        [100, 400, 900],
        title="Random regular graph — wall-clock time vs system size",
        fname="scaling_rrg",
    )
    plot_scaling(
        results,
        [f"maxcut_{n}" for n in [100, 400, 900]],
        [100, 400, 900],
        title="MaxCut — wall-clock time vs system size",
        fname="scaling_maxcut",
    )

    # time and quality comparisons across topologies
    plot_time_comparison(
        results,
        [f"ferro_grid_{s}x{s}" for s in [10, 20, 30]],
        title="Wall-clock time — grid ferro",
        fname="time_grid_ferro",
    )
    plot_quality_comparison(
        results,
        [f"ferro_grid_{s}x{s}" for s in [10, 20, 30]],
        title="Solution quality — grid ferro",
        fname="quality_grid_ferro",
    )

    # temperature sweep
    plot_temperature_sweep(results)

    print("All plots saved.")
