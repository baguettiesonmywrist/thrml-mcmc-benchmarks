import os
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
    """Load and pool results across all runs. Returns None if no runs exist."""
    runs = []
    for run_idx in range(3):
        path = os.path.join(RESULTS_DIR, f"{config_name}__{sampler}__run{run_idx}.npz")
        if os.path.exists(path):
            runs.append(np.load(path))
    if not runs:
        return None
    return {
        # pool chains across runs for richer statistics
        "energy_trajectory": np.concatenate([r["energy_trajectory"] for r in runs], axis=0),
        "best_energy": np.concatenate([r["best_energy"] for r in runs], axis=0),
        # average wall time across runs for a stable estimate
        "wall_time": float(np.mean([r["wall_time"] for r in runs])),
        "output_memory_bytes": float(runs[0]["output_memory_bytes"]) if "output_memory_bytes" in runs[0].files else -1.0,
    }


def load_all_results() -> dict:
    results = defaultdict(dict)
    seen = set()
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".npz"):
            continue
        # filename format: config__sampler__runN.npz
        parts = fname[:-4].rsplit("__", 2)
        if len(parts) != 3 or not parts[2].startswith("run"):
            continue
        config_name, sampler = parts[0], parts[1]
        if sampler not in SAMPLERS:
            continue
        key = (config_name, sampler)
        if key not in seen:
            seen.add(key)
            results[config_name][sampler] = load_result(config_name, sampler)
    return dict(results)


# --- convergence diagnostics ---

def gelman_rubin(chains: np.ndarray) -> float:
    """R-hat statistic. Values close to 1.0 indicate convergence across chains.
    Computed on the energy trajectory [n_chains, n_samples]."""
    n_chains, n_samples = chains.shape
    if n_chains < 2:
        return np.nan
    chain_means = chains.mean(axis=1)
    grand_mean = chain_means.mean()
    B = n_samples / (n_chains - 1) * np.sum((chain_means - grand_mean) ** 2)
    W = np.mean(chains.var(axis=1, ddof=1))
    if W == 0:
        return 1.0
    var_hat = (n_samples - 1) / n_samples * W + B / n_samples
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chain: np.ndarray) -> float:
    """ESS via integrated autocorrelation time on a single chain [n_samples]."""
    n = len(chain)
    centered = chain - chain.mean()
    # autocorrelation via FFT
    acf_full = np.fft.irfft(np.abs(np.fft.rfft(centered, n=2 * n)) ** 2)
    acf = acf_full[:n] / acf_full[0]
    # Geyer's initial monotone sequence: sum until first negative lag
    tau = 1.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        tau += 2 * acf[k]
    return float(n / tau)


def compute_diagnostics(data: dict) -> dict:
    """Compute R-hat, mean ESS, and ESS/second from a result dict."""
    traj = data["energy_trajectory"]  # [n_chains, n_samples]
    r_hat = gelman_rubin(traj)
    ess_per_chain = [effective_sample_size(traj[i]) for i in range(len(traj))]
    mean_ess = float(np.mean(ess_per_chain))
    ess_per_sec = mean_ess / data["wall_time"]
    return {"r_hat": r_hat, "ess": mean_ess, "ess_per_sec": ess_per_sec}


# --- plot helpers ---

def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def _mean_trajectory(data: dict) -> np.ndarray:
    return data["energy_trajectory"].mean(axis=0)


# --- comparison plots ---

def plot_convergence(results: dict, config_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for sampler in SAMPLERS:
        data = results.get(config_name, {}).get(sampler)
        if data is None:
            continue
        traj = _mean_trajectory(data)
        std = data["energy_trajectory"].std(axis=0)
        ax.plot(traj, color=COLORS[sampler], label=LABELS[sampler])
        ax.fill_between(range(len(traj)), traj - std, traj + std,
                        color=COLORS[sampler], alpha=0.15)
    ax.set_xlabel("Sample step")
    ax.set_ylabel("Mean energy")
    ax.set_title(f"Convergence — {config_name}")
    ax.legend()
    fig.tight_layout()
    _save(fig, f"convergence__{config_name}")


def plot_memory_comparison(results: dict, config_names: list[str], title: str, fname: str) -> None:
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(config_names) * 1.8), 4))
    for i, sampler in enumerate(SAMPLERS):
        mems = []
        for c in config_names:
            v = (results.get(c, {}).get(sampler) or {}).get("output_memory_bytes", -1.0)
            mems.append(v / 1e6 if v >= 0 else 0.0)
        ax.bar(x + i * width, mems, width, label=LABELS[sampler], color=COLORS[sampler])
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Output memory (MB)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_time_comparison(results: dict, config_names: list[str], title: str, fname: str) -> None:
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(config_names) * 1.8), 4))
    for i, sampler in enumerate(SAMPLERS):
        times = [results.get(c, {}).get(sampler, {}).get("wall_time", np.nan) for c in config_names]
        ax.bar(x + i * width, times, width, label=LABELS[sampler], color=COLORS[sampler])
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (s)  [log scale]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_scaling(results: dict, config_names: list[str], sizes: list[int], title: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for sampler in SAMPLERS:
        times = [results.get(c, {}).get(sampler, {}).get("wall_time", np.nan) for c in config_names]
        ax.plot(sizes, times, marker="o", color=COLORS[sampler], label=LABELS[sampler])
    ax.set_xlabel("Number of nodes")
    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (s)  [log scale]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_quality_comparison(results: dict, config_names: list[str], title: str,
                            fname: str, higher_is_better: bool = False) -> None:
    x = np.arange(len(config_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(6, len(config_names) * 1.8), 4))
    for i, sampler in enumerate(SAMPLERS):
        vals = [results.get(c, {}).get(sampler, {}).get("best_energy", np.array([np.nan])).mean()
                for c in config_names]
        ax.bar(x + i * width, vals, width, label=LABELS[sampler], color=COLORS[sampler])
    ax.set_xticks(x + width)
    ax.set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean best cut value" if higher_is_better else "Mean best energy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def plot_temperature_sweep(results: dict) -> None:
    betas = [0.1, 0.5, 1.0, 2.0]
    config_names = [f"ferro_grid_70x70_beta{b}" for b in betas]
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


# --- diagnostic plots ---

def plot_diagnostics(results: dict, config_names: list[str], title: str, fname: str) -> None:
    """R-hat and ESS/second for each sampler across configs."""
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(config_names) * 2), 4))
    x = np.arange(len(config_names))
    width = 0.25

    for i, sampler in enumerate(SAMPLERS):
        r_hats, ess_per_secs = [], []
        for c in config_names:
            data = results.get(c, {}).get(sampler)
            if data is None:
                r_hats.append(np.nan)
                ess_per_secs.append(np.nan)
            else:
                diag = compute_diagnostics(data)
                r_hats.append(diag["r_hat"])
                ess_per_secs.append(diag["ess_per_sec"])

        axes[0].bar(x + i * width, r_hats, width, label=LABELS[sampler], color=COLORS[sampler])
        axes[1].bar(x + i * width, ess_per_secs, width, label=LABELS[sampler], color=COLORS[sampler])

    # R-hat convergence threshold
    axes[0].axhline(1.1, color="black", linestyle="--", linewidth=1, label="R-hat = 1.1 threshold")
    axes[0].set_ylabel("R-hat  (lower = better, <1.1 = converged)")
    axes[0].set_title("Gelman-Rubin R-hat")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    axes[0].legend(fontsize=7)

    axes[1].set_yscale("log")
    axes[1].set_ylabel("ESS / second  [log scale]  (higher = better)")
    axes[1].set_title("Effective samples per second")
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(config_names, rotation=30, ha="right", fontsize=8)
    axes[1].legend(fontsize=7)

    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, fname)


def print_diagnostics_table(results: dict, config_names: list[str]) -> None:
    """Save R-hat, ESS, ESS/s, output memory, and mean best energy/cut to results_summary.txt."""
    header = (f"{'Config':<35} {'Sampler':<28} {'R-hat':>6} {'ESS':>8} {'ESS/s':>10}"
              f" {'OutMem(MB)':>11} {'MeanBest':>10}")
    lines = [header, "-" * len(header)]
    for c in config_names:
        is_maxcut = c.startswith("maxcut_d")
        for sampler in SAMPLERS:
            data = results.get(c, {}).get(sampler)
            if data is None:
                continue
            d = compute_diagnostics(data)
            mem = data.get("output_memory_bytes", -1.0)
            mem_str = f"{mem/1e6:>11.1f}" if mem >= 0 else f"{'N/A':>11}"
            mean_best = data["best_energy"].mean()
            lines.append(f"{c:<35} {LABELS[sampler]:<28} {d['r_hat']:>6.3f} {d['ess']:>8.1f}"
                         f" {d['ess_per_sec']:>10.1f}{mem_str} {mean_best:>10.2f}")
    lines.append("")
    output = "\n".join(lines)
    with open("results_summary.txt", "w") as f:
        f.write(output)
    print("saved → results_summary.txt")


# --- main ---

if __name__ == "__main__":
    results = load_all_results()
    print(f"Loaded results for {len(results)} configs")

    # --- convergence plots (key configs only) ---
    key_configs = [
        "ferro_grid_70x70", "ferro_grid_100x100",
        "antiferro_grid_70x70", "antiferro_grid_100x100",
        "ferro_rrg_1600", "ferro_rrg_2500",
        "maxcut_d3_1600", "maxcut_d3_2500",
        "maxcut_d5_1600", "maxcut_d5_2500",
    ]
    for c in key_configs:
        if c in results:
            plot_convergence(results, c)

    # --- scaling ---
    grid_sides = [50, 70, 100]
    grid_nodes = [s * s for s in grid_sides]
    rrg_nodes = [900, 1600, 2500]
    maxcut_nodes = [900, 1600, 2500]

    plot_scaling(results, [f"ferro_grid_{s}x{s}" for s in grid_sides],
                 grid_nodes, title="Grid ferro — scaling", fname="scaling_grid_ferro")
    plot_scaling(results, [f"antiferro_grid_{s}x{s}" for s in grid_sides],
                 grid_nodes, title="Grid antiferro — scaling", fname="scaling_grid_antiferro")
    plot_scaling(results, [f"ferro_rrg_{n}" for n in rrg_nodes],
                 rrg_nodes, title="Random regular graph — scaling", fname="scaling_rrg")
    plot_scaling(results, [f"maxcut_d3_{n}" for n in maxcut_nodes],
                 maxcut_nodes, title="MaxCut degree-3 — scaling", fname="scaling_maxcut_d3")
    plot_scaling(results, [f"maxcut_d5_{n}" for n in maxcut_nodes],
                 maxcut_nodes, title="MaxCut degree-5 — scaling", fname="scaling_maxcut_d5")

    # --- time comparisons ---
    plot_time_comparison(results, [f"ferro_grid_{s}x{s}" for s in grid_sides],
                         title="Wall-clock time — grid ferro", fname="time_grid_ferro")
    plot_time_comparison(results, [f"antiferro_grid_{s}x{s}" for s in grid_sides],
                         title="Wall-clock time — grid antiferro", fname="time_grid_antiferro")
    plot_time_comparison(results, [f"ferro_rrg_{n}" for n in rrg_nodes],
                         title="Wall-clock time — random regular graph", fname="time_rrg")
    plot_time_comparison(results, [f"maxcut_d3_{n}" for n in maxcut_nodes],
                         title="Wall-clock time — MaxCut degree-3", fname="time_maxcut_d3")
    plot_time_comparison(results, [f"maxcut_d5_{n}" for n in maxcut_nodes],
                         title="Wall-clock time — MaxCut degree-5", fname="time_maxcut_d5")

    # --- memory comparisons ---
    plot_memory_comparison(results, [f"ferro_grid_{s}x{s}" for s in grid_sides],
                           title="Output memory — grid ferro", fname="memory_grid_ferro")
    plot_memory_comparison(results, [f"antiferro_grid_{s}x{s}" for s in grid_sides],
                           title="Output memory — grid antiferro", fname="memory_grid_antiferro")
    plot_memory_comparison(results, [f"ferro_rrg_{n}" for n in rrg_nodes],
                           title="Output memory — random regular graph", fname="memory_rrg")
    plot_memory_comparison(results, [f"maxcut_d3_{n}" for n in maxcut_nodes],
                           title="Output memory — MaxCut degree-3", fname="memory_maxcut_d3")
    plot_memory_comparison(results, [f"maxcut_d5_{n}" for n in maxcut_nodes],
                           title="Output memory — MaxCut degree-5", fname="memory_maxcut_d5")

    # --- quality comparisons ---
    plot_quality_comparison(results, [f"ferro_grid_{s}x{s}" for s in grid_sides],
                            title="Solution quality — grid ferro", fname="quality_grid_ferro")
    plot_quality_comparison(results, [f"antiferro_grid_{s}x{s}" for s in grid_sides],
                            title="Solution quality — grid antiferro", fname="quality_grid_antiferro")
    plot_quality_comparison(results, [f"ferro_rrg_{n}" for n in rrg_nodes],
                            title="Solution quality — random regular graph", fname="quality_rrg")
    plot_quality_comparison(results, [f"maxcut_d3_{n}" for n in maxcut_nodes],
                            title="Solution quality — MaxCut degree-3", fname="quality_maxcut_d3",
                            higher_is_better=True)
    plot_quality_comparison(results, [f"maxcut_d5_{n}" for n in maxcut_nodes],
                            title="Solution quality — MaxCut degree-5", fname="quality_maxcut_d5",
                            higher_is_better=True)

    # --- temperature sweep ---
    plot_temperature_sweep(results)

    # --- convergence diagnostics ---
    grid_configs = [f"ferro_grid_{s}x{s}" for s in grid_sides]
    antiferro_configs = [f"antiferro_grid_{s}x{s}" for s in grid_sides]
    rrg_configs = [f"ferro_rrg_{n}" for n in rrg_nodes]
    maxcut_configs = [f"maxcut_d3_{n}" for n in maxcut_nodes] + [f"maxcut_d5_{n}" for n in maxcut_nodes]

    plot_diagnostics(results, grid_configs,
                     title="Diagnostics — grid ferro", fname="diagnostics_grid_ferro")
    plot_diagnostics(results, antiferro_configs,
                     title="Diagnostics — grid antiferro", fname="diagnostics_grid_antiferro")
    plot_diagnostics(results, rrg_configs,
                     title="Diagnostics — random regular graph", fname="diagnostics_rrg")
    plot_diagnostics(results, [f"maxcut_d3_{n}" for n in maxcut_nodes],
                     title="Diagnostics — MaxCut degree-3", fname="diagnostics_maxcut_d3")
    plot_diagnostics(results, [f"maxcut_d5_{n}" for n in maxcut_nodes],
                     title="Diagnostics — MaxCut degree-5", fname="diagnostics_maxcut_d5")

    # --- print diagnostics table ---
    sweep_configs = [f"ferro_grid_70x70_beta{b}" for b in [0.1, 0.5, 1.0, 2.0]]
    all_configs = grid_configs + antiferro_configs + sweep_configs + rrg_configs + maxcut_configs
    print_diagnostics_table(results, all_configs)

    print("All plots saved.")
