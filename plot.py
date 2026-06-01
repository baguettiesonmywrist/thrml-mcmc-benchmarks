import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
SAMPLERS = ["mh", "gibbs", "thrml"]
COLORS = {"mh": "#e41a1c", "gibbs": "#377eb8", "thrml": "#4daf4a"}
LABELS = {"mh": "Metropolis-Hastings", "gibbs": "Standard Gibbs", "thrml": "Block Gibbs (THRML)"}


def _load(config_name: str, sampler: str) -> dict | None:
    path = os.path.join(RESULTS_DIR, f"{config_name}__{sampler}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {
        "energy_trajectory": d["energy_trajectory"],  # [n_chains, n_samples]
        "best_energy": d["best_energy"],               # [n_chains]
        "wall_time": float(d["wall_time"]),
        "output_memory_bytes": float(d["output_memory_bytes"]) if "output_memory_bytes" in d.files else -1.0,
    }


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {path}")


def _parse_params(config_name: str) -> str:
    """Extract readable parameter string from config name for plot title."""
    m = re.search(r"grid_(\d+)x(\d+)", config_name)
    if m:
        n = int(m.group(1)) * int(m.group(2))
        return f"{m.group(1)}×{m.group(2)} grid  |  {n} nodes"
    m = re.search(r"rrg_(\d+)", config_name)
    if m:
        return f"random regular graph  |  {m.group(1)} nodes"
    m = re.search(r"maxcut_(\d+)", config_name)
    if m:
        return f"MaxCut  |  {m.group(1)} nodes"
    return config_name


def plot_detail(config_name: str) -> None:
    """Per-config plot: energy trace + distribution + summary stats."""
    results = {s: _load(config_name, s) for s in SAMPLERS}
    available = {s: r for s, r in results.items() if r is not None}
    if not available:
        print(f"  no results for {config_name}, skipping")
        return

    is_maxcut = config_name.startswith("maxcut")
    metric_label = "Cut value" if is_maxcut else "Energy"

    fig, (ax_trace, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    # --- energy trace ---
    for sampler, data in available.items():
        traj = data["energy_trajectory"]
        mean = traj.mean(axis=0)
        std = traj.std(axis=0)
        ax_trace.plot(mean, color=COLORS[sampler], label=LABELS[sampler], linewidth=1.5)
        ax_trace.fill_between(range(len(mean)), mean - std, mean + std,
                              color=COLORS[sampler], alpha=0.12)
    ax_trace.set_xlabel("Sample index")
    ax_trace.set_ylabel(metric_label)
    ax_trace.set_title(f"{metric_label} trace  (mean ± 1 std across chains)")
    ax_trace.legend(fontsize=9)

    # --- distribution ---
    all_vals = np.concatenate([r["energy_trajectory"].ravel() for r in available.values()])
    emin, emax = all_vals.min(), all_vals.max()

    for sampler, data in available.items():
        ax_hist.hist(data["energy_trajectory"].ravel(), bins=40,
                     color=COLORS[sampler], histtype="step", linewidth=2.0,
                     label=LABELS[sampler], range=(emin, emax))
        best_val = data["best_energy"].max() if is_maxcut else data["best_energy"].min()
        ax_hist.axvline(best_val, color=COLORS[sampler], linestyle="--", linewidth=1.2)
    ax_hist.set_xlabel(metric_label)
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"{metric_label} distribution  (dashed = {'max' if is_maxcut else 'min'})")
    ax_hist.legend(fontsize=9)

    # --- title ---
    params = _parse_params(config_name)
    fig.suptitle(f"{config_name}  |  {params}", fontsize=10, fontweight="bold")

    # --- summary stats inside figure ---
    summary_lines = []
    for sampler in SAMPLERS:
        if sampler not in available:
            continue
        d = available[sampler]
        mean_e = d["energy_trajectory"].mean()
        best_e = d["best_energy"].max() if is_maxcut else d["best_energy"].min()
        best_label = "max cut" if is_maxcut else "min E  "
        t = d["wall_time"]
        mem = d.get("output_memory_bytes", -1.0)
        mem_str = f"  mem = {mem/1e6:.0f}MB" if mem >= 0 else ""
        summary_lines.append(
            f"{LABELS[sampler]:<28}  mean = {mean_e:9.2f}  {best_label} = {best_e:9.2f}  time = {t:.3f}s{mem_str}"
        )

    fig.text(0.5, 0.01, "\n".join(summary_lines), ha="center", va="bottom",
             fontsize=7.5, family="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save(fig, f"detail__{config_name}")


if __name__ == "__main__":
    config_names = set()
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith(".npz") and "__" in fname:
            name = fname[:-4].rsplit("__", 1)[0]
            if "smoke_test" not in name:
                config_names.add(name)

    print(f"Generating detail plots for {len(config_names)} configs...")
    for config_name in sorted(config_names):
        print(f"  {config_name}")
        plot_detail(config_name)

    print("Done.")
