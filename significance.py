import os
from scipy import stats
import numpy as np

RESULTS_DIR = "results"
SAMPLERS = ["mh", "gibbs", "thrml"]
LABELS = {"mh": "MH", "gibbs": "Gibbs", "thrml": "THRML"}

GROUPS = [
    ("Grid ferro",      ["ferro_grid_50x50", "ferro_grid_70x70", "ferro_grid_100x100"]),
    ("Grid antiferro",  ["antiferro_grid_50x50", "antiferro_grid_70x70", "antiferro_grid_100x100"]),
    ("Temp sweep",      ["ferro_grid_70x70_beta0.1", "ferro_grid_70x70_beta0.5",
                         "ferro_grid_70x70_beta1.0", "ferro_grid_70x70_beta2.0"]),
    ("RRG",             ["ferro_rrg_900", "ferro_rrg_1600", "ferro_rrg_2500"]),
    ("MaxCut d3",       ["maxcut_d3_900", "maxcut_d3_1600", "maxcut_d3_2500"]),
    ("MaxCut d5",       ["maxcut_d5_900", "maxcut_d5_1600", "maxcut_d5_2500"]),
]

COMPARISONS = [("thrml", "mh"), ("thrml", "gibbs"), ("mh", "gibbs")]

def load_best(config: str, sampler: str) -> np.ndarray | None:
    """Load and pool best_energy across all runs."""
    runs = []
    for run_idx in range(3):
        path = os.path.join(RESULTS_DIR, f"{config}__{sampler}__run{run_idx}.npz")
        if os.path.exists(path):
            runs.append(np.load(path)["best_energy"])
    if not runs:
        return None
    return np.concatenate(runs)  # [n_chains * n_runs]

def rank_biserial(u: float, n1: int, n2: int) -> float:
    """Effect size r in [-1, 1]. 0 = no effect, ±1 = complete separation."""
    return 2 * u / (n1 * n2) - 1

def sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "** "
    if p < 0.05:  return "*  "
    return "ns "

OUTPUT_FILE = "significance.txt"

def run_tests() -> None:
    lines = []

    col = f"{'Config':<35} {'Comparison':<16} {'val1':>10}  {'val2':>10}  {'p-value':>8}  {'r':>6}  {'sig'}"
    lines.append(col)

    for group_name, configs in GROUPS:
        lines.append(f"\n--- {group_name} ---")
        lines.append("-" * len(col))

        for config in configs:
            is_maxcut = config.startswith("maxcut_d")
            data = {s: load_best(config, s) for s in SAMPLERS}

            for s1, s2 in COMPARISONS:
                a, b = data[s1], data[s2]
                if a is None or b is None:
                    continue

                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                r = rank_biserial(u, len(a), len(b))

                mean1, mean2 = a.mean(), b.mean()
                val_str = f"{mean1:>10.2f}  {mean2:>10.2f}"

                if p < 0.05:
                    better = s1 if (r > 0) == is_maxcut else s2
                    note = f"<- {LABELS[better]} better"
                else:
                    note = ""

                lines.append(
                    f"{config:<35} {LABELS[s1]+' vs '+LABELS[s2]:<16} "
                    f"{val_str}  {p:>8.4f}  {r:>6.3f}  {sig_label(p)}  {note}"
                )

    lines.append("\nval1/val2 = mean best energy (Ising, lower is better) or mean best cut (MaxCut, higher is better)")
    lines.append("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    lines.append("r = rank-biserial effect size (0 = no difference, ±1 = complete separation)")
    lines.append("ns = samplers are statistically equivalent in solution quality\n")

    output = "\n".join(lines)
    print(output)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)
    print(f"saved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    run_tests()
