import os
from scipy import stats
import numpy as np

RESULTS_DIR = "results"
SAMPLERS = ["mh", "gibbs", "thrml"]
LABELS = {"mh": "MH", "gibbs": "Gibbs", "thrml": "THRML"}

GROUPS = [
    ("Grid ferro",      ["ferro_grid_10x10", "ferro_grid_20x20", "ferro_grid_30x30"]),
    ("Grid antiferro",  ["antiferro_grid_10x10", "antiferro_grid_20x20", "antiferro_grid_30x30"]),
    ("Temp sweep",      ["ferro_grid_10x10_beta0.1", "ferro_grid_10x10_beta0.5",
                         "ferro_grid_10x10_beta1.0", "ferro_grid_10x10_beta2.0"]),
    ("RRG",             ["ferro_rrg_100", "ferro_rrg_400", "ferro_rrg_900"]),
    ("MaxCut",          ["maxcut_100", "maxcut_400", "maxcut_900"]),
]

COMPARISONS = [("thrml", "mh"), ("thrml", "gibbs"), ("mh", "gibbs")]


def load_best(config: str, sampler: str) -> np.ndarray | None:
    path = os.path.join(RESULTS_DIR, f"{config}__{sampler}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["best_energy"]  # [n_chains]


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

    col = f"{'Config':<35} {'Comparison':<16} {'p-value':>8}  {'r':>6}  {'sig'}"
    lines.append(col)

    for group_name, configs in GROUPS:
        lines.append(f"\n--- {group_name} ---")
        lines.append("-" * len(col))

        for config in configs:
            is_maxcut = config.startswith("maxcut")
            data = {s: load_best(config, s) for s in SAMPLERS}

            for s1, s2 in COMPARISONS:
                a, b = data[s1], data[s2]
                if a is None or b is None:
                    continue

                u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
                r = rank_biserial(u, len(a), len(b))

                if p < 0.05:
                    better = s1 if (r > 0) == is_maxcut else s2
                    note = f"← {LABELS[better]} better"
                else:
                    note = ""

                lines.append(
                    f"{config:<35} {LABELS[s1]+' vs '+LABELS[s2]:<16} "
                    f"{p:>8.4f}  {r:>6.3f}  {sig_label(p)}  {note}"
                )

    lines.append("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
    lines.append("r = rank-biserial effect size (0 = no difference, ±1 = complete separation)")
    lines.append("ns = samplers are statistically equivalent in solution quality\n")

    output = "\n".join(lines)
    print(output)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)
    print(f"saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    run_tests()
