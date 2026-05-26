import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from analysis import load_all_results, plot_convergence, plot_time_comparison, plot_quality_comparison, PLOTS_DIR

results = load_all_results()
print(f"Loaded configs: {list(results.keys())}")

# only test on smoke_test results
assert "smoke_test" in results, "smoke_test results not found — run verify_benchmark.py first"
assert all(s in results["smoke_test"] for s in ["mh", "gibbs", "thrml"]), "missing sampler results"

# convergence plot
plot_convergence(results, "smoke_test")
assert os.path.exists(os.path.join(PLOTS_DIR, "convergence__smoke_test.png"))
print("convergence plot saved ok")

# time comparison
plot_time_comparison(results, ["smoke_test"], title="Smoke test — time", fname="time__smoke_test")
assert os.path.exists(os.path.join(PLOTS_DIR, "time__smoke_test.png"))
print("time comparison plot saved ok")

# quality comparison
plot_quality_comparison(results, ["smoke_test"], title="Smoke test — quality", fname="quality__smoke_test")
assert os.path.exists(os.path.join(PLOTS_DIR, "quality__smoke_test.png"))
print("quality comparison plot saved ok")

print("\nall checks passed")
