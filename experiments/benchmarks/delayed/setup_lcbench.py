"""One-time setup for the LCBench surrogate benchmark (yahpo_gym).

yahpo_gym ships only the surrogate *code*; the ONNX surrogate models + metadata
(the `yahpo_data` repo, a few hundred MB) are downloaded separately and their
location registered in a local config. Run this once before
`delayed_feedback_experiment.py` with `LCBenchMixedBenchmark`.

Dependencies (install into your env first):
    pip install "yahpo-gym>=1.0" onnxruntime "ConfigSpace>=0.6,<1.0"

Note on ConfigSpace: yahpo_gym 1.0.x pins the older ConfigSpace API
(get_hyperparameters(), OrdinalHyperparameter.sequence). If you have
ConfigSpace>=1.0 installed for something else, create a separate env for this.

Usage (from repo root):
    python -m experiments.benchmarks.delayed.setup_lcbench
    python -m experiments.benchmarks.delayed.setup_lcbench --data-dir ~/yahpo_data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / "yahpo_data"
YAHPO_DATA_REPO = "https://github.com/slds-lmu/yahpo_data.git"


def ensure_data(data_dir: Path) -> None:
    """Clone the yahpo_data surrogate repo if not already present."""
    if (data_dir / "lcbench").exists():
        print(f"--- yahpo_data/lcbench already present at {data_dir} ---")
        return
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"--- Cloning yahpo_data (surrogate ONNX + metadata) into {data_dir} ---")
    subprocess.run(
        ["git", "clone", "--depth", "1", YAHPO_DATA_REPO, str(data_dir)],
        check=True,
    )


def register_and_verify(data_dir: Path) -> None:
    """Point yahpo_gym at the local data dir and run a smoke-test query."""
    import yahpo_gym.benchmarks.lcbench  # noqa: F401
    from yahpo_gym import benchmark_set, local_config

    local_config.init_config()
    local_config.set_data_path(str(data_dir))
    print(f"--- yahpo_gym data path set to {data_dir} ---")

    bench = benchmark_set.BenchmarkSet("lcbench")
    instances = bench.instances
    print(f"--- lcbench available: {len(instances)} instances ---")
    print("    first few:", instances[:8])

    bench.set_instance(instances[0])
    cfg = bench.get_opt_space().sample_configuration().get_dictionary()
    fid_hp = bench.get_fidelity_space().get_hyperparameters()[0]
    cfg[fid_hp.name] = int(fid_hp.upper)
    res = bench.objective_function(cfg)
    res = res[0] if isinstance(res, list) else res
    print(f"--- smoke test on instance {instances[0]} ---")
    print("    objectives:", {k: round(float(v), 4) for k, v in res.items()})
    print("--- setup OK ---")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = ap.parse_args()
    ensure_data(args.data_dir)
    register_and_verify(args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
