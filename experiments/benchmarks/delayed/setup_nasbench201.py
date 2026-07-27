"""One-time setup for the NAS-Bench-201 tabular benchmark (nats_bench).

nats_bench ships only the query *code*; the tabular data (the NATS-Bench
topology-search-space archive, ~2 GB) is downloaded separately. Run this once
before `delayed_feedback_experiment.py` with `NASBench201Benchmark`.

Dependencies (install into your env first):
    pip install nats_bench

Data. Download the topology-search-space (tss) archive `NATS-tss-v1_0-3ffb9-simple`
from the NATS-Bench release (see https://github.com/D-X-Y/NATS-Bench --
Google Drive / figshare mirror linked there; it is access-gated, not on any
auto-download allowlist). Extract it so you have a directory named
`NATS-tss-v1_0-3ffb9-simple`. By default this script expects it under the
benchmark assets dir; pass --nats-path to point elsewhere.

Usage (from repo root):
    python -m experiments.benchmarks.delayed.setup_nasbench201 \
        --nats-path /path/to/NATS-tss-v1_0-3ffb9-simple
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.benchmarks.delayed.nasbench201_bandit import (
    NB201_DATASETS,
    NASBench201Benchmark,
    arch_str_from_ops,
)

DEFAULT_NATS_PATH = (
    Path(__file__).parent / "assets" / "NATS-tss-v1_0-3ffb9-simple"
)


def verify(nats_path: Path, instance: str) -> None:
    """Construct the benchmark (builds/loads the ref cache) and smoke-test a
    couple of queries."""
    if not nats_path.exists():
        raise FileNotFoundError(
            f"NATS-Bench archive not found at {nats_path}. Download "
            f"'NATS-tss-v1_0-3ffb9-simple' from the NATS-Bench release and "
            f"extract it there (or pass --nats-path)."
        )
    print(f"--- Building NASBench201Benchmark(instance={instance!r}) ---")
    print("    (first construction enumerates 15625 archs for p*/median runtime; cached)")
    bench = NASBench201Benchmark(instance=instance, nats_path=str(nats_path))
    print(f"    n_arms(space) = 5**6 = 15625; p* (max val-acc) = {bench.max_value:.4f}")
    print(f"    median training time = {bench._median_runtime_seconds:.1f} s")

    # Smoke-test: the all-3x3-conv cell (a strong reference architecture) and a
    # random cell.
    strong = {f"edge_{i}": "nor_conv_3x3" for i in range(6)}
    p, rt = bench._query(strong)
    print(f"    all-nor_conv_3x3: val_acc={100*p:.2f}%  train_time={rt:.1f}s")
    print(f"    arch string: {arch_str_from_ops([strong[f'edge_{i}'] for i in range(6)])}")
    rnd = {f"edge_{i}": bench.rng.choice(bench._search_space[f'edge_{i}']['choices'])
           for i in range(6)}
    p2, rt2 = bench._query(rnd)
    print(f"    random cell:      val_acc={100*p2:.2f}%  train_time={rt2:.1f}s")
    print("--- setup OK ---")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nats-path", type=Path, default=DEFAULT_NATS_PATH)
    ap.add_argument("--instance", default="cifar100", choices=list(NB201_DATASETS))
    args = ap.parse_args()
    verify(args.nats_path, args.instance)
    return 0


if __name__ == "__main__":
    sys.exit(main())
