"""Build the compact accuracy grid CSV consumed by experiments.benchmarks.tabular_finite.

Downloads HPOBench's precomputed ML tabular benchmark (a per-model zip on
figshare, one parquet per OpenML task_id), extracts a single task_id, fixes a
fidelity level, and averages validation/test accuracy over HPOBench's seeds
for every point of its discretized hyperparameter grid. The zip itself is
cached under experiments/benchmarks/.cache/ (gitignored) so re-running for a
different task_id does not re-download ~400MB every time.

Usage (from repo root):
    python -m experiments.benchmarks.build_rf_grid --task-id 9952
    python -m experiments.benchmarks.build_rf_grid --task-id 31 --fidelity max
"""

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

MODEL_URLS = {
    "xgb": "https://ndownloader.figshare.com/files/30469920",
    "svm": "https://ndownloader.figshare.com/files/30379359",
    "lr": "https://ndownloader.figshare.com/files/30379038",
    "rf": "https://ndownloader.figshare.com/files/30469089",
    "nn": "https://ndownloader.figshare.com/files/30379005",
}

CACHE_DIR = Path(__file__).parent / ".cache"
ASSETS_DIR = Path(__file__).parent / "assets"

# task_ids bundled in rf.zip (confirmed via `unzip -l`; one parquet per OpenML
# task_id, all classification tasks from the OpenML-CC18-style suite used in
# the HPOBench tabular paper). Other models' zips are not guaranteed to carry
# the exact same set.
RF_TASK_IDS = [
    3,  # kr-vs-kp, 3196 rows
    12,  # mfeat-factors, 2000 rows
    31,  # credit-g, 1000 rows
    53,  # vehicle, 846 rows
    3917,  # kc1, 2109 rows
    7592,  # adult, 48842 rows
    9952,  # phoneme, 5404 rows
    9977,  # nomao, 34465 rows
    9981,  # cnae-9, 1080 rows
    10101,  # blood-transfusion-service-center, 748 rows
    14965,  # bank-marketing, 45211 rows
    146195,  # connect-4, 67557 rows
    146212,  # shuttle, 58000 rows
    146606,  # higgs, 98050 rows
    146818,  # Australian, 690 rows
    146821,  # car, 1728 rows
    146822,  # segment, 2310 rows
    167119,  # jungle_chess_2pcs_raw_endgame_complete, 44819 rows
    167120,  # numerai28.6, 96320 rows
    168329,  # helena, 65196 rows
    168330,  # jannis, 83733 rows
    168331,  # volkert, 58310 rows
    168335,  # MiniBooNE, 130064 rows
    168868,  # APSFailure, 76000 rows
    168908,  # christine, 5418 rows
    168910,  # fabert, 8237 rows
    168911,  # jasmine, 2984 rows
    168912,  # sylvine, 5124 rows
]


def download_model_zip(model: str) -> Path:
    zip_path = CACHE_DIR / f"{model}.zip"
    if zip_path.exists():
        return zip_path
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model} tabular benchmark from figshare (~400MB)...")
    urlretrieve(MODEL_URLS[model], zip_path)
    return zip_path


def extract_task(zip_path: Path, model: str, task_id: int) -> tuple[pd.DataFrame, dict]:
    with zipfile.ZipFile(zip_path) as zf:
        parquet_name = f"{task_id}/{model}_{task_id}_data.parquet.gzip"
        metadata_name = f"{task_id}/{model}_{task_id}_metadata.json"
        zf.extract(parquet_name, CACHE_DIR)
        zf.extract(metadata_name, CACHE_DIR)

    table = pd.read_parquet(CACHE_DIR / parquet_name)
    with open(CACHE_DIR / metadata_name) as f:
        metadata = json.load(f)
    return table, metadata


def build_grid(
    table: pd.DataFrame,
    metadata: dict,
    fidelity: str = "max",
    metric_val: str = "val_scores",
    metric_test: str = "test_scores",
    metric_name: str = "acc",
) -> pd.DataFrame:
    fidelity_space = json.loads(metadata["config_spaces"]["z_discrete"])
    fidelity_names = [hp["name"] for hp in fidelity_space["hyperparameters"]]
    param_names = [
        hp["name"]
        for hp in json.loads(metadata["config_spaces"]["x_discrete"])["hyperparameters"]
    ]

    fidelity_values = {}
    for hp in fidelity_space["hyperparameters"]:
        seq = hp["sequence"]
        fidelity_values[hp["name"]] = max(seq) if fidelity == "max" else min(seq)

    mask = pd.Series(True, index=table.index)
    for name in fidelity_names:
        mask &= table[name] == fidelity_values[name]
    subset = table[mask].copy()

    subset["val_acc"] = subset["result"].apply(
        lambda r: r["info"][metric_val][metric_name]
    )
    subset["test_acc"] = subset["result"].apply(
        lambda r: r["info"][metric_test][metric_name]
    )

    grid = (
        subset.groupby(param_names)
        .agg(val_acc=("val_acc", "mean"), test_acc=("test_acc", "mean"))
        .reset_index()
    )
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rf", choices=sorted(MODEL_URLS))
    parser.add_argument(
        "--task-id",
        type=int,
        default=9952,
        help=f"OpenML task_id. For model=rf, see RF_TASK_IDS in this file: {RF_TASK_IDS}",
    )
    parser.add_argument(
        "--fidelity",
        default="max",
        choices=["max", "min"],
        help="Which fidelity grid point to fix (e.g. n_estimators) before averaging over seeds.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: assets/<model>_<task_id>_grid.csv)",
    )
    args = parser.parse_args()

    zip_path = download_model_zip(args.model)
    table, metadata = extract_task(zip_path, args.model, args.task_id)
    grid = build_grid(table, metadata, fidelity=args.fidelity)

    output = (
        Path(args.output)
        if args.output
        else ASSETS_DIR / f"{args.model}_{args.task_id}_grid.csv"
    )
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(output, index=False)

    print(f"{len(grid)} configs -> {output}")
    print(f"best val_acc = {grid['val_acc'].max():.4f}")


if __name__ == "__main__":
    main()
