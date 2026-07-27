"""Build the compact Criteo conversion-bandit assets from the raw log.

Input: the Criteo Sponsored Search Conversion Log, downloaded (terms-of-use
gated) from https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/
-- a single tab-separated file `CriteoSearchData` whose columns are (per the
dataset's own header):

    Sale, SalesAmountInEuro, time_delay_for_conversion, click_timestamp,
    nb_clicks_1week, product_price, product_age_group, device_type,
    audience_id, product_gender, product_brand,
    product_category(1..7), product_country, product_id, product_title,
    partner_id, user_id

This script produces two small assets consumed at experiment time (the raw log
is large; this runs once):

  1. criteo_<instance>_arms.csv       segment, conversion_rate, count
       -- one row per segment (arm) that occurs >= --min-count times, with its
          empirical conversion rate (mean of Sale) and support. Read by
          CriteoConversionBenchmark.
  2. criteo_<instance>_delay.npz      delays_steps[], conversion_rate, ...
       -- the empirical click->conversion delays (positive only), converted
          from seconds to simulator "steps", plus the overall conversion rate
          (context/metadata only). Read by CriteoDelayModel.

Reward is the binary Sale (0/1), drawn per pull by the benchmark. Delay is the
real time_delay_for_conversion (seconds) of CONVERSIONS only (Sale==1); a -1
means no conversion happened (an observed 0, resolved at the patience deadline
-- NOT censored). Censoring in the conversion-signal model is a real conversion
whose delay exceeds the patience window, i.e. a lost positive signal -- it is
NOT the non-conversion rate.

Usage (from repo root):
    # Place CriteoSearchData under experiments/data/criteo/, then:
    python -m experiments.benchmarks.delayed.build_criteo_asset

    # Or pass an explicit path:
    python -m experiments.benchmarks.delayed.build_criteo_asset \
        --raw /path/to/data/criteo/CriteoSearchData
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS_DIR = Path(__file__).parent / "assets"
RAW_DIR = Path(__file__).parents[2] / "data" / "criteo"
DEFAULT_RAW_FILE = RAW_DIR / "CriteoSearchData"


def resolve_raw_path(raw_path: Path) -> Path:
    """Accept a file path or a directory containing CriteoSearchData."""
    path = Path(raw_path)
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "CriteoSearchData"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"No CriteoSearchData file found in {path}. "
            f"Download the log and place it at {DEFAULT_RAW_FILE}"
        )
    if path == DEFAULT_RAW_FILE or path.name == "CriteoSearchData":
        raise FileNotFoundError(
            f"Criteo raw log not found at {path}. "
            f"Download CriteoSearchData from "
            f"https://ailab.criteo.com/criteo-sponsored-search-conversion-log-dataset/ "
            f"and save it to {DEFAULT_RAW_FILE}"
        )
    raise FileNotFoundError(f"Criteo raw log not found: {path}")


# Official column order of the Sponsored Search Conversion Log (the file ships
# without a header row).
COLUMNS = [
    "Sale",
    "SalesAmountInEuro",
    "time_delay_for_conversion",
    "click_timestamp",
    "nb_clicks_1week",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category1",
    "product_category2",
    "product_category3",
    "product_category4",
    "product_category5",
    "product_category6",
    "product_category7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]

# Low-cardinality categoricals whose cross-product defines a segment (arm). All
# present in the log; chosen to yield thousands of well-populated segments
# rather than a handful (too coarse) or millions of singletons (product_id).
DEFAULT_SEGMENT_COLS = [
    "product_category1",
    "device_type",
    "product_age_group",
    "product_gender",
    "product_brand",
]

# 1 simulator step == this many seconds of real conversion delay. Criteo delays
# span seconds..weeks; the RF/LCBench experiments treat a step as ~an hour, so
# default to 3600 s/step to keep the same rough time unit. Tunable.
_SECONDS_PER_STEP = 3600.0


def build(
    raw_path: Path,
    instance: str,
    segment_cols: list[str],
    min_count: int,
    seconds_per_step: float,
    max_rows: int | None,
) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # The log has no header; read the needed columns by position. Missing
    # categoricals in Criteo are encoded as -1 / empty -- keep them as their own
    # bucket (a real "unknown" segment) rather than dropping the row.
    usecols = list(dict.fromkeys(segment_cols + ["Sale", "time_delay_for_conversion"]))
    df = pd.read_csv(
        raw_path,
        sep="\t",
        names=COLUMNS,
        usecols=usecols,
        nrows=max_rows,
        na_values=["", "-1"],
        low_memory=False,
    )

    df["Sale"] = (df["Sale"].fillna(0).astype(float) > 0).astype(int)

    # --- delay asset: real positive conversion delays -> steps -------------
    delay_s = pd.to_numeric(df["time_delay_for_conversion"], errors="coerce")
    positive = delay_s[(df["Sale"] == 1) & (delay_s.notna()) & (delay_s >= 0)]
    delays_steps = np.maximum(0.0, positive.to_numpy() / seconds_per_step)
    conversion_rate = float(df["Sale"].mean())

    np.savez(
        ASSETS_DIR / f"criteo_{instance}_delay.npz",
        delays_steps=delays_steps.astype(np.float32),
        conversion_rate=np.float32(conversion_rate),
        seconds_per_step=np.float32(seconds_per_step),
        n_clicks=np.int64(len(df)),
        n_conversions=np.int64(int(df["Sale"].sum())),
    )

    # --- arm asset: per-segment conversion rate ----------------------------
    seg = df[segment_cols].astype("string").fillna("NA")
    df["segment"] = seg.agg("|".join, axis=1)
    grp = df.groupby("segment")["Sale"].agg(["mean", "count"])
    grp = grp[grp["count"] >= min_count].reset_index()
    grp = grp.rename(columns={"mean": "conversion_rate"})
    grp = grp.sort_values("conversion_rate", ascending=False)
    grp[["segment", "conversion_rate", "count"]].to_csv(
        ASSETS_DIR / f"criteo_{instance}_arms.csv", index=False
    )

    meta = {
        "instance": instance,
        "segment_cols": segment_cols,
        "min_count": min_count,
        "seconds_per_step": seconds_per_step,
        "n_arms": int(len(grp)),
        "n_clicks": int(len(df)),
        "conversion_rate": conversion_rate,
        "best_conversion_rate": float(grp["conversion_rate"].max()),
        "median_delay_steps": (
            float(np.median(delays_steps)) if len(delays_steps) else 0.0
        ),
        "mean_delay_steps": float(delays_steps.mean()) if len(delays_steps) else 0.0,
    }
    (ASSETS_DIR / f"criteo_{instance}_arms_meta.json").write_text(
        json.dumps(meta, indent=2)
    )

    print(json.dumps(meta, indent=2))
    print(
        f"\n-> {len(grp)} arms (segments with >= {min_count} clicks), "
        f"overall conversion rate {conversion_rate:.4f} "
        f"(base reward mean; NOT a censoring rate -- non-conversions are "
        f"observed 0s), "
        f"{len(delays_steps)} positive delays."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_FILE,
        help=f"Path to CriteoSearchData file or its directory (default: {DEFAULT_RAW_FILE})",
    )
    ap.add_argument("--instance", default="sponsored_search")
    ap.add_argument("--segment-cols", nargs="+", default=DEFAULT_SEGMENT_COLS)
    ap.add_argument(
        "--min-count",
        type=int,
        default=200,
        help="Keep only segments with at least this many clicks (reliable rate).",
    )
    ap.add_argument("--seconds-per-step", type=float, default=_SECONDS_PER_STEP)
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows read (for a quick smaller-sample build).",
    )
    args = ap.parse_args()
    raw_path = resolve_raw_path(args.raw)
    build(
        raw_path,
        args.instance,
        args.segment_cols,
        args.min_count,
        args.seconds_per_step,
        args.max_rows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
