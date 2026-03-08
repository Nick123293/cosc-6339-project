#!/usr/bin/env python3

import argparse
import os
from typing import List, Optional

import pandas as pd


DEFAULT_HOUSTON_LAT = 29.7604
DEFAULT_HOUSTON_LON = -95.3698


def parse_feature_args(features: Optional[List[str]]) -> Optional[List[str]]:
    if features is None:
        return None
    cleaned = []
    for f in features:
        cleaned.extend([x.strip() for x in f.split(',') if x.strip()])
    return cleaned or None


def build_usecols(features: Optional[List[str]]) -> Optional[List[str]]:
    base = ["time", "latitude", "longitude"]
    if features is None:
        return None
    return base + features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a CSV to keep only rows within a squared lat/lon radius of Houston."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Filtered output CSV path")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature columns to keep in addition to time, latitude, longitude. Can be space-separated or comma-separated.",
    )
    parser.add_argument(
        "--houston-lat", type=float, default=DEFAULT_HOUSTON_LAT, help="Houston center latitude"
    )
    parser.add_argument(
        "--houston-lon", type=float, default=DEFAULT_HOUSTON_LON, help="Houston center longitude"
    )
    parser.add_argument(
        "--radius-squared",
        type=float,
        required=True,
        help="Keep rows with (lat-h_lat)^2 + (lon-h_lon)^2 <= radius_squared",
    )
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="CSV chunksize for streaming"
    )
    args = parser.parse_args()

    features = parse_feature_args(args.features)
    usecols = build_usecols(features)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    dtype_map = {
        "latitude": "float64",
        "longitude": "float64",
    }

    first_write = True
    total_in = 0
    total_out = 0

    reader = pd.read_csv(
        args.input,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=args.chunksize,
    )

    for chunk in reader:
        total_in += len(chunk)
        dist2 = (
            (chunk["latitude"] - args.houston_lat) ** 2
            + (chunk["longitude"] - args.houston_lon) ** 2
        )
        kept = chunk.loc[dist2 <= args.radius_squared]
        total_out += len(kept)

        if not kept.empty:
            kept.to_csv(args.output, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False

    if first_write:
        # No rows matched: still create an empty CSV with header.
        preview = pd.read_csv(args.input, nrows=0, usecols=usecols)
        preview.to_csv(args.output, index=False)

    print(f"Input rows read: {total_in}")
    print(f"Rows kept: {total_out}")
    print(f"Filtered CSV written to: {args.output}")


if __name__ == "__main__":
    main()
