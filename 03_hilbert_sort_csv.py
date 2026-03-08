#!/usr/bin/env python3

import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def xy2hilbert(x: int, y: int, order: int) -> int:
    d = 0
    n = 1 << order
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d


def normalize_to_hilbert_grid(values: np.ndarray, order: int) -> np.ndarray:
    grid_size = (1 << order) - 1
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.int64)
    scaled = (values - vmin) / (vmax - vmin)
    coords = np.rint(scaled * grid_size).astype(np.int64)
    return np.clip(coords, 0, grid_size)


def choose_dense_hw(n_locations: int) -> Tuple[int, int]:
    h = int(math.floor(math.sqrt(n_locations)))
    w = int(math.ceil(n_locations / h))
    while h * w < n_locations:
        h += 1
    return h, w


def parse_feature_args(features: Optional[List[str]]) -> Optional[List[str]]:
    if features is None:
        return None
    cleaned = []
    for f in features:
        cleaned.extend([x.strip() for x in f.split(',') if x.strip()])
    return cleaned or None


def build_dense_layout(unique_locs: pd.DataFrame) -> Tuple[pd.DataFrame, int, int, int]:
    lat = unique_locs["latitude"].to_numpy(dtype=np.float64, copy=False)
    lon = unique_locs["longitude"].to_numpy(dtype=np.float64, copy=False)
    n_locations = len(unique_locs)
    if n_locations == 0:
        raise ValueError("No unique locations available for Hilbert layout.")

    order = max(1, math.ceil(math.log2(max(2, math.ceil(math.sqrt(n_locations))))))
    xq = normalize_to_hilbert_grid(lon, order)
    yq = normalize_to_hilbert_grid(lat, order)

    hilbert = np.fromiter((xy2hilbert(int(x), int(y), order) for x, y in zip(xq, yq)), dtype=np.int64, count=n_locations)

    layout = unique_locs.copy()
    layout["lat_q"] = yq
    layout["lon_q"] = xq
    layout["hilbert_index"] = hilbert
    layout = layout.sort_values(["hilbert_index", "latitude", "longitude"], kind="mergesort").reset_index(drop=True)

    H, W = choose_dense_hw(n_locations)
    packed = np.arange(n_locations, dtype=np.int64)
    layout["location_id"] = packed
    layout["y"] = packed // W
    layout["x"] = packed % W
    return layout, H, W, order


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add dense Hilbert spatial coordinates to a normalized CSV and sort by time, x, y."
    )
    parser.add_argument("--input", required=True, help="Normalized CSV path")
    parser.add_argument("--output", required=True, help="Hilbert-sorted output CSV path")
    parser.add_argument("--layout-out", required=True, help="Location layout CSV output path")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature columns to keep. If omitted, all columns in the CSV are preserved.",
    )
    args = parser.parse_args()

    features = parse_feature_args(args.features)
    preview_cols = pd.read_csv(args.input, nrows=0).columns.tolist()
    required = ["time", "latitude", "longitude"]
    for col in required:
        if col not in preview_cols:
            raise ValueError(f"Required column '{col}' not found in input CSV")

    if features is None:
        usecols = preview_cols
    else:
        missing = [c for c in features if c not in preview_cols]
        if missing:
            raise ValueError(f"Requested feature columns missing from CSV: {missing}")
        usecols = required + features

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.layout_out)), exist_ok=True)

    df = pd.read_csv(args.input, usecols=usecols)

    unique_locs = df[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    layout_df, H, W, order = build_dense_layout(unique_locs)

    loc_index = pd.Index(pd.MultiIndex.from_frame(layout_df[["latitude", "longitude"]]))
    row_keys = pd.MultiIndex.from_frame(df[["latitude", "longitude"]])
    spatial_pos = loc_index.get_indexer(row_keys)

    if np.any(spatial_pos < 0):
        raise RuntimeError("Some rows could not be mapped to Hilbert layout coordinates.")

    df["location_id"] = layout_df["location_id"].to_numpy(dtype=np.int64)[spatial_pos]
    df["hilbert_index"] = layout_df["hilbert_index"].to_numpy(dtype=np.int64)[spatial_pos]
    df["x"] = layout_df["x"].to_numpy(dtype=np.int64)[spatial_pos]
    df["y"] = layout_df["y"].to_numpy(dtype=np.int64)[spatial_pos]

    # The user requested increasing timestep, increasing x, then increasing y.
    df = df.sort_values(["time", "x", "y"], kind="mergesort").reset_index(drop=True)
    df.to_csv(args.output, index=False)

    layout_df.to_csv(args.layout_out, index=False)

    print(f"Hilbert-sorted CSV written to: {args.output}")
    print(f"Location layout CSV written to: {args.layout_out}")
    print(f"Dense spatial shape: H={H}, W={W}")
    print(f"Unique locations: {len(layout_df)}")
    print(f"Hilbert order: {order}")


if __name__ == "__main__":
    main()
