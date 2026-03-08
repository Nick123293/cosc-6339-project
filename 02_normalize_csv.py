#!/usr/bin/env python3

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


EXCLUDE_FROM_NORMALIZATION = {"time"}


def parse_feature_args(features: Optional[List[str]]) -> Optional[List[str]]:
    if features is None:
        return None
    cleaned = []
    for f in features:
        cleaned.extend([x.strip() for x in f.split(',') if x.strip()])
    return cleaned or None


def infer_numeric_columns(csv_path: str, requested_features: Optional[List[str]]) -> List[str]:
    preview = pd.read_csv(csv_path, nrows=1000)
    if requested_features is not None:
        needed = ["latitude", "longitude"] + requested_features
        missing = [c for c in needed if c not in preview.columns]
        if missing:
            raise ValueError(f"Requested columns missing from CSV: {missing}")
        return needed

    numeric_cols = [
        c for c in preview.columns
        if c not in EXCLUDE_FROM_NORMALIZATION and pd.api.types.is_numeric_dtype(preview[c])
    ]
    required = ["latitude", "longitude"]
    for col in required:
        if col not in numeric_cols:
            numeric_cols.append(col)
    return numeric_cols


def get_float_settings(dtype_name: str):
    if dtype_name == "float32":
        return np.float32, "%.9g"
    if dtype_name == "float64":
        return np.float64, "%.17g"
    raise ValueError("dtype must be float32 or float64")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize numeric CSV columns independently to [0,1]."
    )
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Normalized output CSV path")
    parser.add_argument(
        "--stats-out",
        required=True,
        help="Output CSV containing min/max statistics used for normalization",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature columns to normalize in addition to latitude and longitude. If omitted, normalize all numeric columns except time.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point precision for normalized values",
    )
    parser.add_argument(
        "--chunksize", type=int, default=200_000, help="CSV chunksize for streaming"
    )
    args = parser.parse_args()

    features = parse_feature_args(args.features)
    norm_cols = infer_numeric_columns(args.input, features)
    out_dtype, float_format = get_float_settings(args.dtype)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)), exist_ok=True)

    usecols = None
    preview_cols = pd.read_csv(args.input, nrows=0).columns.tolist()
    for required in ["time", *norm_cols]:
        if required not in preview_cols:
            raise ValueError(f"Required column '{required}' not found in input CSV")

    min_vals: Dict[str, float] = {c: np.inf for c in norm_cols}
    max_vals: Dict[str, float] = {c: -np.inf for c in norm_cols}

    dtype_map = {c: "float64" for c in norm_cols}

    # Pass 1: compute columnwise min/max.
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize, dtype=dtype_map):
        for col in norm_cols:
            values = chunk[col].to_numpy(dtype=np.float64, copy=False)
            if values.size == 0:
                continue
            local_min = np.nanmin(values)
            local_max = np.nanmax(values)
            if local_min < min_vals[col]:
                min_vals[col] = local_min
            if local_max > max_vals[col]:
                max_vals[col] = local_max

    stats_df = pd.DataFrame({
        "column": norm_cols,
        "min": [min_vals[c] for c in norm_cols],
        "max": [max_vals[c] for c in norm_cols],
    })
    stats_df.to_csv(args.stats_out, index=False, float_format=float_format)

    # Pass 2: normalize and write.
    first_write = True
    for chunk in pd.read_csv(args.input, chunksize=args.chunksize, dtype=dtype_map):
        for col in norm_cols:
            cmin = min_vals[col]
            cmax = max_vals[col]
            values = chunk[col].to_numpy(dtype=np.float64, copy=False)
            if np.isclose(cmin, cmax):
                normalized = np.zeros(values.shape, dtype=out_dtype)
            else:
                normalized = ((values - cmin) / (cmax - cmin)).astype(out_dtype, copy=False)
            chunk[col] = normalized

        chunk.to_csv(
            args.output,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
            float_format=float_format,
        )
        first_write = False

    print(f"Normalized CSV written to: {args.output}")
    print(f"Normalization stats written to: {args.stats_out}")
    print(f"Normalized columns: {norm_cols}")


if __name__ == "__main__":
    main()
