#!/usr/bin/env python3

import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


REQUIRED_BASE_COLS = ["time", "x", "y"]


def parse_feature_args(features: Optional[List[str]]) -> Optional[List[str]]:
    if features is None:
        return None
    cleaned = []
    for f in features:
        cleaned.extend([x.strip() for x in f.split(',') if x.strip()])
    return cleaned or None


def infer_features(csv_path: str) -> List[str]:
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    exclude = {"time", "latitude", "longitude", "x", "y", "location_id", "hilbert_index", "lat_q", "lon_q"}
    features = [c for c in cols if c not in exclude]
    return features


def normalize_splits(train_pct: float, val_pct: float, test_pct: float) -> Tuple[float, float, float]:
    total = train_pct + val_pct + test_pct
    if total <= 0:
        raise ValueError("Split percentages must sum to a positive number.")
    return train_pct / total, val_pct / total, test_pct / total


def compute_time_boundaries(n_times: int, train_pct: float, val_pct: float) -> Tuple[int, int]:
    train_n = int(math.floor(n_times * train_pct))
    val_n = int(math.floor(n_times * val_pct))

    if train_n <= 0 and n_times > 0:
        train_n = 1
    if train_n + val_n >= n_times and n_times >= 3:
        val_n = max(1, n_times - train_n - 1)

    split1 = min(train_n, n_times)
    split2 = min(train_n + val_n, n_times)
    return split1, split2


def build_tensor_for_indices(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    selected_time_values: Sequence,
    H: int,
    W: int,
    fill_values: Sequence[float],
    out_dtype: np.dtype,
) -> Tuple[torch.Tensor, List[str]]:
    split_df = df[df["time"].isin(selected_time_values)].copy()
    split_df = split_df.sort_values(["time", "x", "y"], kind="mergesort")

    time_order = list(pd.unique(split_df["time"]))
    time_to_idx = {t: i for i, t in enumerate(time_order)}

    T = len(time_order)
    C = len(feature_columns)
    fill_array = np.asarray(fill_values, dtype=out_dtype)
    if fill_array.shape != (C,):
        raise ValueError(f"Expected {C} fill values, got shape {fill_array.shape}")

    tensor_np = np.broadcast_to(fill_array.reshape(1, C, 1, 1), (T, C, H, W)).copy()

    if T == 0:
        return torch.from_numpy(tensor_np), []

    t_idx = split_df["time"].map(time_to_idx).to_numpy(dtype=np.int64, copy=False)
    x_idx = split_df["x"].to_numpy(dtype=np.int64, copy=False)
    y_idx = split_df["y"].to_numpy(dtype=np.int64, copy=False)
    values = split_df[list(feature_columns)].to_numpy(dtype=out_dtype, copy=False).T

    c_idx = np.arange(C, dtype=np.int64)[:, None]
    tensor_np[t_idx[None, :], c_idx, y_idx[None, :], x_idx[None, :]] = values

    return torch.from_numpy(tensor_np), [str(t) for t in time_order]


def save_split(
    split_name: str,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    selected_time_values: Sequence,
    H: int,
    W: int,
    fill_values: Sequence[float],
    out_dtype: np.dtype,
    output_path: str,
) -> str:
    tensor, times = build_tensor_for_indices(df, feature_columns, selected_time_values, H, W, fill_values, out_dtype)
    payload = {
        "tensor": tensor,
        "times": times,
        "feature_columns": list(feature_columns),
        "H": H,
        "W": W,
        "split": split_name,
    }
    torch.save(payload, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a Hilbert-sorted CSV into train/val/test tensors with shape [T,C,H,W]."
    )
    parser.add_argument("--input", required=True, help="Hilbert-sorted CSV path")
    parser.add_argument("--train-out", required=True, help="Training tensor output .pt path")
    parser.add_argument("--val-out", required=True, help="Validation tensor output .pt path")
    parser.add_argument("--test-out", required=True, help="Testing tensor output .pt path")
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Feature columns to place in the tensor. If omitted, infer all non-metadata columns.",
    )
    parser.add_argument("--train-pct", type=float, default=0.70, help="Training timestamp fraction")
    parser.add_argument("--val-pct", type=float, default=0.15, help="Validation timestamp fraction")
    parser.add_argument("--test-pct", type=float, default=0.15, help="Testing timestamp fraction")
    parser.add_argument(
        "--fill-value",
        type=float,
        default=float("0.0"),
        help="Fallback fill value used only if a feature column mean cannot be computed.",
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="Tensor floating-point dtype")
    parser.add_argument("--workers", type=int, default=3, help="Number of threads used to build/save the three split tensors")
    args = parser.parse_args()

    features = parse_feature_args(args.features)
    if features is None:
        features = infer_features(args.input)

    preview_cols = pd.read_csv(args.input, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED_BASE_COLS + list(features) if c not in preview_cols]
    if missing:
        raise ValueError(f"Required columns missing from input CSV: {missing}")

    out_dtype = np.float32 if args.dtype == "float32" else np.float64

    os.makedirs(os.path.dirname(os.path.abspath(args.train_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.val_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.test_out)), exist_ok=True)

    usecols = REQUIRED_BASE_COLS + list(features)
    dtype_map: Dict[str, str] = {"x": "int64", "y": "int64"}
    for c in features:
        dtype_map[c] = args.dtype

    df = pd.read_csv(args.input, usecols=usecols, dtype=dtype_map)
    time_values = list(pd.unique(df["time"]))

    train_pct, val_pct, test_pct = normalize_splits(args.train_pct, args.val_pct, args.test_pct)
    split1, split2 = compute_time_boundaries(len(time_values), train_pct, val_pct)

    train_times = time_values[:split1]
    val_times = time_values[split1:split2]
    test_times = time_values[split2:]

    H = int(df["y"].max()) + 1
    W = int(df["x"].max()) + 1

    feature_fill_values = df[list(features)].mean(axis=0, skipna=True).to_numpy(dtype=out_dtype, copy=True)
    invalid_fill_mask = ~np.isfinite(feature_fill_values)
    if np.any(invalid_fill_mask):
        feature_fill_values[invalid_fill_mask] = out_dtype.type(args.fill_value)

    jobs = [
        ("train", train_times, args.train_out),
        ("val", val_times, args.val_out),
        ("test", test_times, args.test_out),
    ]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(save_split, name, df, features, times, H, W, feature_fill_values, out_dtype, out_path)
            for name, times, out_path in jobs
        ]
        results = [f.result() for f in futures]

    print(f"Training tensor saved to: {args.train_out}")
    print(f"Validation tensor saved to: {args.val_out}")
    print(f"Testing tensor saved to: {args.test_out}")
    print(f"Tensor shape convention: [T, C, H, W]")
    print(f"Features ({len(features)}): {features}")
    print(f"Spatial shape: H={H}, W={W}")
    print("Per-feature fill values (channel means):")
    for feature_name, fill in zip(features, feature_fill_values.tolist()):
        print(f"  {feature_name}: {fill}")
    print(f"Timestamp counts -> train: {len(train_times)}, val: {len(val_times)}, test: {len(test_times)}")


if __name__ == "__main__":
    main()
