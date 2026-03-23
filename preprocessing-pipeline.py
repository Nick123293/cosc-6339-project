#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


# =============================================================================
# Utility helpers
# =============================================================================

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_csv_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    out: List[str] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def detect_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def standardize_time_series(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.replace("T", " ", regex=False)
    raw = raw.str.replace(r"([+-]\d{2}:\d{2}|Z)$", "", regex=True).str.strip()
    dt = pd.to_datetime(raw, errors="coerce")
    return dt.dt.strftime("%Y-%m-%d %H:%M:%S")


def coerce_zip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d{5})", expand=False)


def numeric_feature_columns(df: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    excl = set(exclude)
    out = []
    for c in df.columns:
        if c in excl:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def json_dump(path: str, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


class PipelineLogger:
    def __init__(self, path: str):
        self.path = path
        self.lines: List[str] = []

    def section(self, title: str) -> None:
        self.lines.append("\n" + "=" * 100)
        self.lines.append(title)
        self.lines.append("=" * 100)

    def add(self, msg: str = "") -> None:
        self.lines.append(msg)

    def kv(self, key: str, value) -> None:
        if isinstance(value, (dict, list, tuple)):
            self.lines.append(f"{key}: {json.dumps(value, indent=2, default=str)}")
        else:
            self.lines.append(f"{key}: {value}")

    def write(self) -> None:
        Path(self.path).write_text("\n".join(self.lines), encoding="utf-8")


# =============================================================================
# Hilbert helpers
# =============================================================================

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
        x, y = rot(n, x, y, rx, ry)
        s //= 2
    return d


def scale_to_grid(v: float, lo: float, hi: float, n: int) -> int:
    if hi <= lo:
        return 0
    t = (v - lo) / (hi - lo)
    return max(0, min(n - 1, int(round(t * (n - 1)))))


# =============================================================================
# Step 1: Merge weather + air quality
# =============================================================================

def step1_merge(
    air_quality_csv: str,
    weather_csv: str,
    output_csv: str,
    time_col: str,
    zip_col: str,
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 1 - Merge air quality and weather CSVs")

    aq = pd.read_csv(air_quality_csv)
    wx = pd.read_csv(weather_csv)

    if time_col not in aq.columns:
        raise ValueError(f"Air quality CSV missing time column: {time_col}")
    if zip_col not in aq.columns:
        raise ValueError(f"Air quality CSV missing zip column: {zip_col}")
    if time_col not in wx.columns:
        raise ValueError(f"Weather CSV missing time column: {time_col}")
    if zip_col not in wx.columns:
        raise ValueError(f"Weather CSV missing zip column: {zip_col}")

    aq = aq.copy()
    wx = wx.copy()

    aq[time_col] = standardize_time_series(aq[time_col])
    wx[time_col] = standardize_time_series(wx[time_col])

    aq[zip_col] = coerce_zip_series(aq[zip_col])
    wx[zip_col] = coerce_zip_series(wx[zip_col])

    join_keys = [time_col, zip_col]
    duplicate_nonkeys = sorted(set(aq.columns).intersection(wx.columns) - set(join_keys))
    wx_keep = [c for c in wx.columns if c not in duplicate_nonkeys]

    merged = aq.merge(wx[wx_keep], on=join_keys, how="inner")
    merged.to_csv(output_csv, index=False)

    logger.kv("air_quality_csv", air_quality_csv)
    logger.kv("weather_csv", weather_csv)
    logger.kv("join_keys", join_keys)
    logger.kv("dropped_duplicate_weather_columns", duplicate_nonkeys)
    logger.kv("rows_air_quality", int(len(aq)))
    logger.kv("rows_weather", int(len(wx)))
    logger.kv("rows_merged", int(len(merged)))
    logger.kv("output_csv", output_csv)

    return {
        "air_quality_columns": aq.columns.tolist(),
        "weather_columns": wx.columns.tolist(),
        "merged_columns": merged.columns.tolist(),
        "dropped_duplicate_weather_columns": duplicate_nonkeys,
        "rows_air_quality": int(len(aq)),
        "rows_weather": int(len(wx)),
        "rows_merged": int(len(merged)),
        "output_csv": output_csv,
    }


# =============================================================================
# Step 2: Spatial impact score
# =============================================================================

def _pick_tri_facility_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "facility_id": detect_column(df.columns, [
            "TRIFID",
            "TRIFD",
            "trifid",
            "trifd",
            "facility_id",
            "tri_facility_id",
        ]),
        "lat": detect_column(df.columns, [
            "LATITUDE83",
            "LATITUDE",
            "latitude",
            "lat",
        ]),
        "lon": detect_column(df.columns, [
            "LONGITUDE83",
            "LONGITUDE",
            "longitude",
            "lon",
        ]),
        "facility_name": detect_column(df.columns, [
            "FACILITY NAME",
            "FAC_NAME",
            "facility_name",
            "facility",
            "name",
        ]),
    }


def _pick_tri_chemical_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "facility_id": detect_column(df.columns, [
            "TRIFID",
            "TRIFD",
            "trifid",
            "trifd",
            "facility_id",
            "tri_facility_id",
        ]),
        "chemical_name": detect_column(df.columns, [
            "CHEMICAL",
            "CHEMICAL NAME",
            "chemical_name",
            "chemical",
        ]),
        "amount": detect_column(df.columns, [
            "TOTAL RELEASE",
            "TOTAL_RELEASE",
            "total_release",
            "ON-SITE RELEASE TOTAL",
            "ONSITE_RELEASE_TOTAL",
            "TOTAL WASTE",
            "TOTAL_WASTE",
            "total_air_emissions_lbs",
            "TOTAL_AIR_EMISSIONS_LBS",
            "stack_lbs",
            "fugitive_lbs",
        ]),
    }


def step2_spatial_impact(
    merged_csv: str,
    tri_facilities_csv: str,
    tri_chemicals_csv: str,
    zip_shapefile: str,
    roads_shapefile: str,
    output_csv: str,
    metadata_json: str,
    zip_col: str,
    logger: PipelineLogger,
    road_radius_km: float = 2.0,
    facility_radius_km: float = 10.0,
) -> Dict:
    logger.section("Step 2 - Spatial impact scoring from roads + TRI facilities")

    df = pd.read_csv(merged_csv)
    df[zip_col] = coerce_zip_series(df[zip_col])
    zips_needed = set(df[zip_col].dropna().unique().tolist())

    zcta = gpd.read_file(zip_shapefile)
    zcta_zip_col = detect_column(
        zcta.columns,
        ["ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "zip", "zcta"]
    )
    if zcta_zip_col is None:
        raise ValueError("Could not detect ZIP/ZCTA column in ZIP shapefile.")

    zcta = zcta.copy()
    zcta["zip_norm"] = coerce_zip_series(zcta[zcta_zip_col])
    zcta = zcta[zcta["zip_norm"].isin(zips_needed)].copy()

    if zcta.empty:
        raise ValueError("None of the merged ZIP codes were found in the ZCTA shapefile.")

    roads = gpd.read_file(roads_shapefile)

    fac = pd.read_csv(tri_facilities_csv)
    chem = pd.read_csv(tri_chemicals_csv)

    fac_cols = _pick_tri_facility_columns(fac)
    chem_cols = _pick_tri_chemical_columns(chem)

    if fac_cols["facility_id"] is None or fac_cols["lat"] is None or fac_cols["lon"] is None:
        raise ValueError(f"Could not detect required TRI facility columns. Detected: {fac_cols}")
    if chem_cols["facility_id"] is None or chem_cols["chemical_name"] is None:
        raise ValueError(f"Could not detect required TRI chemical columns. Detected: {chem_cols}")

    facility_amount_col = detect_column(fac.columns, [
        "total_air_emissions_lbs",
        "TOTAL_AIR_EMISSIONS_LBS",
        "TOTAL RELEASE",
        "TOTAL_RELEASE",
    ])

    fac_keep = [fac_cols["facility_id"], fac_cols["lat"], fac_cols["lon"]]
    if fac_cols["facility_name"] is not None:
        fac_keep.append(fac_cols["facility_name"])
    if facility_amount_col is not None:
        fac_keep.append(facility_amount_col)

    fac_use = fac[fac_keep].copy()
    rename_map = {
        fac_cols["facility_id"]: "facility_id",
        fac_cols["lat"]: "lat",
        fac_cols["lon"]: "lon",
    }
    if fac_cols["facility_name"] is not None:
        rename_map[fac_cols["facility_name"]] = "facility_name"
    if facility_amount_col is not None:
        rename_map[facility_amount_col] = "facility_amount"

    fac_use = fac_use.rename(columns=rename_map)
    fac_use["lat"] = pd.to_numeric(fac_use["lat"], errors="coerce")
    fac_use["lon"] = pd.to_numeric(fac_use["lon"], errors="coerce")
    if "facility_amount" in fac_use.columns:
        fac_use["facility_amount"] = pd.to_numeric(fac_use["facility_amount"], errors="coerce").fillna(0.0)
    fac_use = fac_use.dropna(subset=["lat", "lon"]).copy()

    chem_keep = [chem_cols["facility_id"], chem_cols["chemical_name"]]
    if chem_cols["amount"] is not None:
        chem_keep.append(chem_cols["amount"])

    chem_use = chem[chem_keep].copy()
    chem_use.columns = ["facility_id", "chemical_name"] + (["amount"] if chem_cols["amount"] is not None else [])

    if "amount" not in chem_use.columns:
        chem_use["amount"] = 1.0
    chem_use["amount"] = pd.to_numeric(chem_use["amount"], errors="coerce").fillna(0.0)

    chem_total_amount = chem_use.groupby("facility_id", as_index=False)["amount"].sum()
    chem_unique_count = (
        chem_use.groupby("facility_id", as_index=False)["chemical_name"]
        .nunique()
        .rename(columns={"chemical_name": "chemical_count"})
    )

    fac_enriched = fac_use.merge(chem_total_amount, on="facility_id", how="left")
    fac_enriched = fac_enriched.merge(chem_unique_count, on="facility_id", how="left")

    if "facility_amount" in fac_enriched.columns:
        fac_enriched["amount"] = fac_enriched["facility_amount"]
        if "amount_x" in fac_enriched.columns:
            fac_enriched["amount"] = fac_enriched["amount"].fillna(fac_enriched["amount_x"])
        elif "amount_y" in fac_enriched.columns:
            fac_enriched["amount"] = fac_enriched["amount"].fillna(fac_enriched["amount_y"])
    else:
        if "amount" not in fac_enriched.columns:
            fac_enriched["amount"] = 0.0

    fac_enriched["amount"] = pd.to_numeric(fac_enriched["amount"], errors="coerce").fillna(0.0)
    fac_enriched["chemical_count"] = pd.to_numeric(fac_enriched["chemical_count"], errors="coerce").fillna(0.0)

    fac_gdf = gpd.GeoDataFrame(
        fac_enriched,
        geometry=gpd.points_from_xy(fac_enriched["lon"], fac_enriched["lat"]),
        crs="EPSG:4326",
    )

    target_crs = "EPSG:3857"
    zcta_m = zcta.to_crs(target_crs)
    roads_m = roads.to_crs(target_crs)
    fac_m = fac_gdf.to_crs(target_crs)

    zcta_cent = zcta_m.copy()
    zcta_cent["geometry"] = zcta_cent.geometry.centroid
    zcta_cent = zcta_cent[["zip_norm", "geometry"]].copy()

    road_union = roads_m.geometry.union_all()
    road_dist_m = zcta_cent.geometry.distance(road_union)
    road_radius_m = road_radius_km * 1000.0

    road_df = pd.DataFrame({
        "zip_norm": zcta_cent["zip_norm"].tolist(),
        "road_distance_m": road_dist_m.values,
    })
    road_df["road_score_raw"] = np.exp(-road_df["road_distance_m"] / road_radius_m)
    road_max = max(float(road_df["road_score_raw"].max()), 1e-12)
    road_df["road_impact_score"] = road_df["road_score_raw"] / road_max

    facility_radius_m = facility_radius_km * 1000.0

    cross = (
        zcta_cent.rename(columns={"geometry": "zip_geom"})
        .merge(
            fac_m[["facility_id", "amount", "chemical_count", "geometry"]].rename(columns={"geometry": "fac_geom"}),
            how="cross"
        )
    )

    cross["dist_m"] = cross.apply(lambda r: r["zip_geom"].distance(r["fac_geom"]), axis=1)
    cross = cross[cross["dist_m"] <= facility_radius_m].copy()

    if cross.empty:
        facility_df = pd.DataFrame({
            "zip_norm": zcta_cent["zip_norm"].tolist(),
            "facility_raw": 0.0,
            "facility_count_nearby": 0,
        })
    else:
        amount_scale = cross["amount"].quantile(0.95)
        if not np.isfinite(amount_scale) or amount_scale <= 0:
            amount_scale = max(float(cross["amount"].max()), 1.0)

        chem_scale = max(float(cross["chemical_count"].max()), 1.0)

        cross["amount_norm"] = np.clip(cross["amount"] / amount_scale, 0, 1)
        cross["chem_norm"] = np.clip(cross["chemical_count"] / chem_scale, 0, 1)
        cross["severity"] = 0.7 * cross["amount_norm"] + 0.3 * cross["chem_norm"]
        cross["decay"] = np.exp(-cross["dist_m"] / facility_radius_m)
        cross["pair_impact"] = cross["severity"] * cross["decay"]

        facility_df = (
            cross.groupby("zip_norm", as_index=False)
            .agg(
                facility_raw=("pair_impact", "sum"),
                facility_count_nearby=("facility_id", "nunique"),
            )
        )

    fac_max = max(float(facility_df["facility_raw"].max()), 1e-12)
    facility_df["facility_impact_score"] = facility_df["facility_raw"] / fac_max

    score_df = (
        road_df[["zip_norm", "road_distance_m", "road_impact_score"]]
        .merge(
            facility_df[["zip_norm", "facility_raw", "facility_impact_score", "facility_count_nearby"]],
            on="zip_norm",
            how="left"
        )
        .fillna({"facility_raw": 0.0, "facility_impact_score": 0.0, "facility_count_nearby": 0})
    )

    score_df["overall_spatial_impact_score"] = 0.4 * score_df["road_impact_score"] + 0.6 * score_df["facility_impact_score"]
    score_df["overall_spatial_impact_score"] = np.clip(score_df["overall_spatial_impact_score"], 0, 1)

    out = (
        df.merge(
            score_df[
                [
                    "zip_norm",
                    "road_distance_m",
                    "road_impact_score",
                    "facility_count_nearby",
                    "facility_impact_score",
                    "overall_spatial_impact_score",
                ]
            ],
            left_on=zip_col,
            right_on="zip_norm",
            how="left"
        )
        .drop(columns=["zip_norm"])
    )

    out.to_csv(output_csv, index=False)

    meta = {
        "input_files": {
            "merged_csv": merged_csv,
            "tri_facilities_csv": tri_facilities_csv,
            "tri_chemicals_csv": tri_chemicals_csv,
            "zip_shapefile": zip_shapefile,
            "roads_shapefile": roads_shapefile,
        },
        "detected_columns": {
            "tri_facilities": fac_cols,
            "tri_chemicals": chem_cols,
            "zcta_zip_column": zcta_zip_col,
            "facility_amount_column": facility_amount_col,
        },
        "added_columns": [
            "road_distance_m",
            "road_impact_score",
            "facility_count_nearby",
            "facility_impact_score",
            "overall_spatial_impact_score",
        ],
        "impact_formulas": {
            "road_impact_score": "exp(-distance_to_nearest_road_m / road_radius_m), then divided by max over ZIPs",
            "facility_severity": "0.7 * normalized_total_release + 0.3 * normalized_unique_chemical_count",
            "facility_pair_impact": "facility_severity * exp(-distance_zip_to_facility_m / facility_radius_m)",
            "facility_impact_score": "sum(facility_pair_impact over nearby facilities), then divided by max over ZIPs",
            "overall_spatial_impact_score": "0.4 * road_impact_score + 0.6 * facility_impact_score",
        },
        "parameters": {
            "road_radius_km": road_radius_km,
            "facility_radius_km": facility_radius_km,
        }
    }
    json_dump(metadata_json, meta)

    logger.kv("zip_codes_scored", int(score_df["zip_norm"].nunique()))
    logger.kv("roads_feature_count", int(len(roads)))
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 3: Expand direction columns to sin/cos
# =============================================================================

def step3_expand_direction_columns(
    input_csv: str,
    output_csv: str,
    metadata_json: str,
    logger: PipelineLogger,
    direction_columns: Optional[List[str]] = None,
    auto_detect: bool = True,
    drop_original: bool = True
) -> Dict:
    logger.section("Step 3 - Expand directional columns into sine/cosine")

    df = pd.read_csv(input_csv)

    detected = []
    if auto_detect:
        for c in df.columns:
            cl = c.lower()
            if "direction" in cl or "wind_dir" in cl or "winddirection" in cl:
                detected.append(c)

    user_cols = direction_columns or []
    final_cols = []
    seen = set()

    for c in detected + user_cols:
        if c in df.columns and c not in seen:
            final_cols.append(c)
            seen.add(c)

    expanded = []
    out = df.copy()

    for c in final_cols:
        s = pd.to_numeric(out[c], errors="coerce")
        radians = np.deg2rad(s)
        sin_col = f"{c}_sin"
        cos_col = f"{c}_cos"

        out[sin_col] = np.sin(radians)
        out[cos_col] = np.cos(radians)

        if drop_original:
            out = out.drop(columns=[c])

        expanded.append({
            "original_column": c,
            "sin_column": sin_col,
            "cos_column": cos_col,
            "dropped_original": drop_original,
        })

    out.to_csv(output_csv, index=False)

    meta = {
        "auto_detect": auto_detect,
        "user_requested_direction_columns": user_cols,
        "expanded_columns": expanded,
        "output_csv": output_csv,
    }
    json_dump(metadata_json, meta)

    logger.kv("detected_direction_columns", detected)
    logger.kv("user_requested_direction_columns", user_cols)
    logger.kv("expanded_columns", expanded)
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 4: Normalize
# =============================================================================

def step4_normalize(
    input_csv: str,
    output_csv: str,
    stats_json: str,
    exclude_cols: List[str],
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 4 - Min-max normalization")

    df = pd.read_csv(input_csv)
    features = numeric_feature_columns(df, exclude_cols)

    stats = {}
    out = df.copy()

    for c in features:
        s = pd.to_numeric(out[c], errors="coerce")
        mn = float(s.min()) if s.notna().any() else 0.0
        mx = float(s.max()) if s.notna().any() else 0.0

        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            out[c] = (s - mn) / (mx - mn)
        else:
            out[c] = 0.0

        stats[c] = {"min": mn, "max": mx}

    out.to_csv(output_csv, index=False)
    json_dump(stats_json, stats)

    logger.kv("excluded_from_normalization", exclude_cols)
    logger.kv("normalized_columns", features)
    logger.kv("stats_json", stats_json)
    logger.kv("output_csv", output_csv)

    return {
        "excluded_from_normalization": exclude_cols,
        "normalized_columns": features,
        "stats_json": stats_json,
        "output_csv": output_csv,
    }


# =============================================================================
# Step 5: Variance filter
# =============================================================================

def step5_variance_filter(
    input_csv: str,
    output_csv: str,
    report_json: str,
    exclude_cols: List[str],
    variance_threshold: float,
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 5 - Global variance filter")

    df = pd.read_csv(input_csv)
    features = numeric_feature_columns(df, exclude_cols)

    variances = {}
    removed = []
    for c in features:
        v = float(pd.to_numeric(df[c], errors="coerce").var(ddof=0))
        variances[c] = v
        if v < variance_threshold:
            removed.append(c)

    out = df.drop(columns=removed)
    out.to_csv(output_csv, index=False)

    kept = [c for c in features if c not in removed]
    report = {
        "excluded_from_variance": exclude_cols,
        "variance_threshold": variance_threshold,
        "variances": variances,
        "removed_columns": removed,
        "kept_feature_columns": kept,
        "output_csv": output_csv,
    }
    json_dump(report_json, report)

    logger.kv("excluded_from_variance", exclude_cols)
    logger.kv("variance_threshold", variance_threshold)
    logger.kv("removed_columns", removed)
    logger.kv("report_json", report_json)
    logger.kv("output_csv", output_csv)

    return report


# =============================================================================
# Step 6: PCA
# =============================================================================

def step6_pca(
    input_csv: str,
    output_csv: str,
    report_json: str,
    exclude_cols: List[str],
    retained_variance: float,
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 6 - PCA")

    df = pd.read_csv(input_csv)
    features = numeric_feature_columns(df, exclude_cols)

    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    pca = PCA(n_components=retained_variance, svd_solver="full")
    Z = pca.fit_transform(X)

    out = df[exclude_cols].copy() if exclude_cols else pd.DataFrame(index=df.index)
    pc_names = [f"PC{i+1}" for i in range(Z.shape[1])]

    for i, pc in enumerate(pc_names):
        out[pc] = Z[:, i]

    out.to_csv(output_csv, index=False)

    equations = {}
    for i, pc in enumerate(pc_names):
        equations[pc] = [
            {"feature": feat, "coefficient": float(coef)}
            for feat, coef in zip(features, pca.components_[i])
        ]

    report = {
        "excluded_from_pca": exclude_cols,
        "input_features": features,
        "retained_variance_target": retained_variance,
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "equations": equations,
        "output_csv": output_csv,
    }
    json_dump(report_json, report)

    logger.kv("excluded_from_pca", exclude_cols)
    logger.kv("input_feature_count", len(features))
    logger.kv("output_component_count", int(pca.n_components_))
    logger.kv("retained_variance_target", retained_variance)
    logger.kv("report_json", report_json)
    logger.kv("output_csv", output_csv)

    return report


# =============================================================================
# Step 7: Hilbert encode ZIP locations
# =============================================================================

def step7_hilbert_encode(
    input_csv: str,
    zip_shapefile: str,
    output_csv: str,
    mapping_json: str,
    zip_col: str,
    hilbert_order: int,
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 7 - Hilbert spatial encoding")

    df = pd.read_csv(input_csv)
    df[zip_col] = coerce_zip_series(df[zip_col])

    zcta = gpd.read_file(zip_shapefile)
    zcta_zip_col = detect_column(
        zcta.columns,
        ["ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "zip", "zcta"]
    )
    if zcta_zip_col is None:
        raise ValueError("Could not detect ZIP/ZCTA column in ZIP shapefile.")

    zcta = zcta.copy()
    zcta["zip_norm"] = coerce_zip_series(zcta[zcta_zip_col])

    needed = sorted(df[zip_col].dropna().unique().tolist())
    zcta = zcta[zcta["zip_norm"].isin(needed)].copy()

    if zcta.empty:
        raise ValueError("No ZIPs from PCA output were found in the shapefile.")

    zcta = zcta.to_crs("EPSG:3857")
    cent = zcta.geometry.centroid
    xs = cent.x.to_numpy()
    ys = cent.y.to_numpy()

    n = 1 << hilbert_order
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    rows = []
    for zip_code, x, y in zip(zcta["zip_norm"], xs, ys):
        gx = scale_to_grid(x, x_min, x_max, n)
        gy = scale_to_grid(y, y_min, y_max, n)
        h = xy2hilbert(gx, gy, hilbert_order)
        rows.append({
            "zip": zip_code,
            "centroid_x_m": float(x),
            "centroid_y_m": float(y),
            "grid_x": int(gx),
            "grid_y": int(gy),
            "hilbert_index": int(h),
        })

    mapping = pd.DataFrame(rows).sort_values("hilbert_index").reset_index(drop=True)
    mapping["hilbert_position"] = range(len(mapping))

    out = df.merge(mapping, on="zip", how="left")
    out.to_csv(output_csv, index=False)

    mapping_records = mapping.to_dict(orient="records")
    json_dump(mapping_json, mapping_records)

    report = {
        "hilbert_order": hilbert_order,
        "grid_size_per_axis": n,
        "zip_to_hilbert_mapping_count": int(len(mapping)),
        "mapping_json": mapping_json,
        "output_csv": output_csv,
    }

    logger.kv("hilbert_order", hilbert_order)
    logger.kv("grid_size_per_axis", n)
    logger.kv("mapping_json", mapping_json)
    logger.kv("output_csv", output_csv)

    return report


# =============================================================================
# Step 8: Create tensor [time, features, hilbert]
# =============================================================================

def step8_make_tensor(
    input_csv: str,
    tensor_npy: str,
    tensor_meta_json: str,
    time_col: str,
    hilbert_col: str,
    exclude_cols: List[str],
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 8 - Tensor creation")

    df = pd.read_csv(input_csv)

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    df = df.sort_values([time_col, hilbert_col]).reset_index(drop=True)

    times = sorted(df[time_col].dropna().unique().tolist())
    hilberts = sorted(df[hilbert_col].dropna().astype(int).unique().tolist())

    time_to_idx = {t: i for i, t in enumerate(times)}
    h_to_idx = {h: i for i, h in enumerate(hilberts)}

    tensor = np.full((len(times), len(feature_cols), len(hilberts)), np.nan, dtype=np.float32)

    for _, row in df.iterrows():
        t = row[time_col]
        h = row[hilbert_col]
        if pd.isna(t) or pd.isna(h):
            continue

        ti = time_to_idx[t]
        hi = h_to_idx[int(h)]

        for fi, feat in enumerate(feature_cols):
            val = pd.to_numeric(pd.Series([row[feat]]), errors="coerce").iloc[0]
            if pd.notna(val):
                tensor[ti, fi, hi] = np.float32(val)

    np.save(tensor_npy, tensor)

    meta = {
        "shape": [int(x) for x in tensor.shape],
        "dimensions": ["time", "feature", "hilbert_position"],
        "time_values": times,
        "feature_columns": feature_cols,
        "hilbert_positions": hilberts,
        "tensor_npy": tensor_npy,
    }
    json_dump(tensor_meta_json, meta)

    logger.kv("shape", meta["shape"])
    logger.kv("feature_columns", feature_cols)
    logger.kv("tensor_meta_json", tensor_meta_json)
    logger.kv("tensor_npy", tensor_npy)

    return meta


# =============================================================================
# Step 9: Split tensor by time
# =============================================================================

def step9_split_tensor(
    input_tensor_npy: str,
    tensor_meta_json: str,
    train_tensor_npy: str,
    val_tensor_npy: str,
    test_tensor_npy: str,
    split_meta_json: str,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    logger: PipelineLogger
) -> Dict:
    logger.section("Step 9 - Split tensor into train/validation/test by timestep")

    tensor = np.load(input_tensor_npy)
    meta = json.loads(Path(tensor_meta_json).read_text(encoding="utf-8"))

    n_time = tensor.shape[0]
    if n_time < 3:
        raise ValueError("Need at least 3 timesteps to create train/val/test splits.")

    n_train = int(math.floor(n_time * train_fraction))
    n_val = int(math.floor(n_time * val_fraction))
    n_test = n_time - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Split sizes invalid: n_time={n_time}, train={n_train}, val={n_val}, test={n_test}"
        )

    train_tensor = tensor[:n_train, :, :]
    val_tensor = tensor[n_train:n_train + n_val, :, :]
    test_tensor = tensor[n_train + n_val:, :, :]

    np.save(train_tensor_npy, train_tensor)
    np.save(val_tensor_npy, val_tensor)
    np.save(test_tensor_npy, test_tensor)

    time_values = meta["time_values"]
    split_meta = {
        "full_tensor_shape": [int(x) for x in tensor.shape],
        "feature_columns": meta["feature_columns"],
        "hilbert_positions": meta["hilbert_positions"],
        "fractions": {
            "train": train_fraction,
            "val": val_fraction,
            "test": test_fraction,
        },
        "split_sizes": {
            "train_timesteps": int(train_tensor.shape[0]),
            "val_timesteps": int(val_tensor.shape[0]),
            "test_timesteps": int(test_tensor.shape[0]),
        },
        "time_ranges": {
            "train": time_values[:n_train],
            "val": time_values[n_train:n_train + n_val],
            "test": time_values[n_train + n_val:],
        },
        "output_tensors": {
            "train_tensor_npy": train_tensor_npy,
            "val_tensor_npy": val_tensor_npy,
            "test_tensor_npy": test_tensor_npy,
        }
    }
    json_dump(split_meta_json, split_meta)

    logger.kv("input_tensor_npy", input_tensor_npy)
    logger.kv("full_tensor_shape", split_meta["full_tensor_shape"])
    logger.kv("fractions", split_meta["fractions"])
    logger.kv("split_sizes", split_meta["split_sizes"])
    logger.kv("split_meta_json", split_meta_json)

    return split_meta


# =============================================================================
# Step 10: Fill NaNs with iterative 2D Lorenzo-style predictor
# =============================================================================

def lorenzo_fill_2d(arr2d: np.ndarray, max_iters: int = 50, tol: float = 1e-6) -> np.ndarray:
    out = arr2d.astype(np.float32).copy()

    if np.isnan(out).all():
        return np.zeros_like(out, dtype=np.float32)

    row_means = np.nanmean(out, axis=1)
    col_means = np.nanmean(out, axis=0)
    global_mean = np.nanmean(out)

    row_means = np.where(np.isnan(row_means), global_mean, row_means)
    col_means = np.where(np.isnan(col_means), global_mean, col_means)

    nan_pos = np.argwhere(np.isnan(out))

    for i, j in nan_pos:
        out[i, j] = np.float32((row_means[i] + col_means[j]) / 2.0)

    for _ in range(max_iters):
        max_change = 0.0
        for i, j in nan_pos:
            preds = []

            if i > 0 and j > 0:
                preds.append(out[i - 1, j] + out[i, j - 1] - out[i - 1, j - 1])

            if i > 0:
                preds.append(out[i - 1, j])
            if j > 0:
                preds.append(out[i, j - 1])
            if i + 1 < out.shape[0]:
                preds.append(out[i + 1, j])
            if j + 1 < out.shape[1]:
                preds.append(out[i, j + 1])

            if preds:
                new_val = np.float32(np.mean(preds))
                max_change = max(max_change, float(abs(new_val - out[i, j])))
                out[i, j] = new_val

        if max_change < tol:
            break

    return out


def step10_fill_single_tensor(
    input_tensor_npy: str,
    output_tensor_npy: str,
    report_json: str,
    tensor_name: str,
    logger: PipelineLogger
) -> Dict:
    logger.section(f"Step 10 - Lorenzo fill for {tensor_name} tensor")

    tensor = np.load(input_tensor_npy)
    out = tensor.copy()

    filled_counts = {}
    for feat_idx in range(tensor.shape[1]):
        slice2d = tensor[:, feat_idx, :]
        n_missing = int(np.isnan(slice2d).sum())
        filled_counts[feat_idx] = n_missing
        if n_missing > 0:
            out[:, feat_idx, :] = lorenzo_fill_2d(slice2d)

    np.save(output_tensor_npy, out)

    report = {
        "tensor_name": tensor_name,
        "method": "Per-feature iterative 2D Lorenzo-style prediction on [time, hilbert_position]",
        "input_shape": [int(x) for x in tensor.shape],
        "filled_missing_counts_by_feature_index": filled_counts,
        "input_tensor_npy": input_tensor_npy,
        "output_tensor_npy": output_tensor_npy,
    }
    json_dump(report_json, report)

    logger.kv("tensor_name", tensor_name)
    logger.kv("input_shape", report["input_shape"])
    logger.kv("filled_missing_counts_by_feature_index", filled_counts)
    logger.kv("report_json", report_json)
    logger.kv("output_tensor_npy", output_tensor_npy)

    return report

def step11_make_rnn_ready_tensor(
    input_tensor_npy: str,
    output_tensor_npy: str,
    metadata_json: str,
    tensor_name: str,
    batch_first: bool,
    logger: PipelineLogger
) -> Dict:
    logger.section(f"Step 11 - Convert {tensor_name} tensor to PyTorch RNN-ready shape")

    tensor = np.load(input_tensor_npy).astype(np.float32)   # shape [T, F, H]

    if tensor.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor [time, feature, hilbert], got shape {tensor.shape}"
        )

    T, F, H = tensor.shape

    # Move to [T, F*H]
    flat = tensor.reshape(T, F * H)

    # Add batch dimension
    if batch_first:
        rnn_tensor = flat[np.newaxis, :, :]   # [1, T, F*H]
    else:
        rnn_tensor = flat[:, np.newaxis, :]   # [T, 1, F*H]

    np.save(output_tensor_npy, rnn_tensor)

    meta = {
        "tensor_name": tensor_name,
        "input_shape": [int(T), int(F), int(H)],
        "output_shape": [int(x) for x in rnn_tensor.shape],
        "batch_first": batch_first,
        "interpretation": {
            "time_axis": 0,
            "feature_axis": 1,
            "hilbert_axis": 2,
            "input_size": int(F * H),
            "batch_size": 1,
        },
        "transformation": "Flattened [feature, hilbert] into one input vector per timestep"
    }
    json_dump(metadata_json, meta)

    logger.kv("tensor_name", tensor_name)
    logger.kv("input_shape", [T, F, H])
    logger.kv("output_shape", meta["output_shape"])
    logger.kv("batch_first", batch_first)
    logger.kv("metadata_json", metadata_json)
    logger.kv("output_tensor_npy", output_tensor_npy)

    return meta


# =============================================================================
# Final combined log
# =============================================================================

def write_master_log(
    master_log_path: str,
    summary: Dict,
    intermediate_paths: Dict[str, str]
) -> None:
    lines = []
    lines.append("=" * 100)
    lines.append("FULL PIPELINE SUMMARY")
    lines.append("=" * 100)
    lines.append(json.dumps(summary, indent=2, default=str))
    lines.append("\n" + "=" * 100)
    lines.append("INTERMEDIATE FILES")
    lines.append("=" * 100)
    lines.append(json.dumps(intermediate_paths, indent=2))
    Path(master_log_path).write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# New Step 11: Save as PyTorch .pt
# =============================================================================

def step11_save_as_pt(input_npy: str, output_pt: str, logger: PipelineLogger) -> dict:
    logger.section("New Step 11 - Save as PyTorch .pt file")
    
    # Load the numpy data
    tensor_np = np.load(input_npy)
    
    # Convert it to a PyTorch tensor
    tensor_pt = torch.from_numpy(tensor_np).float()
    
    # Save it as a .pt file
    torch.save(tensor_pt, output_pt)
    
    logger.kv("saved_pt_file", output_pt)
    return {"output_pt": output_pt}


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="End-to-end preprocessing pipeline for RNN input data.")

    ap.add_argument("--air-quality", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--tri-facilities", required=True)
    ap.add_argument("--tri-chemicals", required=True)
    ap.add_argument("--zip-shapefile", required=True)
    ap.add_argument("--roads-shapefile", required=True)
    ap.add_argument("--place-shapefile", default="", help="Accepted but not used by current pipeline.")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--time-col", default="time")
    ap.add_argument("--zip-col", default="zip")

    ap.add_argument("--exclude-normalization", default="time,zip")
    ap.add_argument("--exclude-variance", default="time,zip")
    ap.add_argument("--exclude-pca", default="time,zip")

    ap.add_argument("--variance-threshold", type=float, required=True)
    ap.add_argument("--pca-retained-variance", type=float, required=True)

    ap.add_argument("--hilbert-order", type=int, default=8)
    ap.add_argument("--road-radius-km", type=float, default=2.0)
    ap.add_argument("--facility-radius-km", type=float, default=10.0)

    ap.add_argument("--train-fraction", type=float, required=True)
    ap.add_argument("--val-fraction", type=float, required=True)
    ap.add_argument("--test-fraction", type=float, required=True)

    ap.add_argument(
        "--direction-columns",
        default="",
        help="Comma-separated directional columns to expand into sin/cos before normalization."
    )
    ap.add_argument(
        "--no-auto-detect-direction-columns",
        action="store_true",
        help="Disable automatic detection of direction columns."
    )
    ap.add_argument(
        "--keep-original-direction-columns",
        action="store_true",
        help="Keep original direction columns after creating sin/cos columns."
    )

    args = ap.parse_args()
    
    # Check the math for fractions!
    total_fraction = args.train_fraction + args.val_fraction + args.test_fraction
    if not np.isclose(total_fraction, 1.0, atol=1e-8):
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total_fraction}")
    if args.train_fraction <= 0 or args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("train/val/test fractions must all be > 0")

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "intermediate"))
    ensure_dir(os.path.join(args.output_dir, "metadata"))
    ensure_dir(os.path.join(args.output_dir, "logs"))

    # File tracking lists
    intermediate = {
        "step1_merged_csv": os.path.join(args.output_dir, "intermediate", "01_merged.csv"),
        "step2_spatial_csv": os.path.join(args.output_dir, "intermediate", "02_with_spatial_impact.csv"),
        "step_time_features_csv": os.path.join(args.output_dir, "intermediate", "new_time_features.csv"),
        "step3_direction_expanded_csv": os.path.join(args.output_dir, "intermediate", "03_direction_expanded.csv"),
        "step4_normalized_csv": os.path.join(args.output_dir, "intermediate", "04_normalized.csv"),
        "step5_variance_filtered_csv": os.path.join(args.output_dir, "intermediate", "05_variance_filtered.csv"),
        "step_latlon_csv": os.path.join(args.output_dir, "intermediate", "new_latlon_added.csv"),
        "step8_full_tensor_npy": os.path.join(args.output_dir, "intermediate", "08_full_tensor.npy"),
        "step9_train_tensor_npy": os.path.join(args.output_dir, "intermediate", "09_train_tensor.npy"),
        "step9_val_tensor_npy": os.path.join(args.output_dir, "intermediate", "09_val_tensor.npy"),
        "step9_test_tensor_npy": os.path.join(args.output_dir, "intermediate", "09_test_tensor.npy"),
        "train_filled_tensor_npy": os.path.join(args.output_dir, "train_tensor_filled.npy"),
        "val_filled_tensor_npy": os.path.join(args.output_dir, "val_tensor_filled.npy"),
        "test_filled_tensor_npy": os.path.join(args.output_dir, "test_tensor_filled.npy"),
        "train_pt": os.path.join(args.output_dir, "train_tensor.pt"),
        "val_pt": os.path.join(args.output_dir, "val_tensor.pt"),
        "test_pt": os.path.join(args.output_dir, "test_tensor.pt"),
    }

    meta_paths = {
        "spatial_impact_json": os.path.join(args.output_dir, "metadata", "02_spatial_impact.json"),
        "direction_expand_json": os.path.join(args.output_dir, "metadata", "03_direction_expand.json"),
        "normalization_stats_json": os.path.join(args.output_dir, "metadata", "04_normalization_stats.json"),
        "variance_report_json": os.path.join(args.output_dir, "metadata", "05_variance_report.json"),
        "latlon_mapping_json": os.path.join(args.output_dir, "metadata", "new_latlon_mapping.json"),
        "tensor_meta_json": os.path.join(args.output_dir, "metadata", "08_tensor_metadata.json"),
        "split_meta_json": os.path.join(args.output_dir, "metadata", "09_split_metadata.json"),
        "train_fill_report_json": os.path.join(args.output_dir, "metadata", "10_train_lorenzo_fill_report.json"),
        "val_fill_report_json": os.path.join(args.output_dir, "metadata", "10_val_lorenzo_fill_report.json"),
        "test_fill_report_json": os.path.join(args.output_dir, "metadata", "10_test_lorenzo_fill_report.json"),
    }

    step_log_path = os.path.join(args.output_dir, "logs", "pipeline_steps.log")
    logger = PipelineLogger(step_log_path)

    summary = {
        "inputs": {
            "air_quality": args.air_quality,
            "weather": args.weather,
            "tri_facilities": args.tri_facilities,
            "tri_chemicals": args.tri_chemicals,
            "zip_shapefile": args.zip_shapefile,
            "roads_shapefile": args.roads_shapefile,
        },
        "parameters": {
            "time_col": args.time_col,
            "zip_col": args.zip_col,
            "exclude_normalization": parse_csv_list(args.exclude_normalization),
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        }
    }

    # Run steps!
    s1 = step1_merge(args.air_quality, args.weather, intermediate["step1_merged_csv"], args.time_col, args.zip_col, logger)
    
    s2 = step2_spatial_impact(intermediate["step1_merged_csv"], args.tri_facilities, args.tri_chemicals, args.zip_shapefile, args.roads_shapefile, intermediate["step2_spatial_csv"], meta_paths["spatial_impact_json"], args.zip_col, logger, args.road_radius_km, args.facility_radius_km)
    
    s_time = step_add_time_features(intermediate["step2_spatial_csv"], intermediate["step_time_features_csv"], args.time_col, logger)
    
    s3 = step3_expand_direction_columns(intermediate["step_time_features_csv"], intermediate["step3_direction_expanded_csv"], meta_paths["direction_expand_json"], logger, parse_csv_list(args.direction_columns), not args.no_auto_detect_direction_columns, not args.keep_original_direction_columns)
    
    s4 = step4_normalize(intermediate["step3_direction_expanded_csv"], intermediate["step4_normalized_csv"], meta_paths["normalization_stats_json"], parse_csv_list(args.exclude_normalization), logger)
    
    s5 = step5_variance_filter(intermediate["step4_normalized_csv"], intermediate["step5_variance_filtered_csv"], meta_paths["variance_report_json"], parse_csv_list(args.exclude_variance), args.variance_threshold, logger)
    
    # Skips PCA and Hilbert. Adds Lat/Lon directly.
    s_latlon = step_add_lat_lon(intermediate["step5_variance_filtered_csv"], args.zip_shapefile, intermediate["step_latlon_csv"], meta_paths["latlon_mapping_json"], args.zip_col, logger)
    
    # Create tensor using zip_col as the spatial axis
    s8 = step8_make_tensor(intermediate["step_latlon_csv"], intermediate["step8_full_tensor_npy"], meta_paths["tensor_meta_json"], args.time_col, args.zip_col, [args.time_col, args.zip_col], logger)
    
    s9 = step9_split_tensor(intermediate["step8_full_tensor_npy"], meta_paths["tensor_meta_json"], intermediate["step9_train_tensor_npy"], intermediate["step9_val_tensor_npy"], intermediate["step9_test_tensor_npy"], meta_paths["split_meta_json"], args.train_fraction, args.val_fraction, args.test_fraction, logger)
    
    s10_train = step10_fill_single_tensor(intermediate["step9_train_tensor_npy"], intermediate["train_filled_tensor_npy"], meta_paths["train_fill_report_json"], "train", logger)
    s10_val = step10_fill_single_tensor(intermediate["step9_val_tensor_npy"], intermediate["val_filled_tensor_npy"], meta_paths["val_fill_report_json"], "validation", logger)
    s10_test = step10_fill_single_tensor(intermediate["step9_test_tensor_npy"], intermediate["test_filled_tensor_npy"], meta_paths["test_fill_report_json"], "test", logger)
    
    # Save as PyTorch .pt files
    s11_train = step11_save_as_pt(intermediate["train_filled_tensor_npy"], intermediate["train_pt"], logger)
    s11_val = step11_save_as_pt(intermediate["val_filled_tensor_npy"], intermediate["val_pt"], logger)
    s11_test = step11_save_as_pt(intermediate["test_filled_tensor_npy"], intermediate["test_pt"], logger)

    logger.write()

    summary["steps"] = {
        "step1_merge": s1,
        "step2_spatial_impact": s2,
        "step_add_time_features": s_time,
        "step3_expand_direction_columns": s3,
        "step4_normalize": s4,
        "step5_variance_filter": s5,
        "step_add_lat_lon": s_latlon,
        "step8_make_full_tensor": s8,
        "step9_split_tensor": s9,
        "step10_fill_train": s10_train,
        "step10_fill_val": s10_val,
        "step10_fill_test": s10_test,
        "step11_save_pt": {"train": s11_train, "val": s11_val, "test": s11_test}
    }
    summary["metadata_files"] = meta_paths
    summary["intermediate_files"] = intermediate

    master_log_path = os.path.join(args.output_dir, "pipeline_full.log")
    write_master_log(master_log_path, summary, intermediate)

    print("\nPipeline complete.")
    print(f"Train .pt file: {intermediate['train_pt']}")
    print(f"Validation .pt file: {intermediate['val_pt']}")
    print(f"Test .pt file: {intermediate['test_pt']}")
    print(f"Pipeline summary log: {master_log_path}")


if __name__ == "__main__":
    main()