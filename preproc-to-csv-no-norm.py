#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd


# =============================================================================
# Utility helpers
# =============================================================================
zip_code_to_lat_long_map = {
        77002 : (29.75635, -95.36538),
        77003 : (29.74961, -95.34521),
        77004 : (29.72477, -95.36498),
        77005 : (29.71824, -95.42413),
        77006 : (29.74086, -95.39126),
        77007 : (29.77242, -95.41052),
        77008 : (29.79962, -95.41046),
        77009 : (29.79176, -95.36760),
        77010 : (29.75314, -95.35747),
        77011 : (29.74336, -95.30439),
        77012 : (29.70974, -95.28260),
        77013 : (29.78747, -95.21996),
        77014 : (29.98213, -95.46244),
        77015 : (29.78669, -95.18601),
        77016 : (29.85719, -95.30002),
        77017 : (29.68472, -95.25424),
        77018 : (29.82655, -95.42663),
        77019 : (29.75290, -95.41066),
        77020 : (29.77553, -95.30136),
        77021 : (29.69757, -95.35047),
        77022 : (29.82863, -95.37664),
        77023 : (29.72466, -95.31802),
        77024 : (29.76715, -95.50563),
        77025 : (29.68894, -95.43402),
        77026 : (29.79951, -95.32671),
        77027 : (29.73995, -95.44335),
        77028 : (29.82713, -95.26845),
        77029 : (29.76330, -95.26177),
        77030 : (29.70597, -95.40253),
        77031 : (29.65221, -95.54623),
        77032 : (29.93835, -95.34247),
        77033 : (29.66757, -95.34163),
        77034 : (29.63464, -95.21764),
        77035 : (29.65269, -95.47688),
        77036 : (29.70041, -95.53820),
        77037 : (29.90046, -95.40978),
        77038 : (29.91565, -95.44023),
        77039 : (29.90221, -95.33851),
        77040 : (29.88052, -95.52274),
        77041 : (29.87938, -95.58588),
        77042 : (29.74046, -95.55901),
        77043 : (29.80865, -95.56048),
        77044 : (29.87541, -95.19709),
        77045 : (29.63319, -95.42477),
        77046 : (29.73386, -95.43165),
        77047 : (29.62741, -95.38158),
        77048 : (29.62974, -95.32549),
        77049 : (29.81149, -95.15944),
        77050 : (29.88912, -95.26770),
        77051 : (29.65846, -95.36788),
        77053 : (29.59165, -95.46011),
        77054 : (29.68268, -95.40384),
        77055 : (29.79688, -95.49566),
        77056 : (29.74178, -95.46843),
        77057 : (29.73815, -95.48502),
        77058 : (29.56167, -95.09252),
        77059 : (29.59887, -95.11537),
        77060 : (29.93526, -95.39834),
        77061 : (29.65566, -95.27889),
        77062 : (29.57642, -95.13786),
        77063 : (29.73726, -95.52314),
        77064 : (29.91453, -95.55649),
        77065 : (29.92915, -95.61607),
        77066 : (29.95952, -95.50134),
        77067 : (29.95349, -95.45072),
        77068 : (30.00302, -95.47963),
        77069 : (29.98619, -95.52312),
        77070 : (29.97831, -95.57229),
        77071 : (29.65257, -95.52063),
        77072 : (29.70052, -95.58442),
        77073 : (29.99616, -95.40651),
        77074 : (29.69144, -95.51573),
        77075 : (29.62059, -95.26134),
        77076 : (29.85739, -95.38343),
        77077 : (29.74721, -95.62213),
        77078 : (29.85337, -95.26289),
        77079 : (29.77594, -95.60134),
        77080 : (29.81544, -95.52211),
        77081 : (29.71406, -95.48079),
        77082 : (29.72843, -95.63738),
        77083 : (29.69422, -95.64367),
        77084 : (29.82749, -95.65992),
        77085 : (29.62442, -95.48803),
        77086 : (29.92852, -95.49576),
        77087 : (29.68441, -95.30458),
        77088 : (29.88062, -95.45274),
        77089 : (29.58792, -95.22088),
        77090 : (30.01713, -95.44653),
        77091 : (29.85360, -95.44288),
        77092 : (29.83169, -95.47378),
        77093 : (29.86102, -95.34198),
        77094 : (29.77058, -95.69305),
        77095 : (29.90132, -95.64813),
        77096 : (29.67464, -95.47965),
        77098 : (29.73542, -95.41791),
        77099 : (29.67879, -95.58541),
        77204 : (29.72090, -95.36778)
        }

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
# Step 1: Merge weather + air quality
# =============================================================================


def step1_merge(
    air_quality_csv: str,
    weather_csv: str,
    output_csv: str,
    time_col: str,
    zip_col: str,
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 1 - Merge air quality and weather CSVs")

    aq = pd.read_csv(air_quality_csv).copy()
    wx = pd.read_csv(weather_csv).copy()

    if time_col not in aq.columns:
        raise ValueError(f"Air quality CSV missing time column: {time_col}")
    if zip_col not in aq.columns:
        raise ValueError(f"Air quality CSV missing zip column: {zip_col}")
    if time_col not in wx.columns:
        raise ValueError(f"Weather CSV missing time column: {time_col}")
    if zip_col not in wx.columns:
        raise ValueError(f"Weather CSV missing zip column: {zip_col}")

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
        "rows_air_quality": int(len(aq)),
        "rows_weather": int(len(wx)),
        "rows_merged": int(len(merged)),
        "merged_columns": merged.columns.tolist(),
        "output_csv": output_csv,
    }


# =============================================================================
# Step 2: Spatial impact score
# =============================================================================


def _pick_tri_facility_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "facility_id": detect_column(df.columns, ["TRIFID", "TRIFD", "trifid", "trifd", "facility_id", "tri_facility_id"]),
        "lat": detect_column(df.columns, ["LATITUDE83", "LATITUDE", "latitude", "lat"]),
        "lon": detect_column(df.columns, ["LONGITUDE83", "LONGITUDE", "longitude", "lon"]),
        "facility_name": detect_column(df.columns, ["FACILITY NAME", "FAC_NAME", "facility_name", "facility", "name"]),
    }



def _pick_tri_chemical_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "facility_id": detect_column(df.columns, ["TRIFID", "TRIFD", "trifid", "trifd", "facility_id", "tri_facility_id"]),
        "chemical_name": detect_column(df.columns, ["CHEMICAL", "CHEMICAL NAME", "chemical_name", "chemical"]),
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
    zcta_zip_col = detect_column(zcta.columns, ["ZCTA5CE20", "ZCTA5CE10", "GEOID20", "GEOID10", "zip", "zcta"])
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

    facility_amount_col = detect_column(fac.columns, ["total_air_emissions_lbs", "TOTAL_AIR_EMISSIONS_LBS", "TOTAL RELEASE", "TOTAL_RELEASE"])

    fac_keep = [fac_cols["facility_id"], fac_cols["lat"], fac_cols["lon"]]
    if fac_cols["facility_name"] is not None:
        fac_keep.append(fac_cols["facility_name"])
    if facility_amount_col is not None:
        fac_keep.append(facility_amount_col)

    fac_use = fac[fac_keep].copy().rename(columns={
        fac_cols["facility_id"]: "facility_id",
        fac_cols["lat"]: "lat",
        fac_cols["lon"]: "lon",
        **({fac_cols["facility_name"]: "facility_name"} if fac_cols["facility_name"] is not None else {}),
        **({facility_amount_col: "facility_amount"} if facility_amount_col is not None else {}),
    })

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
    chem_unique_count = chem_use.groupby("facility_id", as_index=False)["chemical_name"].nunique().rename(columns={"chemical_name": "chemical_count"})

    fac_enriched = fac_use.merge(chem_total_amount, on="facility_id", how="left")
    fac_enriched = fac_enriched.merge(chem_unique_count, on="facility_id", how="left")

    if "facility_amount" in fac_enriched.columns:
        fac_enriched["amount"] = fac_enriched["facility_amount"]
        for alt_col in ["amount_x", "amount_y"]:
            if alt_col in fac_enriched.columns:
                fac_enriched["amount"] = fac_enriched["amount"].fillna(fac_enriched[alt_col])
    elif "amount" not in fac_enriched.columns:
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
            how="cross",
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

        facility_df = cross.groupby("zip_norm", as_index=False).agg(
            facility_raw=("pair_impact", "sum"),
            facility_count_nearby=("facility_id", "nunique"),
        )

    fac_max = max(float(facility_df["facility_raw"].max()), 1e-12)
    facility_df["facility_impact_score"] = facility_df["facility_raw"] / fac_max

    score_df = (
        road_df[["zip_norm", "road_distance_m", "road_impact_score"]]
        .merge(
            facility_df[["zip_norm", "facility_raw", "facility_impact_score", "facility_count_nearby"]],
            on="zip_norm",
            how="left",
        )
        .fillna({"facility_raw": 0.0, "facility_impact_score": 0.0, "facility_count_nearby": 0})
    )

    added_columns = [
        "road_distance_m",
        "road_impact_score",
        "facility_count_nearby",
        "facility_impact_score",
        "overall_spatial_impact_score",
    ]
    score_df["overall_spatial_impact_score"] = 0.4 * score_df["road_impact_score"] + 0.6 * score_df["facility_impact_score"]
    score_df["overall_spatial_impact_score"] = np.clip(score_df["overall_spatial_impact_score"], 0, 1)

    out = (
        df.merge(
            score_df[["zip_norm"] + added_columns],
            left_on=zip_col,
            right_on="zip_norm",
            how="left",
        )
        .drop(columns=["zip_norm"])
    )
    out.to_csv(output_csv, index=False)

    meta = {
        "added_columns": added_columns,
        "detected_columns": {
            "tri_facilities": fac_cols,
            "tri_chemicals": chem_cols,
            "zcta_zip_column": zcta_zip_col,
            "facility_amount_column": facility_amount_col,
        },
        "parameters": {
            "road_radius_km": road_radius_km,
            "facility_radius_km": facility_radius_km,
        },
        "output_csv": output_csv,
    }
    json_dump(metadata_json, meta)

    logger.kv("zip_codes_scored", int(score_df["zip_norm"].nunique()))
    logger.kv("roads_feature_count", int(len(roads)))
    logger.kv("added_columns", added_columns)
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


def step3_add_time_features(
    input_csv: str,
    output_csv: str,
    metadata_json: str,
    time_col: str,
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 3 - Add time-derived features")

    df = pd.read_csv(input_csv).copy()
    if time_col not in df.columns:
        raise ValueError(f"Time column not found: {time_col}")

    dt = pd.to_datetime(df[time_col], errors="coerce")
    added = []

    feature_map = {
        "year": dt.dt.year,
        "month": dt.dt.month,
        "month_sin": np.sin(2*np.pi*dt.dt.month/12),
        "month_cos": np.cos(2*np.pi*dt.dt.month/12),
        "day": dt.dt.day,
        # "day_sin": np.sin(2*np.pi*dt.dt.day),
        # "day_cos": np.cos(2*np.pi*dt.dt.day),
        "hour": dt.dt.hour,
        "hour_sin": np.sin(2*np.pi*dt.dt.hour/24),
        "hour_cos": np.cos(2*np.pi*dt.dt.hour/24),
        "day_of_week": dt.dt.dayofweek,
        "day_of_week_sin": np.sin(2*np.pi*dt.dt.day_of_week/7),
        "day_of_week_cos": np.cos(2*np.pi*dt.dt.day_of_week/7),
        "day_of_year": dt.dt.dayofyear,
        "is_weekend": dt.dt.dayofweek.isin([5, 6]).astype(int),
    }

    for col, values in feature_map.items():
        df[col] = values
        added.append(col)

    df.to_csv(output_csv, index=False)


    meta = {"added_time_columns": added, "output_csv": output_csv}
    json_dump(metadata_json, meta)


    logger.kv("added_time_columns", added)
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 4: Expand direction columns to sin/cos
# =============================================================================


def step4_expand_direction_columns(
    input_csv: str,
    output_csv: str,
    metadata_json: str,
    logger: PipelineLogger,
    direction_columns: Optional[List[str]] = None,
    auto_detect: bool = True,
    drop_original: bool = True,
) -> Dict:
    logger.section("Step 4 - Expand directional columns into sine/cosine")

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

    out = df.copy()
    expanded = []
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

    logger.kv("expanded_columns", expanded)
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 5: Add latitude/longitude by ZIP
# =============================================================================


def step5_add_lat_lon(
    input_csv: str,
    zip_shapefile: str,   # kept only for API compatibility; not used
    output_csv: str,
    metadata_json: str,
    zip_col: str,
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 5 - Add latitude/longitude from ZIP mapping")

    df = pd.read_csv(input_csv).copy()
    df[zip_col] = coerce_zip_series(df[zip_col])

    mapping = pd.DataFrame(
        [
            {
                zip_col: int(zip_code),
                "latitude": float(lat),
                "longitude": float(lon),
            }
            for zip_code, (lat, lon) in zip_code_to_lat_long_map.items()
        ]
    )

    mapping[zip_col] = coerce_zip_series(mapping[zip_col])

    out = df.merge(mapping, on=zip_col, how="left")
    out.to_csv(output_csv, index=False)

    zip_values = set(df[zip_col].dropna().unique().tolist())
    mapped_zip_values = set(mapping[zip_col].dropna().unique().tolist())
    missing_zips = sorted(zip_values - mapped_zip_values)

    meta = {
        "zip_count_mapped": int(len(mapped_zip_values & zip_values)),
        "added_columns": ["latitude", "longitude"],
        "mapping_source": "zip_code_to_lat_long_map",
        "missing_zip_count": int(len(missing_zips)),
        "missing_zips": missing_zips,
        "output_csv": output_csv,
    }
    json_dump(metadata_json, meta)

    logger.kv("zip_count_mapped", int(len(mapped_zip_values & zip_values)))
    logger.kv("missing_zip_count", int(len(missing_zips)))
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 6: Variance filter on time-varying features only
# =============================================================================


def step6_variance_filter_time_variant_only(
    input_csv: str,
    output_csv: str,
    report_json: str,
    exclude_cols: List[str],
    variance_threshold: float,
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 6 - Global variance filter on time-varying numeric features")

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
    logger.kv("output_csv", output_csv)
    logger.kv("report_json", report_json)

    return report


# =============================================================================
# Step 7: Split into time-varying and time-invariant CSVs
# =============================================================================


def step7_split_variant_invariant(
    input_csv: str,
    time_variant_csv: str,
    time_invariant_csv: str,
    metadata_json: str,
    time_col: str,
    zip_col: str,
    invariant_feature_cols: List[str],
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 7 - Split output into time-varying and time-invariant CSVs")

    df = pd.read_csv(input_csv).copy()

    missing = [c for c in [zip_col] + invariant_feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required invariant columns for split: {missing}")

    invariant_cols = [zip_col] + invariant_feature_cols
    time_variant_drop = [c for c in invariant_feature_cols if c in df.columns]
    time_variant_df = df.drop(columns=time_variant_drop).copy()

    if time_col not in time_variant_df.columns:
        raise ValueError(f"Time-varying output requires time column '{time_col}', but it was not found.")

    time_invariant_df = df[invariant_cols].drop_duplicates(subset=[zip_col]).sort_values(zip_col).reset_index(drop=True)

    non_invariant = {}
    for c in invariant_feature_cols:
        nunq = df.groupby(zip_col, dropna=False)[c].nunique(dropna=False)
        bad_zips = nunq[nunq > 1]
        if not bad_zips.empty:
            non_invariant[c] = int(len(bad_zips))

    time_variant_df.to_csv(time_variant_csv, index=False)
    time_invariant_df.to_csv(time_invariant_csv, index=False)

    meta = {
        "time_variant_csv": time_variant_csv,
        "time_invariant_csv": time_invariant_csv,
        "time_variant_columns": time_variant_df.columns.tolist(),
        "time_invariant_columns": time_invariant_df.columns.tolist(),
        "invariant_feature_columns": invariant_feature_cols,
        "row_counts": {
            "time_variant": int(len(time_variant_df)),
            "time_invariant": int(len(time_invariant_df)),
        },
        "non_invariant_columns_detected": non_invariant,
    }
    json_dump(metadata_json, meta)

    logger.kv("time_variant_columns", time_variant_df.columns.tolist())
    logger.kv("time_invariant_columns", time_invariant_df.columns.tolist())
    logger.kv("row_counts", meta["row_counts"])
    logger.kv("non_invariant_columns_detected", non_invariant)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Step 8: Add past-timestep features to time-varying CSV
# =============================================================================


def step8_add_past_features_to_time_variant(
    input_csv: str,
    output_csv: str,
    metadata_json: str,
    time_col: str,
    zip_col: str,
    feature_cols: List[str],
    num_past_feats: int,
    logger: PipelineLogger,
) -> Dict:
    logger.section("Step 8 - Add past-timestep features to time-varying CSV")

    df = pd.read_csv(input_csv).copy()

    if num_past_feats <= 0 or not feature_cols:
        df.to_csv(output_csv, index=False)
        meta = {
            "skipped": True,
            "reason": "No lag features requested.",
            "requested_feature_columns": feature_cols,
            "num_past_feats": num_past_feats,
            "output_csv": output_csv,
        }
        json_dump(metadata_json, meta)
        logger.kv("skipped", True)
        logger.kv("reason", meta["reason"])
        logger.kv("output_csv", output_csv)
        logger.kv("metadata_json", metadata_json)
        return meta

    missing_required = [c for c in [time_col, zip_col] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for past features: {missing_required}")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Requested past-feature source columns not found: {missing_features}")

    work = df.copy()
    work[zip_col] = coerce_zip_series(work[zip_col])
    work["__orig_order__"] = np.arange(len(work), dtype=np.int64)
    work["__time_dt__"] = pd.to_datetime(work[time_col], errors="coerce")

    sort_cols = [zip_col, "__time_dt__", "__orig_order__"]
    work = work.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    added_columns: List[str] = []
    for feat in feature_cols:
        numeric_series = pd.to_numeric(work[feat], errors="coerce")
        grouped = numeric_series.groupby(work[zip_col], dropna=False)
        for lag in range(1, num_past_feats + 1):
            new_col = f"{feat}_past_{lag}"
            work[new_col] = grouped.shift(lag).fillna('nan')
            added_columns.append(new_col)

    work = work.sort_values("__orig_order__", kind="stable").drop(columns=["__orig_order__", "__time_dt__"])
    work.to_csv(output_csv, index=False)

    meta = {
        "skipped": False,
        "requested_feature_columns": feature_cols,
        "num_past_feats": num_past_feats,
        "added_columns": added_columns,
        "num_added_columns": len(added_columns),
        "output_csv": output_csv,
    }
    json_dump(metadata_json, meta)

    logger.kv("requested_feature_columns", feature_cols)
    logger.kv("num_past_feats", num_past_feats)
    logger.kv("num_added_columns", len(added_columns))
    logger.kv("added_columns", added_columns)
    logger.kv("output_csv", output_csv)
    logger.kv("metadata_json", metadata_json)

    return meta


# =============================================================================
# Final combined log
# =============================================================================


def write_master_log(master_log_path: str, summary: Dict, intermediate_paths: Dict[str, str]) -> None:
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
# Main
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Preprocessing pipeline that merges AQ/weather data, adds spatial features, and outputs split CSVs."
    )

    ap.add_argument("--air-quality", required=True)
    ap.add_argument("--weather", required=True)
    ap.add_argument("--tri-facilities", required=True)
    ap.add_argument("--tri-chemicals", required=True)
    ap.add_argument("--zip-shapefile", required=True)
    ap.add_argument("--roads-shapefile", required=True)
    ap.add_argument("--place-shapefile", default="", help="Accepted but not used by the current pipeline.")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--time-col", default="time")
    ap.add_argument("--zip-col", default="zip")

    ap.add_argument(
        "--exclude-variance",
        default="time,zip,latitude,longitude,road_distance_m,road_impact_score,facility_count_nearby,facility_impact_score,overall_spatial_impact_score",
        help="Comma-separated columns to exclude from the variance filter. Defaults exclude time-invariant fields.",
    )
    ap.add_argument("--variance-threshold", type=float, default=None, help="Optional. If omitted, the variance filter is skipped.")

    ap.add_argument("--road-radius-km", type=float, default=2.0)
    ap.add_argument("--facility-radius-km", type=float, default=10.0)

    ap.add_argument(
        "--direction-columns",
        default="",
        help="Comma-separated directional columns to expand into sin/cos.",
    )
    ap.add_argument(
        "--no-auto-detect-direction-columns",
        action="store_true",
        help="Disable automatic detection of direction columns.",
    )
    ap.add_argument(
        "--keep-original-direction-columns",
        action="store_true",
        help="Keep original direction columns after creating sin/cos columns.",
    )
    ap.add_argument(
        "--feats-for-past",
        nargs="*",
        default=[],
        help=(
            "Space-separated list of time-varying feature columns for which lagged past-step "
            "columns should be added to the final time-varying CSV. Example: --feats-for-past AQI CM2_5"
        ),
    )
    ap.add_argument(
        "--num-past-feats",
        type=int,
        default=0,
        help="Number of previous timesteps to add for each feature in --feats-for-past.",
    )
    ap.add_argument(
        "--output-mode",
        choices=["split", "combined"],
        default="split",
        help=(
            "Choose whether to output separate time-variant/time-invariant CSVs "
            "('split') or keep everything in a single CSV ('combined')."
        ),
    )

    args = ap.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "intermediate"))
    ensure_dir(os.path.join(args.output_dir, "metadata"))
    ensure_dir(os.path.join(args.output_dir, "logs"))

    intermediate = {
        "step1_merged_csv": os.path.join(args.output_dir, "intermediate", "01_merged.csv"),
        "step2_spatial_csv": os.path.join(args.output_dir, "intermediate", "02_with_spatial_impact.csv"),
        "step3_time_features_csv": os.path.join(args.output_dir, "intermediate", "03_with_time_features.csv"),
        "step4_direction_expanded_csv": os.path.join(args.output_dir, "intermediate", "04_direction_expanded.csv"),
        "step5_with_latlon_csv": os.path.join(args.output_dir, "intermediate", "05_with_latlon.csv"),
        "step6_variance_filtered_csv": os.path.join(args.output_dir, "intermediate", "06_variance_filtered.csv"),
        "step7_time_variant_csv": os.path.join(args.output_dir, "intermediate", "07_time_variant_base.csv"),
        "step7_time_invariant_csv": os.path.join(args.output_dir, "time_invariant_features.csv"),
        "final_time_variant_csv": os.path.join(args.output_dir, "time_variant_features.csv"),
        "final_time_invariant_csv": os.path.join(args.output_dir, "time_invariant_features.csv"),
        "final_combined_csv": os.path.join(args.output_dir, "all_features.csv"),
    }

    meta_paths = {
        "spatial_impact_json": os.path.join(args.output_dir, "metadata", "02_spatial_impact.json"),
        "time_features_json": os.path.join(args.output_dir, "metadata", "03_time_features.json"),
        "direction_expand_json": os.path.join(args.output_dir, "metadata", "04_direction_expand.json"),
        "latlon_mapping_json": os.path.join(args.output_dir, "metadata", "05_latlon_mapping.json"),
        "variance_report_json": os.path.join(args.output_dir, "metadata", "06_variance_report.json"),
        "split_report_json": os.path.join(args.output_dir, "metadata", "07_split_report.json"),
        "past_features_json": os.path.join(args.output_dir, "metadata", "08_past_features.json"),
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
            "variance_threshold": args.variance_threshold,
            "exclude_variance": parse_csv_list(args.exclude_variance),
            "road_radius_km": args.road_radius_km,
            "facility_radius_km": args.facility_radius_km,
            "feats_for_past": args.feats_for_past,
            "num_past_feats": args.num_past_feats,
            "output_mode": args.output_mode,
        },
    }

    s1 = step1_merge(
        args.air_quality,
        args.weather,
        intermediate["step1_merged_csv"],
        args.time_col,
        args.zip_col,
        logger,
    )

    s2 = step2_spatial_impact(
        intermediate["step1_merged_csv"],
        args.tri_facilities,
        args.tri_chemicals,
        args.zip_shapefile,
        args.roads_shapefile,
        intermediate["step2_spatial_csv"],
        meta_paths["spatial_impact_json"],
        args.zip_col,
        logger,
        args.road_radius_km,
        args.facility_radius_km,
    )

    s3 = step3_add_time_features(
    intermediate["step2_spatial_csv"],
    intermediate["step3_time_features_csv"],
    meta_paths["time_features_json"],
    args.time_col,
    logger,
    )

    s4 = step4_expand_direction_columns(
        intermediate["step3_time_features_csv"],
        intermediate["step4_direction_expanded_csv"],
        meta_paths["direction_expand_json"],
        logger,
        parse_csv_list(args.direction_columns),
        not args.no_auto_detect_direction_columns,
        not args.keep_original_direction_columns,
    )

    s5 = step5_add_lat_lon(
        intermediate["step4_direction_expanded_csv"],
        args.zip_shapefile,
        intermediate["step5_with_latlon_csv"],
        meta_paths["latlon_mapping_json"],
        args.zip_col,
        logger,
    )

    current_csv = intermediate["step5_with_latlon_csv"]
    s6 = {"skipped": True, "reason": "variance_threshold not provided"}
    if args.variance_threshold is not None:
        s6 = step6_variance_filter_time_variant_only(
            current_csv,
            intermediate["step6_variance_filtered_csv"],
            meta_paths["variance_report_json"],
            parse_csv_list(args.exclude_variance),
            args.variance_threshold,
            logger,
        )
        current_csv = intermediate["step6_variance_filtered_csv"]

    invariant_feature_cols = [
        "latitude",
        "longitude",
        "road_distance_m",
        "road_impact_score",
        "facility_count_nearby",
        "facility_impact_score",
        "overall_spatial_impact_score",
    ]

    if args.output_mode == "split":
        s7 = step7_split_variant_invariant(
            current_csv,
            intermediate["step7_time_variant_csv"],
            intermediate["step7_time_invariant_csv"],
            meta_paths["split_report_json"],
            args.time_col,
            args.zip_col,
            invariant_feature_cols,
            logger,
        )

        s8 = step8_add_past_features_to_time_variant(
            intermediate["step7_time_variant_csv"],
            intermediate["final_time_variant_csv"],
            meta_paths["past_features_json"],
            args.time_col,
            args.zip_col,
            args.feats_for_past,
            args.num_past_feats,
            logger,
        )

        final_outputs = {
            "mode": "split",
            "time_variant_csv": intermediate["final_time_variant_csv"],
            "time_invariant_csv": intermediate["final_time_invariant_csv"],
        }

    else:
        s7 = {
            "skipped": True,
            "reason": "output_mode='combined'; split step not run.",
        }

        s8 = step8_add_past_features_to_time_variant(
            current_csv,
            intermediate["final_combined_csv"],
            meta_paths["past_features_json"],
            args.time_col,
            args.zip_col,
            args.feats_for_past,
            args.num_past_feats,
            logger,
        )

        final_outputs = {
            "mode": "combined",
            "combined_csv": intermediate["final_combined_csv"],
        }

    logger.write()

    summary["steps"] = {
        "step1_merge": s1,
        "step2_spatial_impact": s2,
        "step3_add_time_features": s3,
        "step4_expand_direction_columns": s4,
        "step5_add_lat_lon": s5,
        "step6_variance_filter": s6,
        "step7_split_variant_invariant": s7,
        "step8_add_past_features": s8,
    }
    summary["metadata_files"] = meta_paths
    summary["intermediate_files"] = intermediate
    summary["final_outputs"] = final_outputs

    master_log_path = os.path.join(args.output_dir, "pipeline_full.log")
    write_master_log(master_log_path, summary, intermediate)

    print("\nPipeline complete.")
    if args.output_mode == "split":
        print(f"Time-varying CSV: {intermediate['final_time_variant_csv']}")
        print(f"Time-invariant CSV: {intermediate['final_time_invariant_csv']}")
    else:
        print(f"Combined CSV: {intermediate['final_combined_csv']}")
    print(f"Pipeline summary log: {master_log_path}")


if __name__ == "__main__":
    main()
