#!/usr/bin/env python3
"""This file is used solely to remove irrelevant data from the .shp file for Zip polygon data you get from the US Census (see preprocessing.py for more info).
You pass the air-quality or weather csv as a command line argument, and this script removes all ZIP info which does not pertain to ZIPs which are in that CSV."""
import argparse
import pandas as pd
import geopandas as gpd


def normalize_zip(value):
    """
    Convert ZIP values to clean 5-character strings.

    Examples:
        77002      -> "77002"
        "77002"   -> "77002"
        7702       -> "07702"
    """
    if pd.isna(value):
        return None

    # Convert to string, remove decimal part if read as float, strip spaces
    value = str(value).strip().split(".")[0]

    # Pad to 5 digits if needed
    return value.zfill(5)


def filter_shapefile_by_dataset_zips(
    csv_path,
    shp_path,
    output_path,
    csv_zip_col="zip",
    shp_zip_col="ZCTA5CE20"
):
    # Read the dataset containing the ZIP codes we want to keep
    df = pd.read_csv(csv_path)

    if csv_zip_col not in df.columns:
        raise ValueError(
            f"CSV file does not contain column '{csv_zip_col}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract unique ZIP codes from the dataset
    dataset_zips = (
        df[csv_zip_col]
        .dropna()
        .apply(normalize_zip)
        .unique()
    )

    dataset_zips = set(dataset_zips)

    print(f"Found {len(dataset_zips)} unique ZIP codes in dataset.")

    # Read the shapefile
    gdf = gpd.read_file(shp_path)

    if shp_zip_col not in gdf.columns:
        raise ValueError(
            f"Shapefile does not contain column '{shp_zip_col}'. "
            f"Available columns: {list(gdf.columns)}"
        )

    # Normalize shapefile ZIP codes too
    gdf["_zip_norm"] = gdf[shp_zip_col].apply(normalize_zip)

    # Keep only ZIP shapes that are present in the dataset
    filtered_gdf = gdf[gdf["_zip_norm"].isin(dataset_zips)].copy()

    print(f"Original shapefile rows: {len(gdf)}")
    print(f"Filtered shapefile rows: {len(filtered_gdf)}")

    # Find ZIPs in the dataset that were not found in the shapefile
    shapefile_zips = set(gdf["_zip_norm"].dropna().unique())
    missing_zips = sorted(dataset_zips - shapefile_zips)

    if missing_zips:
        print(f"Warning: {len(missing_zips)} dataset ZIP codes were not found in the shapefile.")
        print("Missing ZIPs:")
        print(missing_zips)
    else:
        print("All dataset ZIP codes were found in the shapefile.")

    # Remove helper column before saving
    filtered_gdf = filtered_gdf.drop(columns=["_zip_norm"])

    # Save filtered shapefile
    filtered_gdf.to_file(output_path)

    print(f"Filtered shapefile written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter a Texas ZIP/ZCTA shapefile using ZIP codes from a CSV dataset."
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV file containing ZIP codes."
    )

    parser.add_argument(
        "--shp",
        required=True,
        help="Path to input shapefile, e.g. texas_zcta.shp."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output shapefile, e.g. houston_zcta_filtered.shp."
    )

    parser.add_argument(
        "--csv-zip-col",
        default="zip",
        help="ZIP code column in the CSV. Default: zip"
    )

    parser.add_argument(
        "--shp-zip-col",
        default="ZCTA5CE20",
        help="ZIP/ZCTA column in the shapefile. Default: ZCTA5CE20"
    )

    args = parser.parse_args()

    filter_shapefile_by_dataset_zips(
        csv_path=args.csv,
        shp_path=args.shp,
        output_path=args.output,
        csv_zip_col=args.csv_zip_col,
        shp_zip_col=args.shp_zip_col
    )


if __name__ == "__main__":
    main()