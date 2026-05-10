#!/usr/bin/env python3

import argparse
import csv
import json


def normalize_zip(value):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 5:
        return None

    return digits[:5]


def detect_column(fieldnames, candidates):
    lowered = {name.lower(): name for name in fieldnames}

    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]

    for name in fieldnames:
        name_lower = name.lower()
        for cand in candidates:
            if cand.lower() in name_lower:
                return name

    return None


def build_zip_latlon_map(input_csv):
    zip_map = {}

    with open(input_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("Input CSV does not have a header row.")

        zip_col = detect_column(reader.fieldnames, ["zip", "zipcode", "zip_code", "postal_code"])
        lat_col = detect_column(reader.fieldnames, ["latitude", "lat"])
        lon_col = detect_column(reader.fieldnames, ["longitude", "lon", "lng", "long"])

        if zip_col is None:
            raise ValueError("Could not detect a ZIP column.")
        if lat_col is None:
            raise ValueError("Could not detect a latitude column.")
        if lon_col is None:
            raise ValueError("Could not detect a longitude column.")

        for row in reader:
            zip_code = normalize_zip(row.get(zip_col))
            if zip_code is None:
                continue

            try:
                lat = float(row.get(lat_col, ""))
                lon = float(row.get(lon_col, ""))
            except (TypeError, ValueError):
                continue

            if zip_code not in zip_map:
                zip_map[zip_code] = {
                    "latitude": lat,
                    "longitude": lon,
                }

    return zip_map


def write_python_dict(output_path, var_name, zip_map):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{var_name} = ")
        json.dump(zip_map, f, indent=4, sort_keys=True)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a ZIP -> {latitude, longitude} mapping and write it as a Python dictionary."
    )
    parser.add_argument("input_csv", help="Path to the input CSV")
    parser.add_argument(
        "--output",
        required=True,
        help="Output file, e.g. zip_latlon.py or zip_latlon.txt"
    )
    parser.add_argument(
        "--var-name",
        default="zip_latlon",
        help="Variable name to assign the dictionary to"
    )

    args = parser.parse_args()

    zip_map = build_zip_latlon_map(args.input_csv)
    write_python_dict(args.output, args.var_name, zip_map)

    print(f"Found {len(zip_map)} unique ZIP codes.")
    print(f"Wrote Python dictionary to: {args.output}")


if __name__ == "__main__":
    main()