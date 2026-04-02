#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def load_state(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "air_quality": {
            "files": [],
            "sort_keys": {}
        },
        "weather": {
            "files": [],
            "sort_keys": {}
        }
    }


def save_state(path: Path, state: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def classify_file(path: Path) -> Optional[str]:
    name = path.name.lower()

    if not path.is_file() or path.suffix.lower() != ".csv":
        return None

    if "_air_quality_" in name:
        return "air_quality"
    if "_weather_" in name:
        return "weather"

    return None


def extract_sort_key(filename: str) -> Tuple[int, int, str]:
    """
    Supports examples like:
      feb_4thweek_air_quality_hourly_20260312_210107.csv
      march_1stweek_weather_hourly_20260315_120000.csv
      march_10_air_quality_hourly_20260401_120609.csv

    Sorting rule:
      - month from first token
      - numeric part from second token only
      - filename as final tie-breaker
    """
    name = Path(filename).name.lower()

    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    month_token = parts[0]
    day_token = parts[1]

    if month_token not in MONTH_MAP:
        raise ValueError(f"Unknown month token '{month_token}' in filename: {filename}")

    month_num = MONTH_MAP[month_token]

    m = re.match(r"^(\d+)", day_token)
    if not m:
        raise ValueError(
            f"Could not extract numeric date from second token '{day_token}' in filename: {filename}"
        )

    day_num = int(m.group(1))

    return (month_num, day_num, name)


def find_files(input_dir: Path) -> Dict[str, List[Path]]:
    found = {
        "air_quality": [],
        "weather": []
    }

    for p in input_dir.iterdir():
        kind = classify_file(p)
        if kind:
            found[kind].append(p)

    found["air_quality"].sort(key=lambda p: extract_sort_key(p.name))
    found["weather"].sort(key=lambda p: extract_sort_key(p.name))

    return found


def read_header(csv_path: Path) -> List[str]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            raise ValueError(f"CSV is empty: {csv_path}")


def append_csvs(files: List[Path], output_csv: Path, expected_header: Optional[List[str]]) -> List[str]:
    write_header = not output_csv.exists()
    header_in_use = expected_header

    with open(output_csv, "a", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8", newline="") as fin:
                reader = csv.reader(fin)

                try:
                    header = next(reader)
                except StopIteration:
                    print(f"Warning: skipping empty CSV: {file_path.name}")
                    continue

                if header_in_use is None:
                    header_in_use = header
                elif header != header_in_use:
                    raise ValueError(
                        f"Header mismatch in {file_path.name}\n"
                        f"Expected: {header_in_use}\n"
                        f"Found:    {header}"
                    )

                if write_header:
                    writer.writerow(header_in_use)
                    write_header = False

                for row in reader:
                    writer.writerow(row)

    return header_in_use


def rebuild_master(all_files: List[Path], output_csv: Path) -> None:
    if output_csv.exists():
        output_csv.unlink()
    append_csvs(all_files, output_csv, expected_header=None)


def process_category(category: str, files: List[Path], output_csv: Path, state: Dict) -> None:
    print(f"\nProcessing {category}...")

    seen_files = set(state[category]["files"])
    sort_keys = state[category].get("sort_keys", {})

    resolved_files = [str(p.resolve()) for p in files]
    new_files = [p for p in files if str(p.resolve()) not in seen_files]

    if not files:
        print("  No matching files found.")
        return

    if not new_files:
        print("  No new files to append.")
        return

    new_files.sort(key=lambda p: extract_sort_key(p.name))

    existing_last_key = None
    if state[category]["files"]:
        existing_keys = [
            tuple(sort_keys[f]) for f in state[category]["files"] if f in sort_keys
        ]
        if existing_keys:
            existing_last_key = max(existing_keys)

    earliest_new_key = extract_sort_key(new_files[0].name)

    needs_rebuild = (
        not output_csv.exists()
        or existing_last_key is None
        or earliest_new_key < existing_last_key
    )

    if needs_rebuild:
        print("  Rebuilding master file to preserve chronological order...")
        all_files = sorted(
            [Path(f) for f in set(state[category]["files"]) | set(resolved_files)],
            key=lambda p: extract_sort_key(p.name)
        )
        rebuild_master(all_files, output_csv)

        state[category]["files"] = [str(p.resolve()) for p in all_files]
        state[category]["sort_keys"] = {
            str(p.resolve()): list(extract_sort_key(p.name))
            for p in all_files
        }

        print(f"  Rebuilt {output_csv.name} from {len(all_files)} files.")
    else:
        print(f"  Appending {len(new_files)} new file(s)...")

        expected_header = read_header(output_csv) if output_csv.exists() else None
        append_csvs(new_files, output_csv, expected_header)

        for p in new_files:
            rp = str(p.resolve())
            state[category]["files"].append(rp)
            state[category]["sort_keys"][rp] = list(extract_sort_key(p.name))

        print(f"  Appended to {output_csv.name}.")

    print("  Added:")
    for p in new_files:
        print(f"    - {p.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge air quality and weather CSV files into master CSVs while tracking already processed files."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing the source CSV files")
    parser.add_argument("--state-file", default="merge_state.json", help="JSON file used to track processed files")
    parser.add_argument("--air-master", default="air_quality_master.csv", help="Output merged air quality CSV")
    parser.add_argument("--weather-master", default="weather_master.csv", help="Output merged weather CSV")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    state_file = Path(args.state_file).resolve()
    air_master = Path(args.air_master).resolve()
    weather_master = Path(args.weather_master).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    state = load_state(state_file)
    found = find_files(input_dir)

    process_category("air_quality", found["air_quality"], air_master, state)
    process_category("weather", found["weather"], weather_master, state)

    save_state(state_file, state)

    print("\nDone.")
    print(f"State file:      {state_file}")
    print(f"Air master CSV:  {air_master}")
    print(f"Weather master:  {weather_master}")


if __name__ == "__main__":
    main()