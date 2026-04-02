#!/usr/bin/env python3

import argparse
import sys
import pandas as pd


def detect_time_frequency(times: pd.Series) -> pd.Timedelta:
    """
    Infer the expected timestep spacing from sorted unique timestamps
    using the most common difference.
    """
    unique_times = times.dropna().sort_values().unique()

    if len(unique_times) < 2:
        raise ValueError("Not enough unique timestamps to infer timestep frequency.")

    diffs = pd.Series(unique_times[1:] - unique_times[:-1])
    freq = diffs.mode()

    if freq.empty:
        raise ValueError("Could not infer timestep frequency.")

    return freq.iloc[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify completeness of a CSV with 'time' and 'zip' columns. "
            "Checks that every ZIP has every timestep and every timestep has every ZIP."
        )
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output-missing-pairs-csv",
        default=None,
        help="Optional CSV to save all missing (zip, time) pairs"
    )
    parser.add_argument(
        "--output-missing-by-zip-csv",
        default=None,
        help="Optional CSV to save summary of missing timestep counts per ZIP"
    )
    parser.add_argument(
        "--output-missing-by-time-csv",
        default=None,
        help="Optional CSV to save summary of missing ZIP counts per timestep"
    )
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = {"time", "zip"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(
            f"Error: missing required column(s): {', '.join(sorted(missing_cols))}",
            file=sys.stderr
        )
        sys.exit(1)

    df = df.copy()

    # Parse time column
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    bad_time_rows = df["time"].isna().sum()
    if bad_time_rows > 0:
        print(f"Error: found {bad_time_rows} rows with invalid time values.", file=sys.stderr)
        sys.exit(1)

    # Normalize ZIPs as strings
    df["zip"] = df["zip"].astype(str).str.strip()

    # Drop exact duplicate zip-time rows if they exist
    duplicate_pairs = df.duplicated(subset=["zip", "time"]).sum()
    if duplicate_pairs > 0:
        print(
            f"Warning: found {duplicate_pairs} duplicate (zip, time) rows. "
            f"Using unique (zip, time) pairs for completeness check."
        )
        df = df.drop_duplicates(subset=["zip", "time"]).copy()

    unique_times = pd.Series(df["time"].sort_values().unique())
    unique_zips = pd.Series(sorted(df["zip"].unique()))

    if unique_times.empty:
        print("Error: no valid timestamps found.", file=sys.stderr)
        sys.exit(1)

    if unique_zips.empty:
        print("Error: no ZIP codes found.", file=sys.stderr)
        sys.exit(1)

    try:
        freq = detect_time_frequency(unique_times)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    expected_times = pd.date_range(
        start=unique_times.min(),
        end=unique_times.max(),
        freq=freq
    )

    expected_zips = sorted(unique_zips.tolist())

    # Build full expected grid of all (zip, time) pairs
    expected_index = pd.MultiIndex.from_product(
        [expected_zips, expected_times],
        names=["zip", "time"]
    )

    observed_index = pd.MultiIndex.from_frame(df[["zip", "time"]])

    missing_index = expected_index.difference(observed_index)

    missing_pairs_df = missing_index.to_frame(index=False)
    missing_pairs_df = missing_pairs_df.rename(columns={"time": "missing_time"})

    # Summaries from both perspectives
    if missing_pairs_df.empty:
        missing_by_zip_df = pd.DataFrame(columns=["zip", "missing_timestep_count"])
        missing_by_time_df = pd.DataFrame(columns=["time", "missing_zip_count"])
    else:
        missing_by_zip_df = (
            missing_pairs_df.groupby("zip")
            .size()
            .reset_index(name="missing_timestep_count")
            .sort_values(["missing_timestep_count", "zip"], ascending=[False, True])
        )

        missing_by_time_df = (
            missing_pairs_df.groupby("missing_time")
            .size()
            .reset_index(name="missing_zip_count")
            .rename(columns={"missing_time": "time"})
            .sort_values(["missing_zip_count", "time"], ascending=[False, True])
        )

    total_expected_pairs = len(expected_zips) * len(expected_times)
    total_observed_pairs = len(observed_index.unique())
    total_missing_pairs = len(missing_pairs_df)

    zips_with_missing = len(missing_by_zip_df)
    times_with_missing = len(missing_by_time_df)

    print(f"Detected timestep frequency: {freq}")
    print(f"Expected number of timesteps: {len(expected_times)}")
    print(f"Expected number of ZIP codes: {len(expected_zips)}")
    print(f"Expected total (zip, time) pairs: {total_expected_pairs}")
    print(f"Observed unique (zip, time) pairs: {total_observed_pairs}")
    print(f"Missing (zip, time) pairs: {total_missing_pairs}")
    print(f"ZIP codes missing at least one timestep: {zips_with_missing}")
    print(f"Timesteps missing at least one ZIP: {times_with_missing}")

    if missing_pairs_df.empty:
        print("\nComplete grid verified: every ZIP has every timestep, and every timestep has every ZIP.")
    else:
        print("\nSample missing (zip, time) pairs:")
        print(missing_pairs_df.head(20).to_string(index=False))

        print("\nTop ZIPs with missing timesteps:")
        print(missing_by_zip_df.head(20).to_string(index=False))

        print("\nTop timesteps with missing ZIPs:")
        print(missing_by_time_df.head(20).to_string(index=False))

    if args.output_missing_pairs_csv:
        missing_pairs_df.to_csv(args.output_missing_pairs_csv, index=False)
        print(f"\nSaved missing pair report to: {args.output_missing_pairs_csv}")

    if args.output_missing_by_zip_csv:
        missing_by_zip_df.to_csv(args.output_missing_by_zip_csv, index=False)
        print(f"Saved missing-by-ZIP summary to: {args.output_missing_by_zip_csv}")

    if args.output_missing_by_time_csv:
        missing_by_time_df.to_csv(args.output_missing_by_time_csv, index=False)
        print(f"Saved missing-by-time summary to: {args.output_missing_by_time_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())