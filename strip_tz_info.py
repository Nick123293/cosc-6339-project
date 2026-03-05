#!/usr/bin/env python3
import csv
import argparse
import sys
from typing import Optional


def strip_after_second(value: str) -> str:
    """
    Keep only 'YYYY-MM-DD HH:MM:SS' if present at the start of the string.
    If it doesn't match expected length, return the original value unchanged.
    Example:
      '2026-03-05 12:34:56-06:00' -> '2026-03-05 12:34:56'
      '2026-03-05 12:34:56Z'      -> '2026-03-05 12:34:56' (also works)
    """
    if value is None:
        return value
    s = value.strip()
    # Expected prefix length for 'YYYY-MM-DD HH:MM:SS' is 19 chars
    if len(s) >= 19 and s[4] == "-" and s[7] == "-" and s[10] == " " and s[13] == ":" and s[16] == ":":
        return s[:19]
    return value


def find_time_col_index(header, time_col: str) -> int:
    # Match exact or stripped header (helps with ' time ' issues)
    normalized = [h.strip() for h in header]
    time_col_norm = time_col.strip()
    if time_col_norm in normalized:
        return normalized.index(time_col_norm)

    print(f"Error: time column '{time_col}' not found.", file=sys.stderr)
    print("Available columns:", file=sys.stderr)
    for h in normalized:
        print(f"  {h}", file=sys.stderr)
    sys.exit(1)


def process_csv(input_path: str, output_path: str, time_col: str, delimiter: str = ",") -> None:
    with open(input_path, "r", newline="", encoding="utf-8") as f_in, \
         open(output_path, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.reader(f_in, delimiter=delimiter)
        writer = csv.writer(f_out, delimiter=delimiter)

        try:
            header = next(reader)
        except StopIteration:
            print("Error: CSV is empty.", file=sys.stderr)
            sys.exit(1)

        writer.writerow(header)
        time_idx = find_time_col_index(header, time_col)

        in_rows = 0
        changed = 0

        for row in reader:
            in_rows += 1

            # Pad short rows to header length to avoid IndexError
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))

            old = row[time_idx]
            new = strip_after_second(old)
            if new != old:
                row[time_idx] = new
                changed += 1

            writer.writerow(row)

    print(f"Done. Rows processed: {in_rows}. Time values modified: {changed}.")
    print(f"Output written to: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Stream through a CSV and strip timezone suffix from a time column, keeping YYYY-MM-DD HH:MM:SS."
    )
    ap.add_argument("input", help="Input CSV path")
    ap.add_argument("output", help="Output CSV path")
    ap.add_argument("--time-col", default="time", help="Name of the time column (default: time)")
    ap.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")

    args = ap.parse_args()
    process_csv(args.input, args.output, args.time_col, args.delimiter)


if __name__ == "__main__":
    main()