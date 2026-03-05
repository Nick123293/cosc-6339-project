#!/usr/bin/env python3

import csv
import argparse
import sys


def is_empty(value):
    """
    Defines what 'empty' means.
    Modify if needed (e.g., treat 'NA', 'null' as empty).
    """
    return value is None or value.strip() == ""


def remove_empty_columns(input_file, output_file):
    # First pass: determine which columns are entirely empty
    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("Error: CSV file is empty.", file=sys.stderr)
        sys.exit(1)

    header = rows[0]
    data_rows = rows[1:]

    num_cols = len(header)

    # Track whether each column has at least one non-empty value
    keep_column = [False] * num_cols

    for row in data_rows:
        # Pad row if malformed (short row)
        if len(row) < num_cols:
            row += [""] * (num_cols - len(row))

        for i in range(num_cols):
            if not is_empty(row[i]):
                keep_column[i] = True

    # Always drop columns with no data (all empty)
    indices_to_keep = [i for i, keep in enumerate(keep_column) if keep]

    if not indices_to_keep:
        print("Warning: All columns are empty.", file=sys.stderr)

    # Second pass: write filtered CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)

        # Write filtered header
        writer.writerow([header[i] for i in indices_to_keep])

        # Write filtered rows
        for row in data_rows:
            if len(row) < num_cols:
                row += [""] * (num_cols - len(row))

            writer.writerow([row[i] for i in indices_to_keep])

    print(f"Removed {num_cols - len(indices_to_keep)} empty columns.")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove columns that are entirely empty from a CSV file."
    )
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")

    args = parser.parse_args()

    remove_empty_columns(args.input, args.output)


if __name__ == "__main__":
    main()