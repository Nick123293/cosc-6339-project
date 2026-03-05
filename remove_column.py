#!/usr/bin/env python3

import csv
import argparse
import sys


def remove_column(input_file, output_file, column_name):
    with open(input_file, newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)

        try:
            header = next(reader)
        except StopIteration:
            print("Error: CSV file is empty.", file=sys.stderr)
            sys.exit(1)

        if column_name not in header:
            print(f"Error: Column '{column_name}' not found.", file=sys.stderr)
            sys.exit(1)

        remove_index = header.index(column_name)

        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)

            # Write updated header
            new_header = [h for i, h in enumerate(header) if i != remove_index]
            writer.writerow(new_header)

            # Write updated rows
            for row in reader:
                # Pad short rows if malformed
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))

                new_row = [v for i, v in enumerate(row) if i != remove_index]
                writer.writerow(new_row)

    print(f"Column '{column_name}' removed.")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove a specific column from a CSV file by header name."
    )
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument("column", help="Column name to remove")

    args = parser.parse_args()

    remove_column(args.input, args.output, args.column)


if __name__ == "__main__":
    main()