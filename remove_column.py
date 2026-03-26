#!/usr/bin/env python3

import csv
import argparse
import sys


def remove_columns(input_file, output_file, column_names):
    with open(input_file, newline='', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)

        try:
            header = next(reader)
        except StopIteration:
            print("Error: CSV file is empty.", file=sys.stderr)
            sys.exit(1)

        missing = [col for col in column_names if col not in header]
        if missing:
            print(
                f"Error: The following column(s) were not found: {', '.join(missing)}",
                file=sys.stderr
            )
            sys.exit(1)

        remove_indices = {header.index(col) for col in column_names}

        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)

            # Write updated header
            new_header = [h for i, h in enumerate(header) if i not in remove_indices]
            writer.writerow(new_header)

            # Write updated rows
            for row in reader:
                # Pad short rows if malformed
                if len(row) < len(header):
                    row += [""] * (len(header) - len(row))

                new_row = [v for i, v in enumerate(row) if i not in remove_indices]
                writer.writerow(new_row)

    print(f"Removed column(s): {', '.join(column_names)}")
    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove one or more columns from a CSV file by header name."
    )
    parser.add_argument("input", help="Input CSV file")
    parser.add_argument("output", help="Output CSV file")
    parser.add_argument(
        "columns",
        nargs="+",
        help="One or more column names to remove"
    )

    args = parser.parse_args()

    remove_columns(args.input, args.output, args.columns)


if __name__ == "__main__":
    main()