import csv

def check_literal_empty_fields(csv_path: str) -> dict:
    """
    Detect literal empty CSV fields, such as ,, or a trailing comma.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        empty_count_by_column = {col: 0 for col in header}
        total_rows = 0

        for row in reader:
            total_rows += 1

            # Pad short rows if needed
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))

            for col, value in zip(header, row):
                if value == "":
                    empty_count_by_column[col] += 1

    columns_with_missing = {
        col: count
        for col, count in empty_count_by_column.items()
        if count > 0
    }

    return {
        "total_rows": total_rows,
        "has_missing_values": len(columns_with_missing) > 0,
        "missing_count_by_column": empty_count_by_column,
        "columns_with_missing": columns_with_missing,
    }

result = check_literal_empty_fields("../data/pipeline-output/all_features_all_data.csv")

print("Has missing values:", result["has_missing_values"])
print("Columns with missing:", result["columns_with_missing"])