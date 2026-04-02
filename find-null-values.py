import pandas as pd


def check_missing_values(csv_path: str) -> dict:
    """
    Check whether a CSV has any missing values.

    A value is considered missing if pandas reads it as NaN,
    which includes empty cells by default.

    Returns a dictionary containing:
      - total_rows
      - columns_with_missing
      - missing_count_by_column
      - has_missing_values
    """
    df = pd.read_csv(csv_path)

    total_rows = len(df)
    missing_count_by_column = df.isna().sum().to_dict()

    columns_with_missing = {
        col: count
        for col, count in missing_count_by_column.items()
        if count > 0
    }

    return {
        "total_rows": total_rows,
        "columns_with_missing": columns_with_missing,
        "missing_count_by_column": missing_count_by_column,
        "has_missing_values": len(columns_with_missing) > 0,
    }

result = check_missing_values("../data/pipeline-output/all_features_all_data.csv")

print(f"Total rows: {result['total_rows']}")
print(f"Has missing values: {result['has_missing_values']}")

if result["has_missing_values"]:
    print("Columns with missing values:")
    for col, count in result["columns_with_missing"].items():
        print(f"  {col}: {count} missing")
else:
    print("No missing values found.")