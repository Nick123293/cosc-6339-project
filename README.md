# cosc-6339-project

# NOTE
Before you use run_pipeline.sh, it is recommended to use strip_tz_info.py, which gets rid of the timezone information from the CSV (this is redundant information as they are all (-6:00).

To run this strip_tz_info.py do:

```bash python3 strip_tz_info.py input_csv output_csv```

# Hilbert Tensor Preprocessing Pipeline

This pipeline converts a large Texas air-quality CSV into PyTorch tensors suitable for spatiotemporal modeling with RNNs, ConvLSTMs, and related architectures.

The pipeline performs these stages in order:

1. **Filter rows near Houston** using a configurable radius in latitude/longitude space.
2. **Normalize numeric columns** independently to the range `[0, 1]`.
3. **Apply a dense Hilbert spatial layout** and sort the CSV by `time`, then `x`, then `y`.
4. **Split by timestamp** into training / validation / testing tensors with shape `[T, C, H, W]`.

All intermediate CSV files are preserved so you can inspect each stage.

---

## Files in the pipeline

- `01_filter_houston_radius.py`
- `02_normalize_csv.py`
- `03_hilbert_sort_csv.py`
- `04_split_to_tensors.py`
- `run_pipeline.sh`

The recommended entry point is **`run_pipeline.sh`**, which executes all four Python scripts in sequence.

---

## Expected input format

The input CSV should contain at least these columns:

- `time`
- `latitude`
- `longitude`
- one or more numeric feature columns

Example feature columns might include:

- `us_aqi`
- `pm10`
- `pm2_5`
- `carbon_monoxide`
- `nitrogen_dioxide`
- `sulphur_dioxide`
- `ozone`
- `uv_index_clear_sky`
- `uv_index`
- `dust`
- `aerosol_optical_depth`

Any non-feature columns such as `city`, `state`, or `zip` are ignored unless you explicitly adapt the scripts.

---

## Output artifacts

The pipeline writes:

- a **filtered CSV** containing only rows near Houston
- a **normalized CSV** where each selected numeric column is scaled to `[0,1]`
- a **normalization stats CSV** containing the min/max used for each normalized column
- a **Hilbert-sorted CSV** containing spatial layout information
- a **layout CSV** mapping each unique location to its Hilbert-packed grid position
- three PyTorch tensor files:
  - training tensor
  - validation tensor
  - testing tensor

The tensor shape is:

- **`[T, C, H, W]`**

where:

- `T` = number of timesteps in that split
- `C` = number of selected feature columns
- `H, W` = dense Hilbert-packed spatial dimensions

---

## How the split works

The split is performed over **sorted unique timestamps**, not random rows.

For example, if you choose:

- `train_pct = 0.70`
- `val_pct = 0.15`
- `test_pct = 0.15`

then:

- the earliest 70% of timestamps go to training
- the next 15% go to validation
- the final 15% go to testing

This avoids temporal leakage and is appropriate for forecasting future timesteps.

---

## How the Houston radius filter works

The first stage keeps only rows satisfying:

```text
(latitude - houston_lat)^2 + (longitude - houston_lon)^2 <= radius_squared
```

This is a squared Euclidean distance in latitude/longitude coordinates.

Default Houston center:

- `houston_lat = 29.7604`
- `houston_lon = -95.3698`

You can override these if needed.

---

## Normalization behavior

The normalization stage scales each selected numeric column independently:

```text
normalized_value = (value - min(column)) / (max(column) - min(column))
```

If a column has the same value in every row (`min == max`), the normalized output for that column is set to `0`.

The `time` column is not normalized.

---

## Hilbert encoding behavior

The Hilbert stage:

1. extracts unique spatial points from normalized `latitude` and `longitude`
2. computes a Hilbert order over those points
3. packs them densely into a 2D spatial matrix
4. annotates each row with:
   - `location_id`
   - `hilbert_index`
   - `x`
   - `y`
5. sorts the CSV by:
   - increasing `time`
   - increasing `x`
   - increasing `y`

This preserves spatial locality while keeping the spatial matrix dense.

---

## Usage

Run the full pipeline with:

```bash
bash run_pipeline.sh [arguments]
```

### Required / commonly used arguments

#### Input / output paths

- `--input`
  - Path to the original input CSV.

- `--filtered-csv`
  - Path for the filtered Houston-area CSV from stage 1.

- `--normalized-csv`
  - Path for the normalized CSV from stage 2.

- `--normalization-stats-csv`
  - Path for the CSV containing per-column min/max values used during normalization.

- `--hilbert-csv`
  - Path for the Hilbert-annotated and sorted CSV from stage 3.

- `--layout-csv`
  - Path for the CSV mapping each unique normalized location to `(x, y)` in the dense Hilbert layout.

- `--train-out`
  - Path for the training tensor `.pt` file.

- `--val-out`
  - Path for the validation tensor `.pt` file.

- `--test-out`
  - Path for the testing tensor `.pt` file.

#### Feature selection

- `--features`
  - Comma-separated list of feature columns to include in normalization and the final tensors.
  - Example:
    `us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone`

#### Houston-area filter parameters

- `--radius-squared`
  - Keeps rows satisfying:
    `(lat-houston_lat)^2 + (lon-houston_lon)^2 <= radius_squared`

- `--houston-lat`
  - Latitude of the Houston center point.
  - Default: `29.7604`

- `--houston-lon`
  - Longitude of the Houston center point.
  - Default: `-95.3698`

#### Numeric precision

- `--dtype`
  - Floating-point precision for normalization and tensor output.
  - Typical values:
    - `float32`
    - `float64`

#### Split percentages

- `--train-pct`
  - Fraction of sorted timestamps assigned to training.

- `--val-pct`
  - Fraction of sorted timestamps assigned to validation.

- `--test-pct`
  - Fraction of sorted timestamps assigned to testing.

These should sum to `1.0`.

#### Performance / processing options

- `--chunksize`
  - Number of CSV rows to process at a time in chunked stages.
  - Larger values can be faster if you have enough RAM.

- `--workers`
  - Number of worker threads/processes used where supported by the pipeline.
  - For most systems, `3` or `4` is a reasonable starting point.

---

## Full sample command using all arguments

```bash
bash run_pipeline.sh \
  --input /path/to/input.csv \
  --filtered-csv /path/to/filtered_houston.csv \
  --normalized-csv /path/to/normalized_houston.csv \
  --normalization-stats-csv /path/to/normalization_stats.csv \
  --hilbert-csv /path/to/hilbert_sorted.csv \
  --layout-csv /path/to/hilbert_layout.csv \
  --train-out /path/to/train_tensor.pt \
  --val-out /path/to/val_tensor.pt \
  --test-out /path/to/test_tensor.pt \
  --features us_aqi,pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,uv_index_clear_sky,uv_index,dust,aerosol_optical_depth \
  --radius-squared 1.0 \
  --houston-lat 29.7604 \
  --houston-lon -95.3698 \
  --dtype float32 \
  --train-pct 0.70 \
  --val-pct 0.15 \
  --test-pct 0.15 \
  --chunksize 200000 \
  --workers 3
```

---

## What each stage produces

### Stage 1: `01_filter_houston_radius.py`

Input:
- original CSV

Output:
- filtered CSV containing only locations within the configured Houston-area radius

This reduces the size of the downstream workload.

### Stage 2: `02_normalize_csv.py`

Input:
- filtered CSV

Output:
- normalized CSV
- normalization stats CSV

Each selected numeric column is normalized independently to `[0,1]`.

### Stage 3: `03_hilbert_sort_csv.py`

Input:
- normalized CSV

Output:
- Hilbert-sorted CSV
- layout CSV

The CSV gains spatial layout metadata and is sorted by `time`, then `x`, then `y`.

### Stage 4: `04_split_to_tensors.py`

Input:
- Hilbert-sorted CSV

Output:
- `train_tensor.pt`
- `val_tensor.pt`
- `test_tensor.pt`

Each tensor contains all Hilbert locations for its timestamp range.

---

## Inspecting saved tensors

You can load a tensor file in Python like this:

```python
import torch

obj = torch.load("/path/to/train_tensor.pt", map_location="cpu")
print(type(obj))
print(obj.keys() if isinstance(obj, dict) else "Not a dict")
```

If the script saves a dictionary, it will usually contain the tensor plus metadata such as:

- feature order
- timestamp list
- spatial dimensions

---

## Notes and assumptions

- The pipeline assumes the selected feature columns are numeric.
- The split is based on **unique sorted timestamps**, not row counts.
- Missing `(time, location)` combinations are typically filled with `NaN` in the final tensors unless the script was configured otherwise.
- If duplicate rows exist for the same `(time, latitude, longitude)`, later rows may overwrite earlier rows unless you modify the tensor-building logic.
- The Hilbert layout is **dense**, meaning it packs valid locations tightly into the spatial grid rather than preserving exact geographic spacing.

---

## Recommended workflow

1. Run the pipeline once with a moderate `--chunksize`.
2. Inspect:
   - the filtered CSV
   - the normalization stats CSV
   - the Hilbert layout CSV
3. Confirm that the feature order and tensor shape are what your model expects.
4. Increase `--chunksize` if you have spare RAM and want higher throughput.

---

## Troubleshooting

### The script says a feature column is missing
Check that the names passed to `--features` exactly match the CSV header names.

### The split percentages fail
Make sure:

```text
train_pct + val_pct + test_pct = 1.0
```

### The tensors are unexpectedly large
This can happen when:

- the Houston radius is large
- many unique locations remain after filtering
- many timestamps are present
- many feature channels are selected

### The normalized values look wrong
Inspect the normalization stats CSV to verify the min/max used for each column.

---

## Summary

Use `run_pipeline.sh` to execute the entire workflow from raw CSV to `[T,C,H,W]` train/validation/test tensors. The saved intermediate CSVs are there to make every transformation transparent and debuggable.
