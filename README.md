# Preprocessing Pipeline for RNN Air-Quality Forecasting

# NOTE:
Before using run_pipeline.sh you need to run strip_tz_info.py to make the time column an acceptable format. This is done through:  
`python3 strip_tz_info.py /path/to/input /path/to/output`

This repository contains a Python preprocessing pipeline for converting merged air-quality, weather, ZIP-level spatial context, and TRI emissions data into tensors suitable for PyTorch RNN training.

The pipeline performs:

1. **Merge** air-quality and weather CSVs on matching `time` and `zip`
2. **Compute ZIP-level spatial impact features** using roads and TRI facilities/chemicals
3. **Expand wind-direction columns** into cyclic `sin` / `cos` features
4. **Normalize** numeric features (excluding specified columns)
5. **Remove low-variance features**
6. **Apply PCA** to the remaining feature columns
7. **Encode ZIP locations with a Hilbert curve**
8. **Create a tensor** with dimensions `[time, PCA_features, hilbert_position]`
9. **Split** the tensor by timestep into train / validation / test
10. **Fill missing values independently** in each split using iterative 2D Lorenzo-style prediction
11. **Optionally export RNN-ready arrays** by flattening `[feature, hilbert]` per timestep for plain PyTorch RNN/GRU/LSTM usage

---

## Files

- `preprocessing-pipeline.py` — main Python pipeline
- `run_pipeline.sh` — bash wrapper for invoking the pipeline

---

## Input Data

### Required CSV inputs
- Air-quality CSV
- Weather CSV
- TRI facilities CSV
- TRI chemicals CSV

### Required shapefiles
- ZIP/ZCTA shapefile (for ZIP geometry / centroids)
- Roads shapefile (for nearby-road influence)

### Optional shapefile
- Place shapefile  
  Accepted as an argument for completeness, but not used by the current pipeline logic.

---

## Expected Core Columns

The pipeline assumes:

- a **time column** (default: `time`)
- a **ZIP column** (default: `zip`)

These can be overridden with:
- `--time-col`
- `--zip-col`

### Time format
The pipeline standardizes common timestamp formats into:

`YYYY-MM-DD HH:MM:SS`

### ZIP format
ZIP values are coerced to 5-digit ZIP codes.

---

## Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn geopandas shapely pyproj
```

Depending on your system, `geopandas` may also require system GIS libraries.

---

## How the pipeline works

### 1. Merge air-quality and weather
The two CSV files are merged using an **inner join** on:

- `time`
- `zip`

If both files contain non-key columns with the same name, only one copy is kept.

### 2. Compute ZIP-level spatial impact features
The pipeline creates several ZIP-level spatial features:

- `road_distance_m`
- `road_impact_score`
- `facility_count_nearby`
- `facility_impact_score`
- `overall_spatial_impact_score`

#### Road score
For each ZIP code, the ZIP centroid is used to measure distance to the nearest road.

Raw road score:

```text
exp(-distance_to_nearest_road_m / road_radius_m)
```

This is then scaled to `[0, 1]` over the ZIP codes in the dataset.

#### Facility score
For each TRI facility, the pipeline estimates a severity score using:

- total release / air emissions
- number of unique chemicals

Facility severity is computed as:

```text
0.7 * normalized_total_release + 0.3 * normalized_unique_chemical_count
```

Facility impact on a ZIP is then:

```text
facility_severity * exp(-distance_zip_to_facility_m / facility_radius_m)
```

The facility score for a ZIP is the sum of nearby facility impacts, then scaled to `[0, 1]`.

#### Overall spatial impact score
The final combined score is:

```text
0.4 * road_impact_score + 0.6 * facility_impact_score
```

### 3. Expand directional columns into sin/cos
Columns representing directions, such as:

- `wind_direction_10m`
- `wind_direction_100m`

are converted into:

- `<column>_sin`
- `<column>_cos`

This avoids the discontinuity problem of angular data, where `359` degrees and `1` degree are numerically far apart but physically close.

By default, the original direction columns are dropped after expansion.

### 4. Normalize features
Numeric columns are min-max normalized to `[0, 1]`:

```text
(x - min) / (max - min)
```

Columns listed in `--exclude-normalization` are not normalized.

### 5. Remove low-variance columns
Global variance is computed over the full dataset for each numeric feature, excluding columns listed in `--exclude-variance`.

Features with variance below `--variance-threshold` are removed.

### 6. Apply PCA
PCA is applied to the remaining numeric feature columns, excluding columns listed in `--exclude-pca`.

The number of PCA components is chosen automatically so that the retained explained variance is at least the value given by:

- `--pca-retained-variance`

The output after PCA keeps excluded columns such as `time` and `zip`, plus the PCA columns:

- `PC1`
- `PC2`
- `PC3`
- ...

### 7. Hilbert spatial encoding
ZIP centroids are projected into a quantized 2D grid and assigned Hilbert indices.

The pipeline stores metadata describing:

- ZIP code
- centroid coordinates
- grid coordinates
- Hilbert index
- Hilbert position

This preserves approximate spatial locality while mapping ZIPs into a 1D ordering.

### 8. Create tensor
A full tensor is created with shape:

```text
[time, feature, hilbert_position]
```

At this point, the feature axis contains the **PCA components**, not the original raw features.

### 9. Split by time
The full tensor is split into:

- training tensor
- validation tensor
- testing tensor

The split is performed along the **time dimension**, not randomly by row.  
Each split contains all spatial locations.

Fractions are controlled by:

- `--train-fraction`
- `--val-fraction`
- `--test-fraction`

These must sum to `1.0`.

### 10. Fill missing values independently
Each split tensor is filled independently using iterative 2D Lorenzo-style prediction over:

- time
- hilbert position

This is done **per PCA feature**.

That means:

- train missing values are filled using only train data
- validation missing values are filled using only validation data
- test missing values are filled using only test data

This avoids leakage across splits.

---

## Main arguments

### Required arguments

- `--air-quality`  
  Path to the air-quality CSV

- `--weather`  
  Path to the weather CSV

- `--tri-facilities`  
  Path to the TRI facilities CSV

- `--tri-chemicals`  
  Path to the TRI chemicals CSV

- `--zip-shapefile`  
  Path to the ZIP/ZCTA shapefile (`.shp`)

- `--roads-shapefile`  
  Path to the roads shapefile (`.shp`)

- `--output-dir`  
  Output directory for intermediates, metadata, and final tensors

- `--variance-threshold`  
  Threshold below which a feature is removed during variance filtering

- `--pca-retained-variance`  
  Fraction of variance PCA should retain, for example `0.95`

- `--train-fraction`  
  Fraction of timesteps used for training

- `--val-fraction`  
  Fraction of timesteps used for validation

- `--test-fraction`  
  Fraction of timesteps used for testing

### Optional arguments

- `--place-shapefile`  
  Accepted but not used by the current implementation

- `--time-col`  
  Default: `time`

- `--zip-col`  
  Default: `zip`

- `--exclude-normalization`  
  Comma-separated columns not to normalize  
  Default: `time,zip`

- `--exclude-variance`  
  Comma-separated columns excluded from variance filtering  
  Default: `time,zip`

- `--exclude-pca`  
  Comma-separated columns excluded from PCA  
  Default: `time,zip`

- `--hilbert-order`  
  Hilbert curve order  
  Default: `8`

- `--road-radius-km`  
  Road decay radius in kilometers  
  Default: `2.0`

- `--facility-radius-km`  
  Facility decay radius in kilometers  
  Default: `10.0`

- `--direction-columns`  
  Comma-separated direction columns to explicitly expand to sin/cos

- `--no-auto-detect-direction-columns`  
  Disable automatic detection of direction columns

- `--keep-original-direction-columns`  
  Keep original raw direction columns after creating sin/cos versions

---

## Sample run

```bash
bash run_pipeline.sh \
  --air-quality ../data/py-impl-data/hourly-air-quality-0215-0222.csv \
  --weather ../data/py-impl-data/hourly-weather-0215-0222.csv \
  --tri-facilities ../data/py-impl-data/tri_facilities_houston_2023.csv \
  --tri-chemicals ../data/py-impl-data/tri_chemicals_houston_2023.csv \
  --zip-shapefile ../data/texas-shape-data/texas_zcta.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --place-shapefile ../data/texas-shape-data/cb_2024_48_place_500k.shp \
  --output-dir ./pipeline_output \
  --time-col time \
  --zip-col zip \
  --exclude-normalization time,zip \
  --exclude-variance time,zip \
  --exclude-pca time,zip \
  --variance-threshold 1e-6 \
  --pca-retained-variance 0.95 \
  --train-fraction 0.70 \
  --val-fraction 0.15 \
  --test-fraction 0.15 \
  --direction-columns wind_direction_10m,wind_direction_100m
```

---

## Output structure

The pipeline creates the following directory structure inside `--output-dir`:

```text
output_dir/
├── intermediate/
├── metadata/
├── logs/
├── train_tensor_filled.npy
├── val_tensor_filled.npy
├── test_tensor_filled.npy
└── pipeline_full.log
```

### Intermediate files
Examples:

- `01_merged.csv`
- `02_with_spatial_impact.csv`
- `03_direction_expanded.csv`
- `04_normalized.csv`
- `05_variance_filtered.csv`
- `06_pca.csv`
- `07_hilbert_encoded.csv`
- `08_full_tensor.npy`
- `09_train_tensor.npy`
- `09_val_tensor.npy`
- `09_test_tensor.npy`

### Metadata files
Examples:

- spatial impact formulas and detected columns
- normalization min/max statistics
- variance report
- PCA equations and explained variance
- Hilbert mapping
- tensor metadata
- split metadata
- Lorenzo fill reports

### Logs
- `logs/pipeline_steps.log` — step-by-step log
- `pipeline_full.log` — summary log with parameters, outputs, and metadata file paths

---

## Resulting tensor meaning

The final filled tensors:

- `train_tensor_filled.npy`
- `val_tensor_filled.npy`
- `test_tensor_filled.npy`

have shape:

```text
[time, PCA_feature, hilbert_position]
```

These tensors use **PCA components** as the feature axis.

---

## Using the resulting `.npy` files in PyTorch

A `.npy` file can be loaded directly into NumPy, then converted to PyTorch:

```python
import numpy as np
import torch

x = np.load("train_tensor_filled.npy")
x = torch.from_numpy(x).float()
```

However, a plain PyTorch `RNN`, `GRU`, or `LSTM` expects input shaped like:

- `[seq_len, batch, input_size]`, or
- `[batch, seq_len, input_size]` if `batch_first=True`

The pipeline’s tensors are shaped as:

```text
[time, feature, hilbert_position]
```

So for a vanilla RNN, you should flatten `[feature, hilbert_position]` into one vector per timestep:

```python
import numpy as np
import torch

x = np.load("train_tensor_filled.npy")   # [T, F, H]
T, F, H = x.shape
x = x.reshape(T, F * H)                  # [T, F*H]
x = x[np.newaxis, :, :]                  # [1, T, F*H] for batch_first=True
x = torch.from_numpy(x).float()
```

Then use:

```python
import torch.nn as nn

model = nn.RNN(
    input_size=x.shape[2],
    hidden_size=128,
    num_layers=1,
    batch_first=True
)
y, h = model(x)
```

---

## Notes and caveats

- The current spatial impact score uses **ZIP centroids**, which is a practical approximation rather than a full within-ZIP exposure model.
- The Lorenzo fill is an imputation method over the tensor grid. It is useful for completing sparse tensor slots, but it is not a physical atmospheric transport model.
- PCA is applied **before** tensor creation, so the tensor features are PCA components rather than the original columns.
- To preserve the `zip` column through PCA and Hilbert mapping, keep:

```bash
--exclude-pca time,zip
```

If `zip` is not excluded from PCA, the Hilbert encoding step will fail because it needs ZIP codes.

---

## Recommended defaults

A good starting configuration is:

- `--variance-threshold 1e-6`
- `--pca-retained-variance 0.95`
- `--train-fraction 0.70`
- `--val-fraction 0.15`
- `--test-fraction 0.15`
- `--exclude-normalization time,zip`
- `--exclude-variance time,zip`
- `--exclude-pca time,zip`

---

## Troubleshooting

### TRI column detection fails
If the pipeline cannot detect TRI columns automatically, verify that the TRI files contain equivalents of:

- facility ID (`trifd`, `trifid`, etc.)
- latitude
- longitude
- chemical name
- release / emissions amount

### Hilbert step fails due to missing ZIP column
Make sure you passed:

```bash
--exclude-pca time,zip
```

### Train/validation/test split error
The three fractions must:

- all be greater than `0`
- sum to `1.0`

### Direction columns are not expanded
Either rely on auto-detection, or explicitly pass:

```bash
--direction-columns wind_direction_10m,wind_direction_100m
```
