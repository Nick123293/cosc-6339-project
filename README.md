# Houston AQ/Weather Preprocessing Pipeline

This README documents the preprocessing workflow used to convert streamed Houston air-quality and weather CSV files into a single machine-learning-ready feature table. The pipeline merges hourly air-quality and weather data, normalizes time formatting, optionally removes redundant columns, filters ZIP Code Tabulation Area geometry to only the ZIP codes present in the dataset, precomputes static spatial relationships, and writes a final feature CSV with temporal, wind-direction, spatial-impact, lag, cardinality, and optional variance-filter metadata.

The workflow is built around these scripts:

- `merge_data_into_master_file.py`
- `strip_tz_info.py`
- `remove_column.py`
- `filter_houston_zcta.py`
- `preprocessing.py`

`filter_houston_zcta.py` is referenced by the pipeline comments, but its script body was not included in the provided files. The command and purpose below are based on the header documentation in `preprocessing.py`.

---

## 1. What the pipeline produces

The main output is:

```text
<output-dir>/all_features.csv
```

This file is a merged and feature-engineered table where rows are keyed by:

```text
zip, time
```

Depending on the command-line options used, the final table can include:

- Raw air-quality features, such as AQI and pollutant values.
- Raw weather features, such as wind speed, wind direction, temperature, humidity, precipitation, and radiation.
- Wind direction converted into sine/cosine columns.
- Calendar/time features such as month, hour, day of week, weekend flag, and cyclic sine/cosine encodings.
- Spatial impact features:
  - `road_impact_score`
  - `facility_impact_score`
- Lag features such as `us_aqi_past_1`, `us_aqi_past_2`, etc.
- Optional low-cardinality-filtered and variance-filtered columns.

The main script also writes metadata and logs:

```text
<output-dir>/metadata/pipeline_summary.json
<output-dir>/metadata/spatial_lookup.json
<output-dir>/metadata/cardinality_report.json       # if cardinality filtering is enabled
<output-dir>/metadata/variance_report.json          # if variance filtering is enabled
<output-dir>/logs/pipeline_steps.log
<output-dir>/intermediate/pre_filter_all_features.csv
<output-dir>/intermediate/pre_variance_all_features.csv
```

Temporary sorted run files are created during external sorting. They are removed by default unless `--keep-temp-files` is passed.

---

## 2. Data sources used

### 2.1 Streamed air-quality CSV files

**Used by:**

- `merge_data_into_master_file.py`
- `strip_tz_info.py`
- `remove_column.py`, optional
- `filter_houston_zcta.py`
- `preprocessing.py`

**Purpose:**

The air-quality files provide pollutant and AQI features by ZIP code and time. They are first appended into a single master air-quality CSV, then time-zone suffixes are stripped, and then they are merged with the weather master file in `preprocessing.py`.

Expected filenames must contain:

```text
_air_quality_
```

For example:

```text
feb_4thweek_air_quality_hourly_20260312_210107.csv
march_10_air_quality_hourly_20260401_120609.csv
```

The merge script sorts files using the month token and numeric part of the second token in the filename. That means the naming convention matters.

**How to gather:**

The referenced upstream GitHub repository documents hourly air-quality collection through the Open-Meteo Air Quality API. It also stores generated CSV outputs in its `data/` folder. The repository can be used as a reference implementation for gathering the streamed air-quality files:

```text
https://github.com/GlowSand/AQ_ML_Pipeline/tree/master
```

That repository lists Open-Meteo Air Quality API as the source for PM2.5, PM10, AQI, CO, NO2, ozone, and related air-quality variables.

---

### 2.2 Streamed weather CSV files

**Used by:**

- `merge_data_into_master_file.py`
- `strip_tz_info.py`
- `remove_column.py`, optional
- `preprocessing.py`

**Purpose:**

The weather files provide hourly meteorological features by ZIP code and time. In the current spatial-impact logic, the most important weather variables are:

```text
wind_speed_10m
wind_speed_100m
wind_direction_10m
wind_direction_100m
```

or, after direction expansion:

```text
wind_direction_10m_cos
wind_direction_10m_sin
wind_direction_100m_cos
wind_direction_100m_sin
```

These are used to compute downwind road and facility impact scores.

Expected filenames must contain:

```text
_weather_
```

For example:

```text
march_1stweek_weather_hourly_20260315_120000.csv
```

**How to gather:**

The referenced upstream GitHub repository documents hourly weather collection through the Open-Meteo Weather API. It lists weather variables such as temperature, humidity, wind, and precipitation as collected data sources.

---

### 2.3 TRI facilities CSV

**Used by:**

- `preprocessing.py`

**Expected file:**

```text
tri_facilities_houston.csv
```

**Purpose:**

This file provides the locations of toxic-release facilities near Houston. The current preprocessing script expects columns equivalent to:

```text
trifd
latitude
longitude
facility
total_air_emissions_lbs
```

The important columns used directly by `preprocessing.py` are:

```text
trifd
latitude
longitude
total_air_emissions_lbs
```

The script converts facility latitude/longitude into point geometry and computes ZIP-to-facility distances and directions. These are used to build `facility_impact_score`.

**How to gather:**

The provided `preprocessing.py` header says this project used `tri_facilities_houston.csv` from the repository/static data and did not require additional preprocessing before using it in `preprocessing.py`.

The referenced upstream GitHub repository also includes `process_tri_data.py` and notes that TRI text files are processed separately. Use that repository as a reference if regenerating TRI CSVs from raw EPA TRI files.

---

### 2.4 TRI chemicals CSV

**Used by:**

- `preprocessing.py`

**Expected file:**

```text
tri_chemicals_houston.csv
```

**Purpose:**

This file provides chemical-level release information for each TRI facility. The current preprocessing script expects columns equivalent to:

```text
trifd
chemical
total_air_emissions_lbs
```

It groups chemical data by facility and computes:

- Total reported air emissions per facility.
- Number of unique chemicals per facility.

Those values are combined into a facility severity term used in `facility_impact_score`.

**How to gather:**

Use the provided `tri_chemicals_houston.csv` from the project data/static data if available through Git LFS. If regenerating from raw TRI files, use the referenced AQ pipeline repository as a guide.

---

### 2.5 US Census ZIP Code Tabulation Area shapefile

**Used by:**

- `filter_houston_zcta.py`
- `preprocessing.py`

**Purpose:**

The ZCTA shapefile provides ZIP polygon geometry. The pipeline uses it to:

1. Filter the national ZCTA file to only ZIP codes appearing in the collected Houston data.
2. Compute ZIP centroids.
3. Measure distances and directions from ZIP centroids to roads and facilities.

The preprocessing script tries to detect one of these ZIP columns:

```text
ZCTA5CE20
ZCTA5CE10
GEOID20
GEOID10
zip
zcta
```

**How to gather:**

Download the 2020 national ZIP Code Tabulation Area shapefile from the US Census TIGER/Line shapefile portal:

```text
https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layergroup=ZIP%20Code%20Tabulation%20Areas
```

Select the national ZIP Code Tabulation Areas file, unzip it, and point `filter_houston_zcta.py` to the `.shp` file.

---

### 2.6 US Census TIGER/Line Texas roads shapefile

**Used by:**

- `preprocessing.py`

**Purpose:**

The roads shapefile provides road geometry for spatial-impact scoring. The pipeline finds roads within `--road-radius-km` of each ZIP centroid, computes the direction from the road to the ZIP centroid, and combines that direction with wind vectors to estimate a downwind road-impact score.

**How to gather:**

Download the 2025 Texas Primary and Secondary Roads shapefile from the US Census TIGER/Line roads portal:

```text
https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2025&layergroup=Roads
```

Under `Primary and Secondary Roads`, select Texas and download the shapefile. No size reduction is required before using it in this pipeline.

---

## 3. Getting data through Git LFS

Some project data is stored through Git Large File Storage. To download those files from a repository that uses Git LFS:

```bash
sudo apt install git-lfs

git lfs install

git clone <repo-url>
cd <repo-name>

git lfs pull
```

For this project, the expected project data includes the repository `data/` folder plus:

```text
tri_chemicals_houston.csv
tri_facilities_houston.csv
```

The referenced AQ pipeline repository contains `data/` and `static data/` folders and can be used as a model for organizing the data files.

---

## 4. Recommended directory layout

A clean project layout might look like this:

```text
project-root/
├── scripts/
│   ├── merge_data_into_master_file.py
│   ├── strip_tz_info.py
│   ├── remove_column.py
│   ├── filter_houston_zcta.py
│   └── preprocessing.py
├── data/
│   ├── raw-streamed/
│   │   ├── feb_4thweek_air_quality_hourly_20260312_210107.csv
│   │   ├── feb_4thweek_weather_hourly_20260312_210107.csv
│   │   └── ...
│   ├── important-locations/
│   │   ├── tri_facilities_houston.csv
│   │   └── tri_chemicals_houston.csv
│   ├── census-zcta-2020/
│   │   └── tl_2020_us_zcta520.shp
│   ├── texas-shape-data/
│   │   └── tl_2025_48_prisecroads.shp
│   ├── air-quality-master.csv
│   ├── weather-master.csv
│   ├── air-quality-master-tz-stripped.csv
│   ├── weather-master-tz-stripped.csv
│   └── houston_zcta_filtered.shp
└── data/pipeline-output/
```

A shapefile is actually multiple files with the same base name, usually including `.shp`, `.shx`, `.dbf`, `.prj`, and sometimes `.cpg`. Keep all of those files together in the same directory.

---

## 5. Python environment

Recommended Python packages:

```bash
pip install pandas numpy geopandas shapely pyogrio fiona pyproj
```

Depending on your system, installing GeoPandas may require system GIS libraries. On many Linux/WSL systems, using conda is often easier:

```bash
conda create -n aq-preprocess python=3.11 -y
conda activate aq-preprocess
conda install -c conda-forge pandas numpy geopandas shapely pyogrio fiona pyproj -y
```

---

## 6. Full pipeline order

Run the scripts in this order:

1. Merge raw streamed air-quality/weather files into master CSVs.
2. Strip timezone suffixes from the master CSVs.
3. Optionally remove redundant `city` and `state` columns.
4. Download and filter the national ZCTA shapefile to Houston/data ZIPs.
5. Run the main preprocessing pipeline.

---

## 7. Step-by-step commands

The commands below assume you are running them from the directory containing the scripts. Adjust paths as needed.

### Step 1: Merge raw streamed files into master CSVs

Use `merge_data_into_master_file.py` to append all collected air-quality and weather CSVs into two master files.

```bash
python merge_data_into_master_file.py \
  --input-dir ../data/raw-streamed \
  --state-file ../data/merge_state.json \
  --air-master ../data/air-quality-master.csv \
  --weather-master ../data/weather-master.csv
```

What this does:

- Searches `--input-dir` for `.csv` files.
- Classifies files containing `_air_quality_` as air-quality files.
- Classifies files containing `_weather_` as weather files.
- Sorts the files chronologically using their filename tokens.
- Appends new files only once using `--state-file`.
- Rebuilds the master file if newly discovered files belong earlier in chronological order.

You can rerun this command after adding new streamed files. The state JSON tracks which files were already merged.

---

### Step 2: Strip timezone information from both master files

Use `strip_tz_info.py` on both master files. The input and output files must be different paths.

```bash
python strip_tz_info.py \
  ../data/air-quality-master.csv \
  ../data/air-quality-master-tz-stripped.csv \
  --time-col time
```

```bash
python strip_tz_info.py \
  ../data/weather-master.csv \
  ../data/weather-master-tz-stripped.csv \
  --time-col time
```

This keeps only:

```text
YYYY-MM-DD HH:MM:SS
```

For example:

```text
2026-03-05 12:34:56-06:00 -> 2026-03-05 12:34:56
2026-03-05 12:34:56Z      -> 2026-03-05 12:34:56
```

---

### Step 3: Optionally remove redundant columns

Because the project focuses on Houston, `city` and `state` may be redundant. You can either remove them before running `preprocessing.py` or let `preprocessing.py` drop them with `--left-drop-columns city state --right-drop-columns city state`.

To remove them ahead of time:

```bash
python remove_column.py \
  ../data/air-quality-master-tz-stripped.csv \
  ../data/air-quality-master-clean.csv \
  city state
```

```bash
python remove_column.py \
  ../data/weather-master-tz-stripped.csv \
  ../data/weather-master-clean.csv \
  city state
```

If you remove the columns here, use the `*-clean.csv` files in `preprocessing.py`. If you do not remove them here, use the `--left-drop-columns` and `--right-drop-columns` options in the main preprocessing command.

---

### Step 4: Filter the ZCTA shapefile to ZIP codes in the data

After downloading the 2020 national ZCTA shapefile, run:

```bash
python filter_houston_zcta.py \
  --csv ../data/air-quality-master-tz-stripped.csv \
  --shp ../data/census-zcta-2020/tl_2020_us_zcta520.shp \
  --output ../data/houston_zcta_filtered.shp
```

You can use either the air-quality or weather CSV as `--csv`, as long as it contains the ZIP codes used by the pipeline.

This creates a smaller shapefile containing only the ZIP/ZCTA polygons needed for the collected Houston data.

---

### Step 5: Run the full preprocessing pipeline

Basic run:

```bash
python preprocessing.py \
  --air-quality ../data/air-quality-master-tz-stripped.csv \
  --weather ../data/weather-master-tz-stripped.csv \
  --tri-facilities ../data/important-locations/tri_facilities_houston.csv \
  --tri-chemicals ../data/important-locations/tri_chemicals_houston.csv \
  --zip-shapefile ../data/houston_zcta_filtered.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --output-dir ../data/pipeline-output \
  --chunk-rows 25000 \
  --temp-dir ../data/pipeline-output/temp-files \
  --left-drop-columns city state \
  --right-drop-columns city state \
  --feats-for-past us_aqi pm2_5 ozone wind_speed_100m wind_direction_100m_cos wind_direction_100m_sin \
  --num-past-feats 24
```

Smaller validation/test run:

```bash
python preprocessing.py \
  --air-quality ../data/air-quality-master-VALIDATION-tz-stripped.csv \
  --weather ../data/weather-master-VALIDATION-tz-stripped.csv \
  --tri-facilities ../data/important-locations/tri_facilities_houston.csv \
  --tri-chemicals ../data/important-locations/tri_chemicals_houston.csv \
  --zip-shapefile ../data/houston_zcta_filtered.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --output-dir ../data/pipeline-output \
  --chunk-rows 10000 \
  --temp-dir ../data/pipeline-output/temp-files \
  --left-drop-columns city state \
  --right-drop-columns city state \
  --feats-for-past us_aqi \
  --num-past-feats 8
```

Run with cardinality filtering:

```bash
python preprocessing.py \
  --air-quality ../data/air-quality-master-tz-stripped.csv \
  --weather ../data/weather-master-tz-stripped.csv \
  --tri-facilities ../data/important-locations/tri_facilities_houston.csv \
  --tri-chemicals ../data/important-locations/tri_chemicals_houston.csv \
  --zip-shapefile ../data/houston_zcta_filtered.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --output-dir ../data/pipeline-output \
  --chunk-rows 25000 \
  --left-drop-columns city state \
  --right-drop-columns city state \
  --feats-for-past us_aqi pm2_5 ozone \
  --num-past-feats 24 \
  --cardinality-threshold 2
```

Run with both cardinality and normalized variance filtering:

```bash
python preprocessing.py \
  --air-quality ../data/air-quality-master-tz-stripped.csv \
  --weather ../data/weather-master-tz-stripped.csv \
  --tri-facilities ../data/important-locations/tri_facilities_houston.csv \
  --tri-chemicals ../data/important-locations/tri_chemicals_houston.csv \
  --zip-shapefile ../data/houston_zcta_filtered.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --output-dir ../data/pipeline-output \
  --chunk-rows 25000 \
  --left-drop-columns city state \
  --right-drop-columns city state \
  --feats-for-past us_aqi pm2_5 ozone \
  --num-past-feats 24 \
  --cardinality-threshold 2 \
  --variance-threshold 0.0001
```

---

## 8. Main `preprocessing.py` options

### Required inputs

```text
--air-quality        Path to the air-quality master CSV.
--weather            Path to the weather master CSV.
--tri-facilities     Path to tri_facilities_houston.csv.
--tri-chemicals      Path to tri_chemicals_houston.csv.
--zip-shapefile      Path to filtered Houston/ZCTA .shp file.
--roads-shapefile    Path to Texas roads .shp file.
--output-dir         Directory where final CSV, metadata, logs, and intermediates are written.
```

### Sorting and temporary files

```text
--chunk-rows         Number of rows per in-memory sort chunk. Default: 25000.
--temp-dir           Directory for temporary sorted run CSVs.
--keep-temp-files    Keep temporary sort-run files instead of deleting them.
```

The script uses an external-sort pattern so that large input CSVs do not need to be loaded fully into memory. It sorts chunks by `zip, time`, then performs a heap-based k-way merge.

### Column dropping

```text
--left-drop-columns city state
--right-drop-columns city state
```

The left input is the air-quality CSV. The right input is the weather CSV.

Do not drop required key columns:

```text
zip
time
```

### Spatial radius options

```text
--road-radius-km 2.0
--facility-radius-km 10.0
```

Roads within `road-radius-km` and facilities within `facility-radius-km` of a ZIP centroid are considered in spatial-impact scoring.

### Wind options

```text
--facility-wind-mode 10m|100m|blend
--facility-wind-blend-100m 0.7
--road-wind-mode 10m|100m|blend
--road-wind-blend-100m 0.0
```

By default:

- Facility impact uses a blend of 10m and 100m wind, weighted 70% toward 100m wind.
- Road impact uses 10m wind.

### Direction-column options

```text
--direction-columns "wind_direction_10m,wind_direction_100m"
--no-auto-detect-direction-columns
--keep-original-direction-columns
```

By default, direction-like columns are auto-detected and expanded into sine/cosine features. The original direction columns are dropped unless `--keep-original-direction-columns` is passed.

### Lag-feature options

```text
--feats-for-past us_aqi pm2_5 ozone
--num-past-feats 24
```

This creates lag columns for each listed feature. For example, with `--feats-for-past us_aqi --num-past-feats 3`, the output includes:

```text
us_aqi_past_1
us_aqi_past_2
us_aqi_past_3
```

Lag state is tracked separately by ZIP code.

### Cardinality filter options

```text
--cardinality-threshold 2
--exclude-cardinality time,zip,road_impact_score,facility_impact_score,...
```

Columns with cardinality below the threshold are removed unless they are excluded from the cardinality filter.

### Variance filter options

```text
--variance-threshold 0.0001
--exclude-variance time,zip,road_impact_score,facility_impact_score
```

The variance filter uses normalized variance:

```text
variance / (max - min)^2
```

Columns below the threshold are removed unless they are excluded from the variance filter.

---

## 9. How spatial-impact scoring works

The script precomputes ZIP-to-road and ZIP-to-facility relationships once, before streaming the merged rows. This avoids recalculating geometry distances for every row.

### Facility impact

For each ZIP centroid, the script finds nearby TRI facilities within `--facility-radius-km`. For each facility, it computes:

- Direction vector from facility to ZIP centroid.
- Distance decay based on facility distance.
- Facility severity based on total air emissions and number of unique chemicals.

For each row, it computes a wind vector and projects the wind onto the facility-to-ZIP direction. Only positive downwind projections contribute to the score.

Conceptually:

```text
facility_impact_score = sum(severity * distance_decay * max(dot(wind_vector, source_to_zip_unit_vector), 0))
```

### Road impact

For each ZIP centroid, the script finds nearby roads within `--road-radius-km`. For each road, it computes:

- Nearest road point to the ZIP centroid.
- Direction vector from the road to the ZIP centroid.
- Distance decay based on road distance.

For each row, it projects the road wind vector onto the road-to-ZIP direction. Only positive downwind projections contribute to the score.

Conceptually:

```text
road_impact_score = sum(distance_decay * max(dot(wind_vector, road_to_zip_unit_vector), 0))
```

---

## 10. Troubleshooting notes

### `zip` or `time` column errors

The main script expects both air-quality and weather CSVs to contain:

```text
zip
time
```

These are the join keys. Do not remove them.

### Shapefile not found or missing columns

Make sure all shapefile sidecar files are present in the same directory:

```text
.shp
.shx
.dbf
.prj
.cpg, if present
```

If the ZIP shapefile cannot be matched, check that it has one of these columns:

```text
ZCTA5CE20, ZCTA5CE10, GEOID20, GEOID10, zip, zcta
```

### CRS/distance confusion

The spatial precompute converts ZIP polygons, roads, and facilities to EPSG:3857 before distance calculations. In that projected CRS, distance units are meters. Radius arguments are given in kilometers and converted to meters internally.

### Input and output path cannot be the same for `strip_tz_info.py`

Use separate paths:

```bash
python strip_tz_info.py input.csv output.csv
```

Do not overwrite the input file directly.

### Filename pattern matters for master merging

`merge_data_into_master_file.py` expects filenames where:

- The first underscore-separated token is a month, such as `feb` or `march`.
- The second token begins with a number, such as `4thweek`, `1stweek`, or `10`.
- The filename contains `_air_quality_` or `_weather_`.

Files that do not match this pattern may not be merged correctly.

---

## 11. Minimal end-to-end command block

```bash
# 1. Merge streamed raw files.
python merge_data_into_master_file.py \
  --input-dir ../data/raw-streamed \
  --state-file ../data/merge_state.json \
  --air-master ../data/air-quality-master.csv \
  --weather-master ../data/weather-master.csv

# 2. Strip timezone suffixes.
python strip_tz_info.py ../data/air-quality-master.csv ../data/air-quality-master-tz-stripped.csv --time-col time
python strip_tz_info.py ../data/weather-master.csv ../data/weather-master-tz-stripped.csv --time-col time

# 3. Filter ZCTA polygons to the ZIP codes in the data.
python filter_houston_zcta.py \
  --csv ../data/air-quality-master-tz-stripped.csv \
  --shp ../data/census-zcta-2020/tl_2020_us_zcta520.shp \
  --output ../data/houston_zcta_filtered.shp

# 4. Run preprocessing.
python preprocessing.py \
  --air-quality ../data/air-quality-master-tz-stripped.csv \
  --weather ../data/weather-master-tz-stripped.csv \
  --tri-facilities ../data/important-locations/tri_facilities_houston.csv \
  --tri-chemicals ../data/important-locations/tri_chemicals_houston.csv \
  --zip-shapefile ../data/houston_zcta_filtered.shp \
  --roads-shapefile ../data/texas-shape-data/tl_2025_48_prisecroads.shp \
  --output-dir ../data/pipeline-output \
  --chunk-rows 25000 \
  --left-drop-columns city state \
  --right-drop-columns city state \
  --feats-for-past us_aqi pm2_5 ozone wind_speed_100m wind_direction_100m_cos wind_direction_100m_sin \
  --num-past-feats 24
```

---

## 12. Reference repository

The following GitHub repository is useful as a reference for the original data-collection side of the project:

```text
https://github.com/GlowSand/AQ_ML_Pipeline/tree/master
```

Its README describes a broader AQ/ML pipeline that pulls from Open-Meteo Air Quality, Open-Meteo Weather, EPA FRS, US Census ACS, OSMnx/OpenStreetMap, and TRI files. This preprocessing pipeline mainly consumes the resulting streamed air-quality/weather CSVs, TRI CSVs, Census ZCTA shapefiles, and Census roads shapefiles.
