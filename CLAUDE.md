# AI & Neural Networks — Course Project Context

## Student
- **University:** AITU (Astana IT University)
- **Course:** AI & Neural Networks
- **Duration:** 10-week assignment series (each builds on the previous)

## Dataset: SCANIA Component X (Predictive Maintenance)
- Real-world multivariate time series from **33,000+ heavy-duty trucks**
- Goal: **predict component failure** (binary classification: 0 = healthy, 1 = repaired/failed)
- Data is anonymized (sensor names are coded like `171_0`, `666_0`, `459_15`, etc.)

### Raw Data Files (`Assigment1/data/`)
| File | Description |
|------|-------------|
| `train_operational_readouts.csv` | Sporadic, unevenly sampled sensor time series (~1.2GB). 14 sensor variables → 107 columns including `vehicle_id` and `time_step` |
| `train_specifications.csv` | 8 categorical vehicle specs (`Spec_0` to `Spec_7`) |
| `train_tte.csv` | Time-to-event labels: `length_of_study_time_step`, `in_study_repair` (binary target) |
| `train_labels.csv` | Labels file |

### Key Column Notes
- `time_step` = operational time units (like an odometer), NOT calendar time
- `class_label` = binary target (renamed from `in_study_repair`): 0 = no repair, 1 = repaired
- `vehicle_id` = unique truck identifier

## Assignment 1 (Completed) — EDA & Feature Engineering
**Location:** `Assigment1/Assigment1.ipynb`

### What Was Done
1. **Data Loading** — Sampled operational readouts (full file is ~1.2GB), loaded specs and TTE
2. **Merged** all data into single DataFrame on `vehicle_id`
3. **Timestamp Processing** — Created `lifecycle_progress` (normalized 0–1 per vehicle), `hour_bin`, `day_bin`, `is_weekend`, `obs_number`
4. **Visualization** — Correlation matrix, histograms of top features, time series plots, repair distribution
5. **Missing Values** — Forward-fill within each vehicle → backward-fill → median fill
6. **Outlier Handling** — IQR method (1.5×IQR capping)
7. **Feature Engineering:**
   - Lag features (1, 2, 3 steps) for top 3 correlated sensors
   - Rolling means (windows: 3, 6, 12) for top 3 sensors
   - `time_of_day` category (morning/day/evening/night from hour_bin)
   - Rate of change (diff) features for top 3 sensors

### Output: `Assigment1/processed_data.csv`
- **50,000 rows** (sampled from full dataset)
- **~130+ columns** including original sensors, specs, engineered features
- Key engineered columns: `lifecycle_progress`, `*_lag_*`, `*_rolling_*`, `time_of_day`, `*_diff`
- Target column: `class_label` (binary)
- Ready for model building in subsequent assignments

### Libraries Used
- pandas, numpy, matplotlib, seaborn, sklearn (LabelEncoder)

## Assignment 2 — Current (In Progress)
**Location:** `Assigment2/` (empty as of 2026-03-31)

## Project Structure
```
Assigments/
├── CLAUDE.md          ← this file
├── Assigment1/
│   ├── Assigment1.ipynb
│   ├── Report_Assignment1.pdf
│   ├── data/           ← raw data files
│   ├── figures/
│   └── processed_data.csv  ← USE THIS for future assignments
└── Assigment2/
    └── (next assignment goes here)
```

## Important Notes
- The `processed_data.csv` from Assignment 1 is the starting point for all future assignments
- Data is sampled (50K rows from ~1.2GB), keep this in mind for model training
- Sensor column names are anonymized codes, not human-readable names
- The dataset is imbalanced (more healthy trucks than failed ones)
