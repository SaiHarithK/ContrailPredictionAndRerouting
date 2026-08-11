# Contrail Prediction and Route Optimization using Physics-Informed Machine Learning

## Overview

This project presents a physics-informed machine learning framework for predicting persistent aircraft contrails and recommending alternative flight paths to reduce contrail formation.

The system combines ERA5 atmospheric reanalysis data with physics-based feature engineering and an XGBoost classifier to estimate contrail probability for individual flight segments. Based on the predicted risk, alternative routes are generated through lateral and altitude deviations, enabling contrail-aware route optimization.

---

## Features

- ERA5 weather data integration
- Physics-based feature engineering
- Relative Humidity with respect to Ice (RHi)
- Schmidt–Appleman Criterion (Critical Temperature & Delta T)
- Wind Speed
- Wind Shear
- Static Stability
- ISSR Depth
- XGBoost-based contrail prediction
- Candidate route generation
- Weather interpolation for arbitrary flight coordinates

---

## Project Pipeline

```
ERA5 Weather Data
        │
        ▼
Feature Engineering
        │
        ▼
Physics Feature Extraction
        │
        ▼
Label Generation
        │
        ▼
XGBoost Training
        │
        ▼
Weather Query
        │
        ▼
Candidate Route Generation
        │
        ▼
Route Optimization
```

---

## Technologies Used

- Python
- XGBoost
- NumPy
- Pandas
- Xarray
- ERA5 Reanalysis Dataset

---

## Project Structure

```
candidate_generator.py      # Generate alternative flight paths
download_era5.py            # Download ERA5 weather data
feature_engineering.py      # Generate physics-based features
generate_labels.py          # Generate training labels
label_generation.py         # Label generation functions
merged_era5.py              # Merge monthly ERA5 datasets
physics_engine.py           # Physics calculations
point_physics.py            # Feature computation for individual points
train_model.py              # Train XGBoost model
weather_query.py            # Retrieve interpolated weather
```

---

## Dataset

The project uses the **ERA5 Reanalysis Dataset** from the Copernicus Climate Data Store.

Due to GitHub file size limitations, the dataset is not included in this repository.

---

## Running the Project

### 1. Download ERA5 Data

```bash
python download_era5.py
```

### 2. Merge Monthly ERA5 Files

```bash
python merged_era5.py
```

### 3. Generate Features

```bash
python feature_engineering.py
```

### 4. Generate Labels

```bash
python generate_labels.py
```

### 5. Train the Model

```bash
python train_model.py
```

### 6. Query Weather

```bash
python weather_query.py
```

### 7. Generate Candidate Routes

```bash
python candidate_generator.py
```

---

## Machine Learning Model

**Classifier:** XGBoost

Input Features:

- Temperature
- Pressure Level
- Relative Humidity with respect to Ice (RHi)
- Delta T
- Wind Speed
- Wind Shear
- Static Stability
- ISSR Depth

Output:

- Contrail Probability
- Contrail / No Contrail Classification

---

## Future Work

- OpenSky API integration for real flight trajectories
- Complete route optimization module
- Route visualization
- Validation using observational datasets (e.g., CALIOP/IAGOS)

---

## Authors

Sai Harith K
