# Urban Heat Island (UHI) Prediction

This project focuses on predicting Urban Heat Island (UHI) effects using ML techniques. It processes various geospatial and environmental datasets to model and predict UHI intensity across different urban areas in New york and the Bronx.

## Project Overview

Urban Heat Islands (UHI) refer to the phenomenon where urban areas experience higher temperatures than their rural surroundings due to human activities and urban development. This project aims to:

- Process and integrate multiple geospatial datasets
- Extract meaningful features related to urban heat patterns
- Build and evaluate machine learning models to predict UHI intensity
- Provide interpretable insights into factors contributing to UHI effects

## Data Sources

The project utilizes the following data sources:

- **Satellite Imagery**: Sentinel-2 and Landsat data for surface temperature and land cover analysis
- **Land Cover Data**: Polygonized land cover classifications
- **Building Footprints**: Detailed building data for urban structure analysis
- **Weather Data**: Local meteorological observations
- **Geospatial Features**: Various derived geographical features and indices

## Project Structure

### Main Scripts

1. **Data Processing**
   - `GetPrecomputedData.py`: Extracts statistical and textural features from satellite imagery at multiple scales
   - `LandCover_Polygonized.py`: Processes land cover data to compute metrics like coverage, fragmentation, and proximity to key features
   - `Projected_BuildingFP.py`: Analyzes building footprints to derive urban structure features

2. **Modeling**
   - `Model_dev.ipynb`: Jupyter notebook containing the main modeling workflow using  CatBoost
   - `HyperParameter_optuna.py`: Performs hyperparameter optimization for CatBoost models using Optuna  
   - `CNN` folder is dedicated to the Convolutional Neural Network (CNN) models with Landcover data.

3. **Model Evaluation**
   - The notebook includes comprehensive model evaluation using metrics like R² and RMSE
   - SHAP (SHapley Additive exPlanations) is used for feature importance analysis

## Key Features

- **Multi-scale Feature Extraction**: Processes data at various spatial scales to capture different aspects of urban environments
- **Model Interpretability**: Incorporates SHAP values to understand feature contributions
- **Feature Selection**: Implements automated feature selection based on importance scores
- **Hyperparameter Optimization**: Uses Optuna for efficient hyperparameter search

## Dependencies

- Core: `pandas`, `numpy`, `xarray`
- Geospatial: `geopandas`, `rioxarray`, `rasterio`, `shapely`
- Machine Learning: `pytorch`, `xgboost`, `catboost`, `scikit-learn`, `optuna`
- Visualization: `matplotlib`, `seaborn`, `shap`

## Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the data processing scripts in the following order:
   ```bash
   python GetPrecomputedData.py
   python LandCover_Polygonized.py
   python Projected_BuildingFP.py
   ```
