# Soil Quality Assessment using Hyperspectral Imaging

This repository contains the implementation of a project for a capstone review, aiming to assess soil quality using hyperspectral imaging. The project involves preprocessing hyperspectral data, training several machine learning models to predict soil nutrient levels, and calculating a Soil Quality Index (SQI).

## Project Overview

The goal of this project is to build a model that can predict the levels of key soil nutrients (Phosphorus, Potassium, Magnesium, and pH) from hyperspectral images. The project also includes a method for calculating a Soil Quality Index (SQI) based on the predicted nutrient levels.

## Methodology

### 1. Data Preprocessing

The hyperspectral data is preprocessed using the following steps:
1.  **Savitzky-Golay Smoothing:** To reduce noise in the spectral data.
2.  **Standard Normal Variate (SNV):** To correct for scatter effects.
3.  **Multiplicative Scatter Correction (MSC):** To further correct for scatter effects.
4.  **Spatial Averaging:** The 11x11 spatial dimensions of the hyperspectral cubes are averaged to produce a single spectrum per sample.

### 2. Machine Learning Models

Three different machine learning models are trained and evaluated for each nutrient:
*   **Random Forest Regressor**
*   **Partial Least Squares (PLS) Regression**
*   **XGBoost Regressor**

The models are trained on the preprocessed hyperspectral data to predict the nutrient levels. The performance of the models is evaluated using the R-squared (R2) score.

### 3. Soil Quality Index (SQI)

A Soil Quality Index (SQI) is calculated as a weighted average of the normalized predicted nutrient values. The SQI provides a single score to represent the overall soil quality. In this implementation, equal weights are used for all nutrients.

## Results

The models were trained on a mock dataset, and the following results were obtained.

### Model Performance Comparison

The following chart compares the R2 scores of the different models for each nutrient.

![R2 Scores Comparison Updated](visualizations_updated/r2_scores_comparison_updated.png)

### Predicted vs. Actual Values

The following plots show the predicted vs. actual values for the best performing model for each nutrient.

| Phosphorus (P) | Potassium (K) |
| :---: | :---: |
| ![P vs Actual](visualizations_updated/P_pred_vs_actual_comparison.png) | ![K vs Actual](visualizations_updated/K_pred_vs_actual_comparison.png) |

| Magnesium (Mg) | pH |
| :---: | :---: |
| ![Mg vs Actual](visualizations_updated/Mg_pred_vs_actual_comparison.png) | ![pH vs Actual](visualizations_updated/pH_pred_vs_actual_comparison.png) |

## How to Run the Code

1.  **Install the required libraries:**
    ```bash
    pip install numpy scipy scikit-learn xgboost matplotlib
    ```
2.  **Run the `baseline_model.py` script:**
    ```bash
    python baseline_model.py
    ```
3.  **Run the `create_visualizations_updated.py` script to generate the visualizations:**
    ```bash
    python create_visualizations_updated.py
    ```

## Updates from Gemini CLI Session (January 29, 2026)

This section summarizes the key developments and model implementations performed during the recent Gemini CLI session.

### 1. Data Handling & Mock Labels
- **HYPERVIEW2 Dataset Integration:** The project now utilizes the actual HYPERVIEW2 dataset.
- **Mock P, K, Mg, pH Labels:** Due to a discrepancy between required target nutrients (P, K, Mg, pH) and available ground truth, a new script (`generate_mock_gt_pk_mg_ph.py`) was created to generate mock labels for P, K, Mg, pH. This allows continued model development.

### 2. Competitive Baseline Model (EagleEyes)
- **Implementation:** A new model, `eagle_eyes_baseline_model.py`, implements the "Competitive Baseline (EagleEyes)" model.
- **Features:** It uses spatially averaged hyperspectral data with handcrafted features, including spectral derivatives, SAVI, and RVI.
- **Architecture:** Employs an ensemble of Random Forest and K-Nearest Neighbors regressors.

### 3. HyperSoilNet-like Model (Phase 1)
- **Implementation:** Initial development of the "New Model for Capstone Project" is in `hypersoilnet_base.py`.
- **Preprocessing:** Includes band trimming to focus on the 462–938 nm range.
- **Advanced Feature Engineering:** Incorporates Singular Value Decomposition (SVD) and Fast Fourier Transform (FFT) features.
- **Model Training:** Trains individual Random Forest, PLS Regression, and XGBoost regressors for each nutrient (without ensembling at this stage).
- **Note on DWT:** Discrete Wavelet Transform (DWT) features were planned but excluded due to persistent library installation issues (`pywt`).

### How to Run New Models:
- **Generate Mock Labels:** `python generate_mock_gt_pk_mg_ph.py`
- **Run EagleEyes Baseline:** `python eagle_eyes_baseline_model.py`
- **Run HyperSoilNet Phase 1 Base Model:** `python hypersoilnet_base.py`

These steps are crucial for reproducing the current state and results of the project.