import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle
import os

def preprocess_data(hyperspectral_cubes, smooth=True, snv=True, msc=True):
    """
    Preprocesses the hyperspectral data.

    Args:
        hyperspectral_cubes (np.ndarray): The hyperspectral data cubes.
        smooth (bool): Apply Savitzky-Golay smoothing.
        snv (bool): Apply Standard Normal Variate.
        msc (bool): Apply Multiplicative Scatter Correction.

    Returns:
        np.ndarray: The preprocessed hyperspectral data.
    """
    # Average the spatial dimensions (11x11) to get a single spectrum per sample
    spectra = np.mean(hyperspectral_cubes, axis=(2, 3))
    
    if smooth:
        spectra = savgol_filter(spectra, window_length=5, polyorder=2, axis=1)
        
    if snv:
        spectra = (spectra - np.mean(spectra, axis=1, keepdims=True)) / np.std(spectra, axis=1, keepdims=True)

    if msc:
        mean_spectrum = np.mean(spectra, axis=0)
        corrected_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            p = np.polyfit(mean_spectrum, spectra[i, :], 1)
            corrected_spectra[i, :] = (spectra[i, :] - p[1]) / p[0]
        spectra = corrected_spectra

    return spectra

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains a Random Forest Regressor and evaluates its performance.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest R2 score for {nutrient_name}: {r2:.4f}")
    return model, r2

def train_and_evaluate_pls(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains a PLS Regressor and evaluates its performance.
    """
    model = PLSRegression(n_components=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"PLSRegression R2 score for {nutrient_name}: {r2:.4f}")
    return model, r2

def train_and_evaluate_xgb(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains an XGBoost Regressor and evaluates its performance.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost R2 score for {nutrient_name}: {r2:.4f}")
    return model, r2

def calculate_sqi(p, k, mg, ph):
    """
    Calculates the Soil Quality Index (SQI).
    """
    # Normalize each nutrient to a 0-1 range (assuming min=0, max can be estimated from data)
    p_norm = p / (np.max(p) if np.max(p) > 0 else 1)
    k_norm = k / (np.max(k) if np.max(k) > 0 else 1)
    mg_norm = mg / (np.max(mg) if np.max(mg) > 0 else 1)
    ph_norm = ph / (np.max(ph) if np.max(ph) > 0 else 1)
    
    # Calculate SQI with equal weights
    sqi = 0.25 * p_norm + 0.25 * k_norm + 0.25 * mg_norm + 0.25 * ph_norm
    return sqi

def main(data_path="dataset/HYPERVIEW2/mock_data.npz"):
    """
    Main function to run the baseline model training and evaluation.
    """
    data = np.load(data_path)
    hyperspectral_cubes = data["hyperspectral_cubes"]
    X = preprocess_data(hyperspectral_cubes)
    
    nutrients = ["P", "K", "Mg", "pH"]
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    results = {}
    for nutrient in nutrients:
        print(f"--- Training models for {nutrient} ---")
        y = data[nutrient]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": train_and_evaluate_rf,
            "PLSRegression": train_and_evaluate_pls,
            "XGBoost": train_and_evaluate_xgb,
        }

        results[nutrient] = {}
        for model_name, train_func in models.items():
            model, r2 = train_func(X_train, X_test, y_train, y_test, nutrient)
            results[nutrient][model_name] = {"model": model, "r2": r2, "X_test": X_test, "y_test": y_test, "y_pred": model.predict(X_test)}
            
            model_path = os.path.join(models_dir, f"{nutrient}_{model_name.lower().replace(' ', '_')}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"{model_name} model for {nutrient} saved to {model_path}")
        print("-" * 30)

    # Calculate and print SQI for the test set using the best model for each nutrient
    best_models = {}
    for nutrient in nutrients:
        best_model_name = max(results[nutrient], key=lambda model_name: results[nutrient][model_name]['r2'])
        best_models[nutrient] = results[nutrient][best_model_name]
    
    p_pred = best_models['P']['y_pred']
    k_pred = best_models['K']['y_pred']
    mg_pred = best_models['Mg']['y_pred']
    ph_pred = best_models['pH']['y_pred']
    
    sqi_pred = calculate_sqi(p_pred, k_pred, mg_pred, ph_pred)
    print("\n--- Soil Quality Index (SQI) ---")
    print(f"Predicted SQI for the test set (first 5 samples):\n{sqi_pred[:5]}")

    # Save results for visualization
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
