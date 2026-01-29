import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import pickle
import os

# --- Helper function to load all hyperspectral data and labels ---
def load_hyperview2_data(data_dir="data/HYPERVIEW2"):
    """
    Loads hyperspectral data (.npz files) and corresponding mock nutrient labels (.csv).

    Args:
        data_dir (str): Base directory for HYPERVIEW2 data.

    Returns:
        tuple: (hyperspectral_cubes_array, labels_df)
            hyperspectral_cubes_array (np.ndarray): Stacked numpy array of hyperspectral cubes.
            labels_df (pd.DataFrame): DataFrame containing sample_index and nutrient labels,
                                      filtered to include only samples with available HSI data.
    """
    hsi_data_list = []
    
    # Load mock P, K, Mg, pH labels
    labels_path = os.path.join(data_dir, "train_gt_pk_mg_ph_mock.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Mock labels CSV not found at {labels_path}. Please run generate_mock_gt_pk_mg_ph.py first.")
    labels_df = pd.read_csv(labels_path)
    
    hsi_train_airborne_dir = os.path.join(data_dir, "train", "hsi_airborne")
    hsi_train_satellite_dir = os.path.join(data_dir, "train", "hsi_satellite")
    
    # Get all .npz files and store their paths with sample_index
    file_sample_map = {}
    if os.path.exists(hsi_train_airborne_dir):
        for f in os.listdir(hsi_train_airborne_dir):
            if f.endswith('.npz'):
                sample_idx = int(f.split('.')[0])
                file_sample_map[sample_idx] = os.path.join(hsi_train_airborne_dir, f)
    if os.path.exists(hsi_train_satellite_dir):
        for f in os.listdir(hsi_train_satellite_dir):
            if f.endswith('.npz'):
                sample_idx = int(f.split('.')[0])
                # Prioritize airborne if both exist for a sample, or handle collision as needed
                # For now, just add if not already present
                if sample_idx not in file_sample_map:
                    file_sample_map[sample_idx] = os.path.join(hsi_train_satellite_dir, f)

    # Filter labels_df to only include samples for which we have HSI data
    labels_df = labels_df[labels_df['sample_index'].isin(file_sample_map.keys())].sort_values('sample_index').reset_index(drop=True)
    
    if len(labels_df) == 0:
        raise ValueError("No matching hyperspectral data found for the generated mock labels after filtering.")

    # Load hyperspectral cubes in the order of labels_df's sample_index
    for idx in labels_df['sample_index']:
        file_path = file_sample_map[idx]
        try:
            hsi_data = np.load(file_path)['data'] # (num_bands, height, width)
            # Perform spatial averaging here to handle inhomogeneous shapes
            # Convert (num_bands, height, width) to (num_bands,)
            averaged_spectrum = np.mean(hsi_data, axis=(1, 2))
            hsi_data_list.append(averaged_spectrum)
        except Exception as e:
            print(f"Error loading or processing {file_path}: {e}. Removing sample {idx} from labels.")
            # Remove this sample from labels_df if its data can't be loaded
            labels_df = labels_df[labels_df['sample_index'] != idx]
    
    # Re-index labels_df after potential removals
    labels_df = labels_df.reset_index(drop=True)

    if not hsi_data_list:
        raise ValueError("No hyperspectral data loaded successfully after processing.")
        
    # Stack the 1D spectra to form a single numpy array (num_samples, num_bands)
    hyperspectral_spectra_array = np.array(hsi_data_list)
    
    return hyperspectral_spectra_array, labels_df

def calculate_spectral_derivatives(spectra):
    """
    Calculates first and second order spectral derivatives.
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_bands).
    Returns:
        np.ndarray: Concatenated array of original spectra, first and second derivatives.
    """
    # Ensure spectra has enough bands for differentiation
    if spectra.shape[1] < 3:
        # Cannot calculate second derivative if less than 3 bands
        # Cannot calculate first derivative if less than 2 bands
        print("Warning: Not enough bands to calculate spectral derivatives. Skipping derivative features.")
        return np.array([]) # Return empty array if not enough bands

    # Calculate first derivative
    # Pad to maintain original number of bands, e.g., replicate last value
    first_derivative = np.diff(spectra, axis=1, append=spectra[:, -1:]) 
    
    # Calculate second derivative
    # Pad to maintain original number of bands, e.g., replicate last value
    second_derivative = np.diff(first_derivative, axis=1, append=first_derivative[:, -1:]) 
    
    return np.concatenate((first_derivative, second_derivative), axis=1)

def calculate_savi(spectra, red_band_idx, nir_band_idx, L=0.5):
    """
    Calculates the Soil Adjusted Vegetation Index (SAVI).
    SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_bands).
        red_band_idx (int): Index of the Red band.
        nir_band_idx (int): Index of the NIR band.
        L (float): Soil brightness correction factor.
    Returns:
        np.ndarray: 1D array of SAVI values.
    """
    # Ensure band indices are valid
    if red_band_idx >= spectra.shape[1] or nir_band_idx >= spectra.shape[1]:
        print(f"Warning: Red ({red_band_idx}) or NIR ({nir_band_idx}) band index out of bounds ({spectra.shape[1]} bands). Returning zeros for SAVI.")
        return np.zeros((spectra.shape[0], 1))

    red = spectra[:, red_band_idx]
    nir = spectra[:, nir_band_idx]
    
    # Avoid division by zero
    denominator = (nir + red + L)
    savi = np.zeros_like(red, dtype=float)
    non_zero_denom = denominator != 0
    savi[non_zero_denom] = ((nir[non_zero_denom] - red[non_zero_denom]) / denominator[non_zero_denom]) * (1 + L)
    
    return savi.reshape(-1, 1) # Reshape to (num_samples, 1)

def calculate_rvi(spectra, red_band_idx, nir_band_idx):
    """
    Calculates the Ratio Vegetation Index (RVI).
    RVI = NIR / Red
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_bands).
        red_band_idx (int): Index of the Red band.
        nir_band_idx (int): Index of the NIR band.
    Returns:
        np.ndarray: 1D array of RVI values.
    """
    # Ensure band indices are valid
    if red_band_idx >= spectra.shape[1] or nir_band_idx >= spectra.shape[1]:
        print(f"Warning: Red ({red_band_idx}) or NIR ({nir_band_idx}) band index out of bounds ({spectra.shape[1]} bands). Returning zeros for RVI.")
        return np.zeros((spectra.shape[0], 1))

    red = spectra[:, red_band_idx]
    nir = spectra[:, nir_band_idx]
    
    # Avoid division by zero
    rvi = np.zeros_like(red, dtype=float)
    non_zero_red = red != 0
    rvi[non_zero_red] = nir[non_zero_red] / red[non_zero_red]
    
    return rvi.reshape(-1, 1) # Reshape to (num_samples, 1)


# Define constants for band indices for feature calculations
RED_BAND_IDX = 80  # Approximate band index for 669.85 nm in hsi_aerial_wavelengths
NIR_BAND_IDX = 138 # Approximate band index for 855.26 nm in hsi_aerial_wavelengths

def preprocess_data(hyperspectral_spectra, smooth=True, snv=True, msc=True):
    """
    Preprocesses the spatially averaged hyperspectral data and extracts handcrafted features.
    Args:
        hyperspectral_spectra (np.ndarray): 2D array of spatially averaged spectra (num_samples, num_bands).
        smooth (bool): Apply Savitzky-Golay smoothing.
        snv (bool): Apply Standard Normal Variate.
        msc (bool): Apply Multiplicative Scatter Correction.
    Returns:
        np.ndarray: The preprocessed data with handcrafted features.
    """
    spectra = hyperspectral_spectra # Input is already spatially averaged
    
    # Apply original preprocessing steps
    if smooth:
        if spectra.shape[1] < 5:
            smooth = False
            print("Warning: Not enough bands for Savitzky-Golay smoothing. Skipping smoothing.")
        else:
            window_length = min(5, spectra.shape[1] if spectra.shape[1] % 2 == 1 else spectra.shape[1] - 1)
            if window_length < 3:
                window_length = 3
            spectra = savgol_filter(spectra, window_length=window_length, polyorder=2, axis=1)
        
    if snv:
        std_dev = np.std(spectra, axis=1, keepdims=True)
        spectra = (spectra - np.mean(spectra, axis=1, keepdims=True)) / (std_dev + 1e-6)

    if msc:
        mean_spectrum = np.mean(spectra, axis=0)
        corrected_spectra = np.zeros_like(spectra)
        for i in range(spectra.shape[0]):
            p = np.polyfit(mean_spectrum, spectra[i, :], 1)
            if abs(p[0]) < 1e-6:
                corrected_spectra[i, :] = spectra[i, :]
            else:
                corrected_spectra[i, :] = (spectra[i, :] - p[1]) / p[0]
        spectra = corrected_spectra

    # --- Handcrafted Feature Extraction ---
    # Spectral Derivatives
    derivative_features = calculate_spectral_derivatives(spectra)
    
    # Vegetation Indices
    savi_feature = calculate_savi(spectra, RED_BAND_IDX, NIR_BAND_IDX)
    rvi_feature = calculate_rvi(spectra, RED_BAND_IDX, NIR_BAND_IDX)

    # Combine all features
    if derivative_features.size > 0:
        X_features = np.concatenate((spectra, derivative_features, savi_feature, rvi_feature), axis=1)
    else:
        # If no derivatives calculated, just use spectra and VIs
        X_features = np.concatenate((spectra, savi_feature, rvi_feature), axis=1)

    return X_features

def train_and_evaluate_rf(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains a Random Forest Regressor and evaluates its performance.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Random Forest R2 score for {nutrient_name}: {r2:.4f}")
    return model, y_pred, r2 # Return y_pred for ensembling

def train_and_evaluate_knn(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains a K-Nearest Neighbors Regressor and evaluates its performance.
    """
    model = KNeighborsRegressor(n_neighbors=5) # Example n_neighbors, can be tuned
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"K-Nearest Neighbors R2 score for {nutrient_name}: {r2:.4f}")
    return model, y_pred, r2 # Return y_pred for ensembling

# The original PLS and XGB functions are removed as per EagleEyes spec
# def train_and_evaluate_pls(...)
# def train_and_evaluate_xgb(...)

def calculate_sqi(p, k, mg, ph):
    """
    Calculates the Soil Quality Index (SQI).
    (This function remains the same)
    """
    # Normalize each nutrient to a 0-1 range (assuming min=0, max can be estimated from data)
    # Ensure no division by zero if max_val is 0.
    p_norm = p / (np.max(p) if np.max(p) > 0 else 1)
    k_norm = k / (np.max(k) if np.max(k) > 0 else 1)
    mg_norm = mg / (np.max(mg) if np.max(mg) > 0 else 1)
    ph_norm = ph / (np.max(ph) if np.max(ph) > 0 else 1)
    
    # Calculate SQI with equal weights
    sqi = 0.25 * p_norm + 0.25 * k_norm + 0.25 * mg_norm + 0.25 * ph_norm
    return sqi

def main(data_dir="data/HYPERVIEW2"):
    """
    Main function to run the EagleEyes baseline model training and evaluation.
    """
    print("Loading HYPERVIEW2 data and mock labels for EagleEyes baseline...")
    hyperspectral_spectra, labels_df = load_hyperview2_data(data_dir) # Changed variable name
    print(f"Loaded {hyperspectral_spectra.shape[0]} hyperspectral spectra and {len(labels_df)} corresponding labels.")

    # Now preprocess_data expects 2D spectra
    X = preprocess_data(hyperspectral_spectra) 
    
    nutrients = ["P", "K", "Mg", "pH"]
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    results = {}
    for nutrient in nutrients:
        print(f"\n--- Training models for {nutrient} ---")
        y = labels_df[nutrient].values # Use mock labels

        # Ensure that X and y have the same number of samples
        if X.shape[0] != len(y):
            # This should ideally not happen if load_hyperview2_data is robust
            # and filters labels_df based on available HSI data
            print(f"Warning: Mismatch in number of samples between features ({X.shape[0]}) and labels ({len(y)}) for {nutrient}.")
            # Attempt to align them by taking the minimum
            min_samples = min(X.shape[0], len(y))
            X_aligned = X[:min_samples]
            y_aligned = y[:min_samples]
        else:
            X_aligned = X
            y_aligned = y

        X_train, X_test, y_train, y_test = train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=42)

        # Train Random Forest
        rf_model, rf_y_pred, rf_r2 = train_and_evaluate_rf(X_train, X_test, y_train, y_test, nutrient)
        
        # Train K-Nearest Neighbors
        knn_model, knn_y_pred, knn_r2 = train_and_evaluate_knn(X_train, X_test, y_train, y_test, nutrient)

        # Ensemble Prediction (Simple Averaging)
        ensemble_y_pred = (rf_y_pred + knn_y_pred) / 2
        ensemble_r2 = r2_score(y_test, ensemble_y_pred)
        print(f"Ensemble (RF+KNN) R2 score for {nutrient}: {ensemble_r2:.4f}")

        results[nutrient] = {
            "Random Forest": {"model": rf_model, "r2": rf_r2, "y_pred": rf_y_pred},
            "K-Nearest Neighbors": {"model": knn_model, "r2": knn_r2, "y_pred": knn_y_pred},
            "Ensemble": {"y_pred": ensemble_y_pred, "r2": ensemble_r2, "y_test": y_test} # Store y_test for SQI
        }
            
        # Save individual models (RF and KNN) used in the ensemble for this baseline
        rf_model_path = os.path.join(models_dir, f"{nutrient}_eagleeyes_rf_model.pkl")
        knn_model_path = os.path.join(models_dir, f"{nutrient}_eagleeyes_knn_model.pkl")
        with open(rf_model_path, "wb") as f:
            pickle.dump(rf_model, f)
        with open(knn_model_path, "wb") as f:
            pickle.dump(knn_model, f)
        print(f"EagleEyes RF model for {nutrient} saved to {rf_model_path}")
        print(f"EagleEyes KNN model for {nutrient} saved to {knn_model_path}")
        print("-" * 30)

    # Calculate and print SQI for the test set using ensemble predictions
    p_pred = results['P']['Ensemble']['y_pred']
    k_pred = results['K']['Ensemble']['y_pred']
    mg_pred = results['Mg']['Ensemble']['y_pred']
    ph_pred = results['pH']['Ensemble']['y_pred']
    
    sqi_pred = calculate_sqi(p_pred, k_pred, mg_pred, ph_pred)
    print("\n--- Soil Quality Index (SQI) ---")
    print(f"Predicted SQI for the test set (first 5 samples):\n{sqi_pred[:5]}")

    # Save results for visualization
    with open("results_eagleeyes.pkl", "wb") as f: # Save with diff indicator
        pickle.dump(results, f)

if __name__ == "__main__":
    main()