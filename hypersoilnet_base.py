import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import xgboost as xgb
import pickle
import os

# --- Helper function to load all hyperspectral data and labels ---
def load_hyperview2_data(data_dir="data/HYPERVIEW2"):
    """
    Loads hyperspectral data (.npz files) and corresponding mock nutrient labels (.csv).

    Args:
        data_dir (str): Base directory for HYPERVIEW2 data.

    Returns:
        tuple: (hyperspectral_spectra_array, labels_df)
            hyperspectral_spectra_array (np.ndarray): Stacked numpy array of spatially averaged spectra.
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

def calculate_svd_features(spectra, n_components=10):
    """
    Calculates Singular Value Decomposition (SVD) features.
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_bands).
        n_components (int): Number of principal components to retain.
    Returns:
        np.ndarray: SVD features (transformed data).
    """
    # Ensure n_components is not greater than min(num_samples, num_bands)
    n_components = min(n_components, spectra.shape[0], spectra.shape[1])
    
    # Center the data
    centered_spectra = spectra - np.mean(spectra, axis=0)
    
    # Perform SVD
    U, s, Vt = np.linalg.svd(centered_spectra, full_matrices=False)
    
    # Return the first n_components of U (principal components)
    return U[:, :n_components]

def calculate_fft_features(spectra, n_components=10):
    """
    Calculates Fast Fourier Transform (FFT) features.
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_bands).
        n_components (int): Number of frequency components to retain (magnitude).
    Returns:
        np.ndarray: FFT features.
    """
    fft_features_list = []
    for i in range(spectra.shape[0]):
        # Perform FFT on each spectrum
        fft_result = np.fft.fft(spectra[i, :])
        # Take the magnitude of the first n_components (excluding DC component, but including positive frequencies)
        # We take n_components from the magnitude of the FFT result
        # Ensure n_components does not exceed half the number of bands (Nyquist limit)
        actual_n_components = min(n_components, len(spectra[i,:]) // 2)
        fft_magnitude = np.abs(fft_result[1:actual_n_components + 1]) # Exclude DC component (index 0)
        fft_features_list.append(fft_magnitude)
    return np.array(fft_features_list)

# Define constants for band indices for feature calculations
# These will now be relative to the trimmed 150-band spectrum

# Wavelengths for trimming
BAND_TRIM_START_WAVELENGTH = 462.0  # nm
BAND_TRIM_END_WAVELENGTH = 938.0    # nm

# Wavelengths for Red and NIR for SAVI/RVI (these should ideally be within the trimmed range)
RED_WAVELENGTH_SAVI = 669.85 # nm (originally Band 80)
NIR_WAVELENGTH_SAVI = 855.26 # nm (originally Band 138)

def get_hsi_aerial_wavelengths():
    """Loads hsi_aerial_wavelengths from wavelengths.json."""
    wavelengths_path = os.path.join("data", "HYPERVIEW2", "wavelengths.json")
    if not os.path.exists(wavelengths_path):
        raise FileNotFoundError(f"wavelengths.json not found at {wavelengths_path}")
    with open(wavelengths_path, 'r') as f:
        all_wavelengths = json.load(f)
    return {int(k.replace('Band ', '')): v for k, v in all_wavelengths["hsi_aerial_wavelengths"].items()}


def trim_bands(spectra, wavelengths_map, start_wl, end_wl):
    """
    Trims the hyperspectral spectra to a specified wavelength range.
    Args:
        spectra (np.ndarray): 2D array of spectra (num_samples, num_original_bands).
        wavelengths_map (dict): Dictionary mapping band index to wavelength.
        start_wl (float): Starting wavelength for trimming.
        end_wl (float): Ending wavelength for trimming.
    Returns:
        np.ndarray: Trimmed spectra (num_samples, num_trimmed_bands).
        list: List of wavelengths for the trimmed spectra.
        tuple: (start_idx, end_idx) of the trimmed bands in the original spectra.
    """
    original_wavelengths = np.array(list(wavelengths_map.values()))
    original_indices = np.array(list(wavelengths_map.keys()))

    # Find indices within the desired wavelength range
    trimmed_indices = np.where((original_wavelengths >= start_wl) & (original_wavelengths <= end_wl))[0]
    
    if len(trimmed_indices) == 0:
        raise ValueError(f"No bands found within the specified wavelength range ({start_wl}-{end_wl} nm).")

    # Get the actual indices in the original data based on sorted keys
    all_original_sorted_keys = sorted(wavelengths_map.keys())
    start_band_idx_orig = all_original_sorted_keys[trimmed_indices[0]]
    end_band_idx_orig = all_original_sorted_keys[trimmed_indices[-1]]

    # Trim the spectra
    trimmed_spectra = spectra[:, trimmed_indices]
    trimmed_wavelengths = original_wavelengths[trimmed_indices]

    return trimmed_spectra, trimmed_wavelengths, (start_band_idx_orig, end_band_idx_orig)


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
    # Load wavelengths map
    wavelengths_map = get_hsi_aerial_wavelengths()

    # Trim bands to the specified range (462-938 nm)
    spectra, trimmed_wavelengths, _ = trim_bands(hyperspectral_spectra, wavelengths_map, BAND_TRIM_START_WAVELENGTH, BAND_TRIM_END_WAVELENGTH)
    
    # Dynamically determine RED_BAND_IDX and NIR_BAND_IDX based on trimmed wavelengths
    red_band_idx_trimmed = np.abs(trimmed_wavelengths - RED_WAVELENGTH_SAVI).argmin()
    nir_band_idx_trimmed = np.abs(trimmed_wavelengths - NIR_WAVELENGTH_SAVI).argmin()

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
    savi_feature = calculate_savi(spectra, red_band_idx_trimmed, nir_band_idx_trimmed)
    rvi_feature = calculate_rvi(spectra, red_band_idx_trimmed, nir_band_idx_trimmed)

    # Combine all features
    all_features = [spectra, derivative_features, savi_feature, rvi_feature]

    # SVD Features
    svd_features = calculate_svd_features(spectra, n_components=10) # Example: retain 10 components
    if svd_features.size > 0:
        all_features.append(svd_features)

    # FFT Features
    fft_features = calculate_fft_features(spectra, n_components=10) # Example: retain 10 components
    if fft_features.size > 0:
        all_features.append(fft_features)

    # Concatenate all collected features
    X_features = np.concatenate(all_features, axis=1)

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

def train_and_evaluate_pls(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains a PLS Regressor and evaluates its performance.
    """
    model = PLSRegression(n_components=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # PLSRegression predict returns (n_samples, n_targets) even for single target, so flatten
    r2 = r2_score(y_test, y_pred.flatten()) 
    print(f"PLSRegression R2 score for {nutrient_name}: {r2:.4f}")
    return model, y_pred.flatten(), r2 # Return y_pred for consistency

def train_and_evaluate_xgb(X_train, X_test, y_train, y_test, nutrient_name):
    """
    Trains an XGBoost Regressor and evaluates its performance.
    """
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost R2 score for {nutrient_name}: {r2:.4f}")
    return model, y_pred, r2 # Return y_pred for consistency

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
    Main function to run the HyperSoilNet Phase 1 model training and evaluation.
    """
    print("Loading HYPERVIEW2 data and mock labels for HyperSoilNet Phase 1...")
    hyperspectral_spectra, labels_df = load_hyperview2_data(data_dir)
    print(f"Loaded {hyperspectral_spectra.shape[0]} hyperspectral spectra and {len(labels_df)} corresponding labels.")

    X = preprocess_data(hyperspectral_spectra) 
    
    nutrients = ["P", "K", "Mg", "pH"]
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    results = {}
    for nutrient in nutrients:
        print(f"\n--- Training models for {nutrient} ---")
        y = labels_df[nutrient].values # Use mock labels

        if X.shape[0] != len(y):
            print(f"Warning: Mismatch in number of samples between features ({X.shape[0]}) and labels ({len(y)}) for {nutrient}.")
            min_samples = min(X.shape[0], len(y))
            X_aligned = X[:min_samples]
            y_aligned = y[:min_samples]
        else:
            X_aligned = X
            y_aligned = y

        X_train, X_test, y_train, y_test = train_test_split(X_aligned, y_aligned, test_size=0.2, random_state=42)

        models_to_train = {
            "Random Forest": train_and_evaluate_rf,
            "PLSRegression": train_and_evaluate_pls,
            "XGBoost": train_and_evaluate_xgb,
        }
        
        results[nutrient] = {}
        # Store y_test once per nutrient for SQI calculation later
        results[nutrient]['y_test'] = y_test 

        for model_name, train_func in models_to_train.items():
            model, y_pred, r2 = train_func(X_train, X_test, y_train, y_test, nutrient)
            results[nutrient][model_name] = {"model": model, "r2": r2, "y_pred": y_pred}
            
            model_path = os.path.join(models_dir, f"{nutrient}_hypersoilnet_base_{model_name.lower().replace(' ', '_')}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"HyperSoilNet Base {model_name} model for {nutrient} saved to {model_path}")
        print("-" * 30)

    # Calculate and print SQI for the test set using predictions from the Random Forest model as a representative
    # Or, one could choose the best R2 score model, but for simplicity, using RF here.
    p_pred = results['P']['Random Forest']['y_pred']
    k_pred = results['K']['Random Forest']['y_pred']
    mg_pred = results['Mg']['Random Forest']['y_pred']
    ph_pred = results['pH']['Random Forest']['y_pred']
    
    sqi_pred = calculate_sqi(p_pred, k_pred, mg_pred, ph_pred)
    print("\n--- Soil Quality Index (SQI) ---")
    print(f"Predicted SQI for the test set (first 5 samples):\n{sqi_pred[:5]}")

    # Save results for visualization
    with open("results_hypersoilnet_base.pkl", "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    # Ensure json module is imported for get_hsi_aerial_wavelengths
    import json
    main()


if __name__ == "__main__":
    main()
