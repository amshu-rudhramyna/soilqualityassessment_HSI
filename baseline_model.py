import numpy as np
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import pickle
import os

def preprocess_data(hyperspectral_cubes):
    """
    Preprocesses the hyperspectral data by applying Savitzky-Golay smoothing
    and averaging the spatial dimensions.

    Args:
        hyperspectral_cubes (np.ndarray): The hyperspectral data cubes.

    Returns:
        np.ndarray: The preprocessed hyperspectral data.
    """
    # Apply Savitzky-Golay filter to each spectrum
    smoothed_cubes = savgol_filter(hyperspectral_cubes, window_length=5, polyorder=2, axis=1)
    
    # Average the spatial dimensions (11x11) to get a single spectrum per sample
    preprocessed_data = np.mean(smoothed_cubes, axis=(2, 3))
    
    return preprocessed_data

def train_and_evaluate_rf(X, y, nutrient_name):
    """
    Trains a Random Forest Regressor and evaluates its performance.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The target values.
        nutrient_name (str): The name of the nutrient being modeled.

    Returns:
        RandomForestRegressor: The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest R2 score for {nutrient_name}: {r2:.4f}")
    
    return model

def train_and_evaluate_pls(X, y, nutrient_name):
    """
    Trains a PLS Regressor and evaluates its performance.

    Args:
        X (np.ndarray): The input features.
        y (np.ndarray): The target values.
        nutrient_name (str): The name of the nutrient being modeled.

    Returns:
        PLSRegression: The trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = PLSRegression(n_components=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"PLSRegression R2 score for {nutrient_name}: {r2:.4f}")
    
    return model

def main(data_path="dataset/HYPERVIEW2/mock_data.npz"):
    """
    Main function to run the baseline model training and evaluation.

    Args:
        data_path (str): The path to the .npz data file.
    """
    # Load the data
    data = np.load(data_path)
    hyperspectral_cubes = data["hyperspectral_cubes"]
    
    # Preprocess the data
    X = preprocess_data(hyperspectral_cubes)
    
    # Nutrients to model
    nutrients = ["P", "K", "Mg", "pH"]
    
    # Create a directory to save the models
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    for nutrient in nutrients:
        print(f"--- Training model for {nutrient} ---")
        y = data[nutrient]
        
        # Train and evaluate Random Forest
        rf_model = train_and_evaluate_rf(X, y, nutrient)
        model_path = os.path.join(models_dir, f"{nutrient}_rf_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(rf_model, f)
        print(f"Random Forest model for {nutrient} saved to {model_path}")

        # Train and evaluate PLS Regression
        pls_model = train_and_evaluate_pls(X, y, nutrient)
        model_path = os.path.join(models_dir, f"{nutrient}_pls_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pls_model, f)
        print(f"PLSRegression model for {nutrient} saved to {model_path}")
        
        print("-" * 30)

if __name__ == "__main__":
    main()
