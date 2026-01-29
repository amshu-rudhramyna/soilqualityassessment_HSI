import numpy as np
import pandas as pd
import os

def generate_mock_pk_mg_ph_labels(output_csv_path="data/HYPERVIEW2/train_gt_pk_mg_ph_mock.csv", num_samples=None):
    """
    Generates mock P, K, Mg, pH labels for a given number of samples.
    
    Args:
        output_csv_path (str): The path to save the generated CSV file.
        num_samples (int): The number of samples for which to generate labels.
                           If None, it tries to determine from existing train_gt.csv.
    """
    if not os.path.exists(os.path.dirname(output_csv_path)):
        os.makedirs(os.path.dirname(output_csv_path))

    if num_samples is None:
        # Try to infer num_samples from train_gt.csv if it exists
        original_train_gt_path = "data/HYPERVIEW2/train_gt.csv"
        if os.path.exists(original_train_gt_path):
            original_gt = pd.read_csv(original_train_gt_path)
            num_samples = len(original_gt)
        else:
            print("Warning: num_samples not provided and original train_gt.csv not found. Using a default of 1800 samples.")
            num_samples = 1800 # Default if no other info

    if num_samples == 0:
        print("Error: No samples to generate labels for.")
        return

    # Generate synthetic nutrient levels (P, K, Mg, pH)
    # These ranges are chosen to be plausible but distinct for each nutrient
    p_levels = np.random.uniform(5, 50, num_samples)    # Example range for Phosphorus
    k_levels = np.random.uniform(100, 600, num_samples) # Example range for Potassium
    mg_levels = np.random.uniform(20, 250, num_samples) # Example range for Magnesium
    ph_levels = np.random.uniform(5.5, 8.5, num_samples) # Example range for pH

    # Create a DataFrame
    mock_labels_df = pd.DataFrame({
        "sample_index": np.arange(num_samples),
        "P": p_levels,
        "K": k_levels,
        "Mg": mg_levels,
        "pH": ph_levels
    })

    # Save to CSV
    mock_labels_df.to_csv(output_csv_path, index=False)
    print(f"Mock P, K, Mg, pH labels generated for {num_samples} samples and saved to {output_csv_path}")

if __name__ == "__main__":
    generate_mock_pk_mg_ph_labels()
