import numpy as np
import os

def generate_mock_data(output_path="dataset/HYPERVIEW2/mock_data.npz"):
    """
    Generates synthetic hyperspectral data and saves it to a .npz file.

    The mock data consists of hyperspectral cubes and corresponding nutrient levels.

    Args:
        output_path (str): The path to save the generated .npz file.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Generate synthetic hyperspectral data
    num_samples = 200
    num_bands = 150
    height = 11
    width = 11
    
    # Create random hyperspectral cubes
    hyperspectral_cubes = np.random.rand(num_samples, num_bands, height, width).astype(np.float32)

    # Generate synthetic nutrient levels (P, K, Mg, pH)
    p_levels = np.random.rand(num_samples) * 100
    k_levels = np.random.rand(num_samples) * 500
    mg_levels = np.random.rand(num_samples) * 200
    ph_levels = np.random.uniform(4, 8, num_samples)

    # Save the data to a .npz file
    np.savez(output_path, 
             hyperspectral_cubes=hyperspectral_cubes, 
             P=p_levels, 
             K=k_levels, 
             Mg=mg_levels, 
             pH=ph_levels)

    print(f"Mock data generated and saved to {output_path}")

if __name__ == "__main__":
    generate_mock_data()
