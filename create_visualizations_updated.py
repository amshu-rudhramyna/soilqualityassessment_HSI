import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def create_updated_visualizations(results_paths_and_labels):
    """
    Creates and saves comparative visualizations from multiple model result files.

    Args:
        results_paths_and_labels (list): A list of tuples, where each tuple is
                                         (results_path, label_for_model_set).
                                         Example: [("results_eagleeyes.pkl", "EagleEyes"),
                                                   ("results_hypersoilnet_base.pkl", "HyperSoilNet-Phase1")]
    """
    all_results_data = {}
    
    # Load all results files
    for results_path, label in results_paths_and_labels:
        if not os.path.exists(results_path):
            print(f"Warning: Results file not found at {results_path}. Skipping {label}.")
            continue
        with open(results_path, "rb") as f:
            all_results_data[label] = pickle.load(f)

    if not all_results_data:
        print("No results data loaded. Cannot create visualizations.")
        return

    visualizations_dir = "visualizations_updated" # New directory for updated visualizations
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    nutrients = ["P", "K", "Mg", "pH"] # Assuming these nutrients are consistent

    # --- 1. Comparative Bar chart of R2 scores ---
    r2_scores_for_plot = {} # Structure: {nutrient: {model_name_with_label: r2_score}}
    
    # Collect R2 scores from all loaded result sets
    for nutrient in nutrients:
        r2_scores_for_plot[nutrient] = {}
        for label, results_dict in all_results_data.items():
            if nutrient in results_dict:
                # For EagleEyes, we want the Ensemble R2
                if label == "EagleEyes" and "Ensemble" in results_dict[nutrient]:
                    r2_scores_for_plot[nutrient][f"{label} (Ensemble)"] = results_dict[nutrient]["Ensemble"]['r2']
                # For HyperSoilNet-Phase1, we want R2 for each individual model
                elif label == "HyperSoilNet-Phase1" and "Random Forest" in results_dict[nutrient]:
                    r2_scores_for_plot[nutrient][f"{label} (RF)"] = results_dict[nutrient]["Random Forest"]['r2']
                    r2_scores_for_plot[nutrient][f"{label} (PLS)"] = results_dict[nutrient]["PLSRegression"]['r2']
                    r2_scores_for_plot[nutrient][f"{label} (XGB)"] = results_dict[nutrient]["XGBoost"]['r2']
            
    if not any(r2_scores_for_plot[n] for n in nutrients):
        print("No R2 scores collected for comparison chart.")
        return

    # Prepare data for bar chart
    all_model_labels = sorted(list(set(model_label for nutrient_data in r2_scores_for_plot.values() for model_label in nutrient_data.keys())))
    num_models = len(all_model_labels)
    x = np.arange(len(nutrients))
    width = 0.8 / num_models # Adjust bar width based on number of models
    
    fig, ax = plt.subplots(figsize=(12, 7), layout='constrained')
    
    for i, model_label in enumerate(all_model_labels):
        model_r2_scores = [r2_scores_for_plot[n].get(model_label, 0) for n in nutrients] # Get score or 0 if not present
        offset = (i - num_models / 2 + 0.5) * width
        rects = ax.bar(x + offset, model_r2_scores, width, label=model_label)
        ax.bar_label(rects, padding=3, fmt='%.2f')

    ax.set_ylabel('R2 Score')
    ax.set_title('Comparative Model Performance (R2 Score) for EagleEyes vs. HyperSoilNet-Phase1')
    ax.set_xticks(x, nutrients)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=num_models) # Adjust legend position
    ax.set_ylim(min(min(scores.values()) for scores in r2_scores_for_plot.values()) - 0.2, 
                max(max(scores.values()) for scores in r2_scores_for_plot.values()) + 0.2)
    ax.axhline(0, color='grey', linewidth=0.8) # Add a line at R2=0 for reference
    
    r2_scores_comparison_path = os.path.join(visualizations_dir, "r2_scores_comparison_updated.png")
    plt.savefig(r2_scores_comparison_path)
    print(f"Updated R2 scores comparison chart saved to {r2_scores_comparison_path}")
    plt.close()

    # --- 2. Scatter plots of predicted vs. actual for representative models ---
    # For EagleEyes, use the Ensemble. For HyperSoilNet-Phase1, use the best individual model (e.g., RF)
    
    for nutrient in nutrients:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained') # Two subplots for comparison
        fig.suptitle(f'Predicted vs. Actual for {nutrient}', fontsize=16)

        # Plot for EagleEyes (Ensemble)
        if "EagleEyes" in all_results_data and nutrient in all_results_data["EagleEyes"] and "Ensemble" in all_results_data["EagleEyes"][nutrient]:
            ee_data = all_results_data["EagleEyes"][nutrient]["Ensemble"]
            y_test_ee = ee_data['y_test']
            y_pred_ee = ee_data['y_pred']
            
            axes[0].scatter(y_test_ee, y_pred_ee, alpha=0.5)
            axes[0].plot([min(y_test_ee), max(y_test_ee)], [min(y_test_ee), max(y_test_ee)], 'r--')
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'EagleEyes (Ensemble) R2: {ee_data["r2"]:.2f}')
        else:
            axes[0].set_title('EagleEyes (Ensemble) - No Data')
            axes[0].set_visible(False) # Hide subplot if no data


        # Plot for HyperSoilNet-Phase1 (e.g., Random Forest as representative)
        if "HyperSoilNet-Phase1" in all_results_data and nutrient in all_results_data["HyperSoilNet-Phase1"] and "Random Forest" in all_results_data["HyperSoilNet-Phase1"][nutrient]:
            hsn_data = all_results_data["HyperSoilNet-Phase1"][nutrient]["Random Forest"]
            y_test_hsn = all_results_data["HyperSoilNet-Phase1"][nutrient]['y_test'] # y_test is stored at nutrient level
            y_pred_hsn = hsn_data['y_pred']
            
            axes[1].scatter(y_test_hsn, y_pred_hsn, alpha=0.5)
            axes[1].plot([min(y_test_hsn), max(y_test_hsn)], [min(y_test_hsn), max(y_test_hsn)], 'r--')
            axes[1].set_xlabel('Actual Values')
            axes[1].set_ylabel('Predicted Values')
            axes[1].set_title(f'HyperSoilNet-Phase1 (RF) R2: {hsn_data["r2"]:.2f}')
        else:
            axes[1].set_title('HyperSoilNet-Phase1 (RF) - No Data')
            axes[1].set_visible(False) # Hide subplot if no data

        scatter_path_comp = os.path.join(visualizations_dir, f"{nutrient}_pred_vs_actual_comparison.png")
        plt.savefig(scatter_path_comp)
        print(f"Comparative scatter plot for {nutrient} saved to {scatter_path_comp}")
        plt.close()


if __name__ == "__main__":
    results_files = [
        ("results_eagleeyes.pkl", "EagleEyes"),
        ("results_hypersoilnet_base.pkl", "HyperSoilNet-Phase1")
    ]
    create_updated_visualizations(results_files)