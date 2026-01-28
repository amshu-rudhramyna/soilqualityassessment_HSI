import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def create_visualizations(results_path="results.pkl"):
    """
    Creates and saves visualizations from the model results.

    Args:
        results_path (str): Path to the results.pkl file.
    """
    with open(results_path, "rb") as f:
        results = pickle.load(f)

    visualizations_dir = "visualizations"
    if not os.path.exists(visualizations_dir):
        os.makedirs(visualizations_dir)

    nutrients = list(results.keys())
    models = list(results[nutrients[0]].keys())
    
    # 1. Bar chart of R2 scores
    r2_scores = {model: [results[n][model]['r2'] for n in nutrients] for model in models}
    
    x = np.arange(len(nutrients))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    
    for model_name, model_r2_scores in r2_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, model_r2_scores, width, label=model_name)
        ax.bar_label(rects, padding=3, fmt='%.2f')
        multiplier += 1

    ax.set_ylabel('R2 Score')
    ax.set_title('Model Performance Comparison (R2 Score)')
    ax.set_xticks(x + width, nutrients)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(min(min(model_r2_scores) for model_r2_scores in r2_scores.values()) - 0.2, max(max(model_r2_scores) for model_r2_scores in r2_scores.values()) + 0.2)
    
    r2_scores_path = os.path.join(visualizations_dir, "r2_scores_comparison.png")
    plt.savefig(r2_scores_path)
    print(f"R2 scores comparison chart saved to {r2_scores_path}")
    plt.close()

    # 2. Scatter plots of predicted vs. actual for the best model for each nutrient
    for nutrient in nutrients:
        best_model_name = max(results[nutrient], key=lambda model: results[nutrient][model]['r2'])
        best_model_data = results[nutrient][best_model_name]
        
        y_test = best_model_data['y_test']
        y_pred = best_model_data['y_pred']
        
        fig, ax = plt.subplots(layout='constrained')
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs. Actual for {nutrient} (Best Model: {best_model_name})')
        
        scatter_path = os.path.join(visualizations_dir, f"{nutrient}_pred_vs_actual.png")
        plt.savefig(scatter_path)
        print(f"Scatter plot for {nutrient} saved to {scatter_path}")
        plt.close()

if __name__ == "__main__":
    create_visualizations()
