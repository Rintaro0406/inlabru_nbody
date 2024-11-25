import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import json

# Use Agg backend for non-interactive plotting
from matplotlib import use
use("Agg")


def read_yaml_config(file_path):
    """
    Reads the YAML configuration file and returns its content as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_json_results(json_path):
    """
    Loads the 'results' section of a JSON file and normalizes keys for processing.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Ensure the JSON file contains the "results" key
    if "results" not in data:
        raise ValueError("Invalid JSON file: Missing 'results' key.")
    
    results = data["results"]
    
    # Normalize keys in 'results' to match the expected format in processing
    normalized_results = {}
    for key, value in results.items():
        if "_train_" in key:  # Process keys with "_train_" in their names
            normalized_results[key.lower()] = value
        else:
            print(f"Skipping unexpected or invalid key: {key}")
    
    return normalized_results



def plot_heatmap(matrix, title, output_path, cmap="coolwarm", fmt=".4g", vmin=None, vmax=None, annot=True):
    """
    Plots a heatmap of the given matrix with enhanced visualization options.

    Parameters:
        matrix (pd.DataFrame): The matrix to be visualized.
        title (str): Title of the heatmap.
        output_path (str): Path to save the heatmap image.
        cmap (str): Color map to use for the heatmap.
        fmt (str): Format for annotations (e.g., ".4g" for numbers with 4 significant digits).
        vmin (float, optional): Minimum value for color scale. Defaults to None (auto-scaling).
        vmax (float, optional): Maximum value for color scale. Defaults to None (auto-scaling).
        annot (bool): Whether to annotate the heatmap with values. Defaults to True.
    """
    plt.figure(figsize=(10, 8))
    vmin = matrix.values.min() if vmin is None else vmin
    vmax = matrix.values.max() if vmax is None else vmax
    # Create heatmap
    sns.heatmap(matrix, annot=annot, cmap=cmap, fmt=fmt, cbar_kws={'label': 'Value'}, vmin=vmin, vmax=vmax)
    
    # Add title and axis labels
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Test Dataset", fontsize=12)
    plt.ylabel("Train Dataset", fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()



def save_ttest_results(comparisons, output_path):
    """
    Saves the T-test results to a JSON file.
    """
    with open(output_path, 'w') as file:
        json.dump(comparisons, file, indent=4)

def create_performance_matrices(results):
    """
    Creates performance matrices (MSE, R^2, Mean Bias) from the results JSON.
    """
    indices = ["high_density", "low_density", "random_sample"]  # Match lowercase indices
    mse_matrix = pd.DataFrame(index=indices, columns=indices)
    r2_matrix = pd.DataFrame(index=indices, columns=indices)
    mean_bias_matrix = pd.DataFrame(index=indices, columns=indices)

    for key, value in results.items():
        if "_train_" in key and "MSE" in value and "R2" in value and "Delta_z" in value:
            try:
                train, test = key.split("_train_")
                train = train.lower()  # Ensure lowercase for alignment
                test = test.lower().replace("_test", "")  # Remove `_test` suffix

                # Ensure train and test are valid indices
                if train in indices and test in indices:
                    mse_matrix.loc[train, test] = value["MSE"]
                    r2_matrix.loc[train, test] = value["R2"]
                    mean_bias_matrix.loc[train, test] = np.mean(value["Delta_z"])
                else:
                    print(f"Skipping unexpected train/test pair: {train}, {test}")
            except Exception as e:
                print(f"Error processing key {key}: {e}")
        else:
            print(f"Skipping unexpected or invalid key: {key}")

    # Convert matrices to float and fill missing values with 0
    return (
        mse_matrix.astype(float).fillna(0),
        r2_matrix.astype(float).fillna(0),
        mean_bias_matrix.astype(float).fillna(0),
    )

def perform_ttests(results):
    """
    Performs Student's T-tests for differences in Delta_z distributions across train-test pairs.
    Returns a p-value matrix for visualization.
    """
    indices = ["high_density", "low_density", "random_sample"]
    delta_z_data = {
        key: np.array(value["Delta_z"])
        for key, value in results.items()
        if "_train_" in key and "Delta_z" in value
    }

    p_value_matrix = pd.DataFrame(index=indices, columns=indices)

    for train1 in indices:
        for train2 in indices:
            key1 = f"{train1}_train_{train1}_test"
            key2 = f"{train2}_train_{train2}_test"
            if key1 in delta_z_data and key2 in delta_z_data:
                delta_z1 = delta_z_data[key1]
                delta_z2 = delta_z_data[key2]
                _, p_value = ttest_ind(delta_z1, delta_z2, equal_var=False)  # Welch's T-test
                p_value_matrix.loc[train1, train2] = p_value

    return p_value_matrix.astype(float)


def plot_ttest_heatmap(matrix, title, output_path, cmap="coolwarm", fmt=".4g"):
    """
    Plots a heatmap for T-test p-values.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap=cmap, fmt=fmt, cbar_kws={'label': 'P-Value'})
    plt.title(title)
    plt.xlabel("Train Dataset")
    plt.ylabel("Train Dataset")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(config_path):
    """
    Main function to process data and generate performance matrices, heatmaps, and T-test results.
    """
    config = read_yaml_config(config_path)
    json_path = config["input"]["json_file"]
    output_dir = config["output"]["directory"]

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_json_results(json_path)
    for key, value in results.items():
        print(f"Key: {key}")
        print(f"MSE: {value['MSE']}, R2: {value['R2']}, Mean Delta_z: {np.mean(value['Delta_z'])}")
    # Create performance matrices
    mse_matrix, r2_matrix, mean_bias_matrix = create_performance_matrices(results)
    print("MSE Matrix:")    
    print(mse_matrix)
    print("\nR^2 Matrix:")
    print(r2_matrix)    
    print("\nMean Bias Matrix:")
    print(mean_bias_matrix)
    # Handle empty matrices gracefully
    if mse_matrix.isnull().values.any():
        print("Warning: Some entries in the performance matrices are missing.")
        mse_matrix = mse_matrix.fillna(0)
        r2_matrix = r2_matrix.fillna(0)
        mean_bias_matrix = mean_bias_matrix.fillna(0)

    # Plot heatmaps
    plot_heatmap(mse_matrix, "MSE Heatmap", os.path.join(output_dir, "mse_heatmap.png"))
    plot_heatmap(r2_matrix, "R^2 Heatmap", os.path.join(output_dir, "r2_heatmap.png"))
    plot_heatmap(mean_bias_matrix, "Mean Bias Heatmap", os.path.join(output_dir, "mean_bias_heatmap.png"))

    # Perform T-tests and visualize results
    p_value_matrix = perform_ttests(results)
    plot_ttest_heatmap(p_value_matrix, "T-Test P-Value Heatmap", os.path.join(output_dir, "ttest_pvalues_heatmap.png"))
    save_ttest_results(p_value_matrix.to_dict(), os.path.join(output_dir, "ttest_results.json"))


if __name__ == "__main__":
    main("/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_plot.yml")