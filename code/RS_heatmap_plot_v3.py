import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, bartlett
import json

# Use Agg backend for non-interactive plotting
from matplotlib import use
use("Agg")

def read_yaml_config(file_path):
    """Reads the YAML configuration file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_json_results(json_path):
    """Loads the 'results' section of a JSON file and normalizes keys."""
    with open(json_path, 'r') as file:
        data = json.load(file)
    if "results" not in data:
        raise ValueError("Invalid JSON file: Missing 'results' key.")
    return data["results"]

def plot_heatmap(matrix, title, output_path, cmap="coolwarm", fmt=".4g", vmin=None, vmax=None, annot=True):
    """Plots a heatmap of the given matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=annot, cmap=cmap, fmt=fmt, cbar_kws={'label': 'Value'}, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=14, weight='bold')
    plt.xlabel("Test Dataset", fontsize=12)
    plt.ylabel("Train Dataset", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_json(data, output_path):
    """Saves data to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)

from scipy.stats import skew, kurtosis

def create_performance_matrices(results):
    """
    Creates performance matrices (MSE, R^2, Mean Bias, Variance, Skewness, Kurtosis) 
    for 1NN and Random Forest models.
    """
    indices = ["high_density", "low_density", "random_sample"]
    models = ["1nn", "random_forest"]
    matrices = {
        model: {
            "mse": pd.DataFrame(index=indices, columns=indices),
            "r2": pd.DataFrame(index=indices, columns=indices),
            "mean_bias": pd.DataFrame(index=indices, columns=indices),
            "variance": pd.DataFrame(index=indices, columns=indices),
            "skewness": pd.DataFrame(index=indices, columns=indices),
            "kurtosis": pd.DataFrame(index=indices, columns=indices),
        }
        for model in models
    }

    # Iterate through results and populate matrices
    for key, value in results.items():
        try:
            # Match the model name (random_forest or 1nn) and split the key accordingly
            if key.startswith("random_forest"):
                model = "random_forest"
                rest = key[len("random_forest") + 1:]  # Remove "random_forest_"
            elif key.startswith("1nn"):
                model = "1nn"
                rest = key[len("1nn") + 1:]  # Remove "1nn_"
            else:
                print(f"Skipping key '{key}': Unknown model.")
                continue

            # Extract train and test datasets
            parts = rest.split("_train_")
            if len(parts) != 2:
                print(f"Skipping key '{key}': Unexpected format.")
                continue

            train = parts[0].lower()
            test = parts[1].replace("_test", "").lower()

            if train not in indices or test not in indices:
                print(f"Skipping key '{key}': Train or test dataset not recognized.")
                continue

            # Extract metrics from the value
            mse = value.get("MSE")
            r2 = value.get("R2")
            delta_z = value.get("Delta_z")

            if mse is None or r2 is None or delta_z is None:
                print(f"Skipping key '{key}': Missing required metrics (MSE, R2, Delta_z).")
                continue

            # Calculate skewness and kurtosis
            delta_z_array = np.array(delta_z)
            skewness = skew(delta_z_array)
            kurt = kurtosis(delta_z_array)

            # Populate the matrices for the corresponding model
            matrices[model]["mse"].loc[train, test] = mse
            matrices[model]["r2"].loc[train, test] = r2
            matrices[model]["mean_bias"].loc[train, test] = np.mean(delta_z)
            matrices[model]["variance"].loc[train, test] = np.var(delta_z)
            matrices[model]["skewness"].loc[train, test] = skewness
            matrices[model]["kurtosis"].loc[train, test] = kurt

        except Exception as e:
            print(f"Error processing key '{key}': {e}")
            continue

    # Fill NaN with zeros and cast to float
    for model in models:
        for metric in matrices[model]:
            matrices[model][metric] = matrices[model][metric].astype(float).fillna(0)

    return matrices

def perform_statistical_tests(results):
    """
    Performs statistical tests (Student's t-test, Welch's t-test, Bartlett's test) and 
    generates matrices for 1NN and Random Forest models.
    """
    indices = ["high_density", "low_density", "random_sample"]
    models = ["1nn", "random_forest"]
    delta_z_data = {}

    # Parse results and populate delta_z_data
    for key, value in results.items():
        try:
            # Match the model name and split the key
            if key.startswith("random_forest"):
                model = "random_forest"
                rest = key[len("random_forest") + 1:]
            elif key.startswith("1nn"):
                model = "1nn"
                rest = key[len("1nn") + 1:]
            else:
                print(f"Skipping key '{key}': Unknown model.")
                continue

            parts = rest.split("_train_")
            if len(parts) != 2:
                print(f"Skipping key '{key}': Unexpected format.")
                continue

            train = parts[0].lower()
            test = parts[1].replace("_test", "").lower()

            if train not in indices or test not in indices:
                print(f"Skipping key '{key}': Train or test dataset not recognized.")
                continue

            # Save delta_z data for statistical tests
            delta_z_data[f"{model}_{train}_train_{test}_test"] = np.array(value["Delta_z"])

        except Exception as e:
            print(f"Error processing key '{key}': {e}")
            continue

    # Initialize matrices
    matrices = {
        model: {
            "student_t_stat": pd.DataFrame(index=indices, columns=indices),
            "student_p_value": pd.DataFrame(index=indices, columns=indices),
            "welch_t_stat": pd.DataFrame(index=indices, columns=indices),
            "welch_p_value": pd.DataFrame(index=indices, columns=indices),
            "variance_diff": pd.DataFrame(index=indices, columns=indices),
            "bartlett_stat": pd.DataFrame(index=indices, columns=indices),
            "bartlett_p": pd.DataFrame(index=indices, columns=indices),
        }
        for model in models
    }

    # Perform tests
    for model in models:
        for train1 in indices:
            for train2 in indices:
                key1 = f"{model}_{train1}_train_{train1}_test"
                key2 = f"{model}_{train2}_train_{train2}_test"

                if key1 in delta_z_data and key2 in delta_z_data:
                    delta_z1 = delta_z_data[key1]
                    delta_z2 = delta_z_data[key2]

                    # Variance difference
                    matrices[model]["variance_diff"].loc[train1, train2] = np.var(delta_z1) - np.var(delta_z2)

                    # Student's t-test (assumes equal variances)
                    student_t_stat, student_p_value = ttest_ind(delta_z1, delta_z2, equal_var=True)
                    matrices[model]["student_t_stat"].loc[train1, train2] = student_t_stat
                    matrices[model]["student_p_value"].loc[train1, train2] = student_p_value

                    # Welch's t-test (does not assume equal variances)
                    welch_t_stat, welch_p_value = ttest_ind(delta_z1, delta_z2, equal_var=False)
                    matrices[model]["welch_t_stat"].loc[train1, train2] = welch_t_stat
                    matrices[model]["welch_p_value"].loc[train1, train2] = welch_p_value

                    # Bartlett's test for equal variances
                    bartlett_stat, bartlett_p = bartlett(delta_z1, delta_z2)
                    matrices[model]["bartlett_stat"].loc[train1, train2] = bartlett_stat
                    matrices[model]["bartlett_p"].loc[train1, train2] = bartlett_p

    # Fill NaN with zeros and cast to float
    for model in models:
        for metric in matrices[model]:
            matrices[model][metric] = matrices[model][metric].astype(float).fillna(0)

    return matrices

def main(config_path):
    """Main function to process data and generate heatmaps and statistical results."""
    config = read_yaml_config(config_path)
    json_path = config["input"]["json_file"]
    output_dir = config["output"]["directory"]
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_json_results(json_path)

    # Create performance matrices
    performance_matrices = create_performance_matrices(results)

    # Plot performance metrics
    for model in performance_matrices:
        for metric, matrix in performance_matrices[model].items():
            title = f"{model.upper()} {metric.replace('_', ' ').title()} Heatmap"
            output_path = os.path.join(output_dir, f"{model}_{metric}_heatmap.png")
            plot_heatmap(matrix, title, output_path)

    # Perform statistical tests
    statistical_results = perform_statistical_tests(results)

    # Plot statistical results
    for model in statistical_results:
        for metric, matrix in statistical_results[model].items():
            title = f"{model.upper()} {metric.replace('_', ' ').title()} Heatmap"
            output_path = os.path.join(output_dir, f"{model}_{metric}_heatmap.png")
            plot_heatmap(matrix, title, output_path)

    # Save statistical results
    for model in statistical_results:
        for metric, matrix in statistical_results[model].items():
            output_path = os.path.join(output_dir, f"{model}_{metric}_results.json")
            save_json(matrix.to_dict(), output_path)

if __name__ == "__main__":
    main("/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_plot_LSST_With_Noise.yml")