import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from scipy.stats import ttest_ind, bartlett

def load_results(json_file):
    """Load results from a JSON file."""
    with open(json_file, "r") as file:
        return json.load(file)


def load_config(yaml_file):
    """Load configuration from a YAML file."""
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def plot_heatmap(data, title, output_path, file_name):
    """Generate and save a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data, annot=True, fmt=".4f", cmap="coolwarm", cbar=True, square=True
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Test Dataset", fontsize=14)
    plt.ylabel("Train Dataset", fontsize=14)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, file_name), dpi=300)
    plt.close()

def generate_heatmaps(results, mode, model, metric, output_dir):
    """Generate heatmaps for a specific mode, model, and metric."""
    data = {}
    print(f"Processing {model} for {mode} mode, metric: {metric}")

    for key, values in results.items():
        # Split the string key to extract components
        if key.startswith(model) and key.endswith(mode):
            parts = key.split("_")
            train_key = parts[1] + "_" + parts[2]  # Full train key
            test_key = parts[4] + "_" + parts[5]  # Full test key
            metric_value = values.get(metric, None)

            # Add the metric value to the data dictionary
            if train_key not in data:
                data[train_key] = {}
            data[train_key][test_key] = metric_value
            print(f"Key: {key}, Train: {train_key}, Test: {test_key}, Metric: {metric_value}")

    if not data:
        print(f"No data found for {mode} - {model} - {metric}. Skipping heatmap.")
        return

    # Convert to matrix form
    train_keys = sorted(data.keys())  # Ensure consistent row keys
    test_keys = sorted(set(key for train_data in data.values() for key in train_data))  # Ensure consistent column keys
    print(f"Train Keys: {train_keys}")
    print(f"Test Keys: {test_keys}")

    heatmap_matrix = np.full((len(train_keys), len(test_keys)), np.nan)
    for i, train_key in enumerate(train_keys):
        for j, test_key in enumerate(test_keys):
            heatmap_matrix[i, j] = data.get(train_key, {}).get(test_key, np.nan)
            print(f"Matrix[{i}][{j}] = {heatmap_matrix[i, j]}")

    if np.isnan(heatmap_matrix).all():
        print(f"No valid data for heatmap for {mode} - {model} - {metric}. Skipping.")
        return

    print("Heatmap Matrix:")
    print(heatmap_matrix)

    # Plot and save the heatmap
    title = f"Heatmap of {metric} for {model.capitalize()} ({mode.capitalize()} Mode)"
    file_name = f"heatmap_{model}_{metric}_{mode}.png"
    plot_heatmap(heatmap_matrix, title, output_dir, file_name)

def perform_statistical_tests(results, output_dir):
    """
    Perform statistical tests and generate heatmaps for 1NN and Random Forest models for 'true' and 'realization'.
    """
    indices = ["high_density", "low_density", "random_sample"]
    models = ["1nn", "random_forest"]
    modes = ["true", "realization"]
    delta_z_data = {}

    # Parse results and populate delta_z_data
    for key, value in results.items():
        try:
            parts = key.split("_")
            model = parts[0]  # e.g., "random_forest" or "1nn"
            mode = parts[-1]  # "true" or "realization"
            train_key = parts[1] + "_" + parts[2]  # Full train key
            test_key = parts[4] + "_" + parts[5]  # Full test key
            # Validate model, train, test, and mode
            if model not in models or train_key not in indices or test_key not in indices or mode not in modes:
                print(f"Skipping key '{key}': Unrecognized model, train/test dataset, or mode.")
                continue

            # Save delta_z data for statistical tests
            delta_z_data[f"{model}_{train_key}_train_{test_key}_test_{mode}"] = np.array(value["Delta_z"])

        except Exception as e:
            print(f"Error processing key '{key}': {e}")
            continue

    # Initialize matrices
    matrices = {
        model: {
            mode: {
                "student_t_stat": pd.DataFrame(index=indices, columns=indices),
                "student_p_value": pd.DataFrame(index=indices, columns=indices),
                "welch_t_stat": pd.DataFrame(index=indices, columns=indices),
                "welch_p_value": pd.DataFrame(index=indices, columns=indices),
                "variance_diff": pd.DataFrame(index=indices, columns=indices),
                "bartlett_stat": pd.DataFrame(index=indices, columns=indices),
                "bartlett_p": pd.DataFrame(index=indices, columns=indices),
            }
            for mode in modes
        }
        for model in models
    }

    # Perform tests
    for model in models:
        for mode in modes:
            for train1 in indices:
                for train2 in indices:
                    key1 = f"{model}_{train1}_train_{train1}_test_{mode}"
                    key2 = f"{model}_{train2}_train_{train2}_test_{mode}"

                    if key1 in delta_z_data and key2 in delta_z_data:
                        delta_z1 = delta_z_data[key1]
                        delta_z2 = delta_z_data[key2]

                        # Variance difference
                        matrices[model][mode]["variance_diff"].loc[train1, train2] = np.var(delta_z1) - np.var(delta_z2)

                        # Student's t-test (assumes equal variances)
                        student_t_stat, student_p_value = ttest_ind(delta_z1, delta_z2, equal_var=True)
                        matrices[model][mode]["student_t_stat"].loc[train1, train2] = student_t_stat
                        matrices[model][mode]["student_p_value"].loc[train1, train2] = student_p_value

                        # Welch's t-test (does not assume equal variances)
                        welch_t_stat, welch_p_value = ttest_ind(delta_z1, delta_z2, equal_var=False)
                        matrices[model][mode]["welch_t_stat"].loc[train1, train2] = welch_t_stat
                        matrices[model][mode]["welch_p_value"].loc[train1, train2] = welch_p_value

                        # Bartlett's test for equal variances
                        bartlett_stat, bartlett_p = bartlett(delta_z1, delta_z2)
                        matrices[model][mode]["bartlett_stat"].loc[train1, train2] = bartlett_stat
                        matrices[model][mode]["bartlett_p"].loc[train1, train2] = bartlett_p

    # Generate and save heatmaps
    for model in models:
        for mode in modes:
            for metric, matrix in matrices[model][mode].items():
                # Ensure the matrix is valid and has no missing data
                if matrix.isnull().values.all():
                    print(f"No valid data for {model} - {mode} - {metric}. Skipping heatmap.")
                    continue

                matrix.fillna(0, inplace=True)  # Replace NaN with 0
                title = f"{model.capitalize()} - {metric.replace('_', ' ').title()} ({mode.capitalize()})"
                file_name = f"{model}_{metric}_{mode}_heatmap.png"

                # Convert DataFrame to a NumPy array for heatmap plotting
                plot_heatmap(matrix.values, title, output_dir, file_name)

                # Save matrix to CSV
                csv_file = os.path.join(output_dir, f"{model}_{metric}_{mode}_matrix.csv")
                matrix.to_csv(csv_file)
                print(f"Saved {metric} matrix for {model} ({mode}) to {csv_file}.")

def main():
    # Load configuration
    config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micat1_schnell.yml"
    config = load_config(config_path)

    # Extract paths from configuration
    json_file = config["input"]["json_file"]
    mode_dirs = {
        "true": config["output"]["true"]["heatmap_directory"],
        "realization": config["output"]["realization"]["heatmap_directory"],
    }

    # Load results from JSON
    results = load_results(json_file)["results"]

    # Define modes, models, and metrics
    modes = ["true", "realization"]
    models = ["random_forest", "1nn"]
    metrics = ["MSE", "R2", "Variance", "Skewness", "Kurtosis"]

    # Generate heatmaps for performance metrics
    for mode in modes:
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}  # Filter results by mode
        output_dir = mode_dirs[mode]
        for model in models:
            for metric in metrics:
                generate_heatmaps(mode_results, mode, model, metric, output_dir)

    # Perform statistical tests and save heatmaps
    for mode in modes:
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}  # Filter results by mode
        output_dir = mode_dirs[mode]
        statistical_matrices = perform_statistical_tests(mode_results, output_dir)

        if not statistical_matrices:
            print(f"No statistical matrices generated for {mode} mode. Skipping.")
            continue

        for model, matrices in statistical_matrices.items():
            for metric, matrix in matrices.items():
                if matrix.empty:
                    print(f"No valid data for {model} - {mode} - {metric}. Skipping.")
                    continue

                file_name = f"{model}_{metric}_heatmap_{mode}.png"
                title = f"Statistical Test: {metric.replace('_', ' ').title()} for {model.capitalize()} ({mode.capitalize()})"
                plot_heatmap(matrix.values, title, output_dir, file_name)

                # Save matrix to CSV
                csv_file_path = os.path.join(output_dir, f"{model}_{metric}_matrix_{mode}.csv")
                matrix.to_csv(csv_file_path)
                print(f"Saved {metric} matrix for {model} ({mode}) to {csv_file_path}")
if __name__ == "__main__":
    main()