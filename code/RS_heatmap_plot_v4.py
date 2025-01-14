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


def plot_heatmap(data, title, output_path, file_name, x_labels=None, y_labels=None):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    data = np.nan_to_num(data.astype(float))  # Ensure numeric

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        xticklabels=x_labels,
        yticklabels=y_labels
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Test Dataset", fontsize=14)
    plt.ylabel("Train Dataset", fontsize=14)
    plt.xticks(rotation=45, ha='right')  # Rotate x-labels if desired
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, file_name), dpi=300)
    plt.close()

def extract_train_test_keys(key, model):
    """
    Extract train_key and test_key based on the model name.
    """
    parts = key.split("_")
    model_parts = model.split("_")
    model_len = len(model_parts)

    # For 1nn:
    # parts: [0:"1nn", 1:"high", 2:"density", 3:"train", 4:"low", 5:"density", 6:"test", 7:"true"]
    # train_key = parts[1]+"_"+parts[2]
    # test_key = parts[4]+"_"+parts[5]

    # For random_forest:
    # parts: [0:"random", 1:"forest", 2:"high", 3:"density", 4:"train", 5:"high", 6:"density", 7:"test", 8:"true"]
    # train_key = parts[2]+"_"+parts[3]
    # test_key = parts[5]+"_"+parts[6]

    train_key = parts[model_len] + "_" + parts[model_len+1]
    test_key = parts[model_len+3] + "_" + parts[model_len+4]

    return train_key, test_key

def generate_heatmaps(results, mode, model, metric, output_dir):
    """Generate heatmaps for a specific mode, model, and metric."""
    data = {}
    print(f"Processing {model} for {mode} mode, metric: {metric}")

    for key, values in results.items():
        # Check if key matches model and mode
        if key.startswith(model) and key.endswith(mode):
            train_key, test_key = extract_train_test_keys(key, model)
            metric_value = values.get(metric, None)

            # Add the metric value to the data dictionary
            if train_key not in data:
                data[train_key] = {}
            data[train_key][test_key] = metric_value
            print(f"Key: {key}, Train: {train_key}, Test: {test_key}, Metric: {metric_value}")

    if not data:
        print(f"No data found for {mode} - {model} - {metric}. Skipping heatmap.")
        return

    # Extract all keys used for train and test
    all_train_keys = set(data.keys())
    all_test_keys = set()
    for t_data in data.values():
        all_test_keys.update(t_data.keys())

    train_keys = sorted(all_train_keys)
    test_keys = sorted(all_test_keys)

    print(f"Train Keys: {train_keys}")
    print(f"Test Keys: {test_keys}")

    heatmap_matrix = np.full((len(train_keys), len(test_keys)), np.nan)
    for i, tr_key in enumerate(train_keys):
        for j, te_key in enumerate(test_keys):
            heatmap_matrix[i, j] = data.get(tr_key, {}).get(te_key, np.nan)
            print(f"Matrix[{i}][{j}] = {heatmap_matrix[i, j]}")

    if np.isnan(heatmap_matrix).all():
        print(f"No valid data for heatmap for {mode} - {model} - {metric}. Skipping.")
        return

    print("Heatmap Matrix:")
    print(heatmap_matrix)

    # Plot and save the heatmap
    title = f"Heatmap of {metric} for {model.capitalize()} ({mode.capitalize()} Mode)"
    file_name = f"heatmap_{model}_{metric}_{mode}.png"
    plot_heatmap(
        heatmap_matrix,
        title,
        output_dir,
        file_name,
        x_labels=test_keys,
        y_labels=train_keys
    )

def perform_statistical_tests(results, mode, model, output_dir):
    """
    Perform statistical tests and generate heatmaps for a given mode and model.
    Also, create a heatmap for the mean of Delta_z.
    """
    indices = ["high_density", "low_density", "random_sample"]
    delta_z_data = {}

    # Extract Delta_z data for the given mode and model
    for key, values in results.items():
        if key.startswith(model) and key.endswith(mode):
            parts = key.split("_")
            if len(parts) < 6 or "train" not in parts or "test" not in parts:
                print(f"Invalid key format: {key}. Skipping.")
                continue

            train_key, test_key = extract_train_test_keys(key, model)

            if train_key in indices and test_key in indices:
                delta_z = values.get("Delta_z", [])
                if delta_z:
                    delta_z_data[(train_key, test_key)] = np.array(delta_z)
                else:
                    print(f"Missing Delta_z for {key}. Skipping this pair.")

    if not delta_z_data:
        print(f"No valid Delta_z data found for {model} in {mode} mode. Skipping statistical tests.")
        return None

    # --- Create a heatmap for the mean Delta_z ---
    # Initialize a DataFrame for mean Delta_z
    # --- Create a heatmap for the mean Delta_z ---
    mean_delta_z_matrix = pd.DataFrame(index=indices, columns=indices, dtype=float)
    std_delta_z_matrix = pd.DataFrame(index=indices, columns=indices, dtype=float)

    for train1 in indices:
        for train2 in indices:
            dz_values = delta_z_data.get((train1, train2), [])
            if len(dz_values) > 0:
                mean_delta_z_matrix.loc[train1, train2] = np.mean(dz_values)
                std_delta_z_matrix.loc[train1, train2] = float(np.std(dz_values, ddof=1))  # ddof=1 for sample std.
            else:
                mean_delta_z_matrix.loc[train1, train2] = np.nan
                std_delta_z_matrix.loc[train1, train2] = np.nan

    # Plot mean Delta_z heatmap
    if not mean_delta_z_matrix.isna().all().all():
        mean_delta_z_matrix_filled = mean_delta_z_matrix.fillna(0)
        title = f"{model.capitalize()} - Mean Delta_z ({mode.capitalize()})"
        file_name = f"{model}_mean_Delta_z_{mode}_heatmap.png"
        plot_heatmap(mean_delta_z_matrix_filled.values, title, output_dir, file_name, 
                    x_labels=mean_delta_z_matrix_filled.columns, 
                    y_labels=mean_delta_z_matrix_filled.index)

    # Plot std. deviation Delta_z heatmap
    if not std_delta_z_matrix.isna().all().all():
        std_delta_z_matrix_filled = std_delta_z_matrix.fillna(0)
        title = f"{model.capitalize()} - Std. Deviation of Delta_z ({mode.capitalize()})"
        file_name = f"{model}_std_Delta_z_{mode}_heatmap.png"
        plot_heatmap(std_delta_z_matrix_filled.values, title, output_dir, file_name,
                    x_labels=std_delta_z_matrix_filled.columns,
                    y_labels=std_delta_z_matrix_filled.index)
        # Optionally save the matrix to CSV
        csv_file = os.path.join(output_dir, f"{model}_mean_Delta_z_{mode}_matrix.csv")
        mean_delta_z_matrix.to_csv(csv_file)
        print(f"Saved Mean Delta_z matrix for {model} ({mode}) to {csv_file}.")

    # Initialize 3x3 matrices for statistical results
    metrics = ["student_t_stat", "student_p_value", "welch_t_stat", "welch_p_value", "variance_diff", "bartlett_stat", "bartlett_p"]
    matrices = {metric: pd.DataFrame(index=indices, columns=indices) for metric in metrics}

    # Perform statistical tests
    for train1 in indices:
        for train2 in indices:
            # Compare (train1, train1) vs (train2, train2)
            delta_z1 = delta_z_data.get((train1, train1), [])
            delta_z2 = delta_z_data.get((train2, train2), [])

            if len(delta_z1) == 0 or len(delta_z2) == 0:
                print(f"Missing Delta_z data for {train1} vs {train2}. Skipping statistical tests for this pair.")
                continue

            # Variance difference
            matrices["variance_diff"].loc[train1, train2] = float(np.var(delta_z1) - np.var(delta_z2))

            # Student's t-test
            t_stat, p_value = ttest_ind(delta_z1, delta_z2, equal_var=True, nan_policy="omit")
            matrices["student_t_stat"].loc[train1, train2] = float(t_stat)
            matrices["student_p_value"].loc[train1, train2] = p_value

            # Welch's t-test
            t_stat, p_value = ttest_ind(delta_z1, delta_z2, equal_var=False, nan_policy="omit")
            matrices["welch_t_stat"].loc[train1, train2] = t_stat
            matrices["welch_p_value"].loc[train1, train2] = p_value

            # Bartlett's test
            try:
                bartlett_stat, bartlett_p = bartlett(delta_z1, delta_z2)
                matrices["bartlett_stat"].loc[train1, train2] = bartlett_stat
                matrices["bartlett_p"].loc[train1, train2] = bartlett_p
            except ValueError:
                print(f"Bartlett test failed for {train1} vs {train2}. Setting NaN values.")
                matrices["bartlett_stat"].loc[train1, train2] = np.nan
                matrices["bartlett_p"].loc[train1, train2] = np.nan

    # Save and plot matrices
    for metric, matrix in matrices.items():
        matrix = matrix.apply(pd.to_numeric, errors="coerce")  # Ensure numeric values
        matrix.fillna(0, inplace=True)  # Replace NaNs with 0
        title = f"{model.capitalize()} - {metric.replace('_', ' ').title()} ({mode.capitalize()})"
        file_name = f"{model}_{metric}_{mode}_heatmap.png"

        # Plot heatmap
        plot_heatmap(matrix.values, title, output_dir, file_name, x_labels=matrix.columns, y_labels=matrix.index)

        # Save matrix to CSV
        csv_file = os.path.join(output_dir, f"{model}_{metric}_{mode}_matrix.csv")
        matrix.to_csv(csv_file)
        print(f"Saved {metric} matrix for {model} ({mode}) to {csv_file}.")

def main():
    # Load configuration
    #config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micecat1_schnell.yml"
    #config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micecat1_error.yml"
    #config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micecat1_full.yml"
    #config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micecat1_debug.yml"
    config_path = "/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_micecat1_new_experiment.yml"
    config = load_config(config_path)

    # Extract paths from configuration
    json_file = config["input"]["json_file"]
    mode_dirs = {
        "true": config["output"]["true_dir"]["heatmap_directory"],
        "realization": config["output"]["realization_dir"]["heatmap_directory"],
    }

    # Load results from JSON
    results = load_results(json_file)["results"]

    # Define modes and models
    modes = ["true", "realization"]
    models = ["random_forest", "1nn"]

    # Generate heatmaps for performance metrics
    for mode in modes:
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}  # Filter results by mode
        output_dir = mode_dirs[mode]
        for model in models:
            for metric in ["MSE", "R2", "Variance", "Skewness", "Kurtosis"]:
                generate_heatmaps(mode_results, mode, model, metric, output_dir)

    # Perform statistical tests and also produce mean Delta_z heatmap
    for mode in modes:
        mode_results = {k: v for k, v in results.items() if k.endswith(mode)}  # Filter results by mode
        output_dir = mode_dirs[mode]
        for model in models:
            perform_statistical_tests(mode_results, mode, model, output_dir)

if __name__ == "__main__":
    main()