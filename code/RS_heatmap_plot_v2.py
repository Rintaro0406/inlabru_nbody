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

def create_performance_matrices(results):
    """Creates performance matrices (MSE, R^2, Mean Bias) from the results JSON."""
    indices = ["high_density", "low_density", "random_sample"]
    mse_matrix = pd.DataFrame(index=indices, columns=indices)
    r2_matrix = pd.DataFrame(index=indices, columns=indices)
    mean_bias_matrix = pd.DataFrame(index=indices, columns=indices)

    for key, value in results.items():
        train, test = key.split("_train_")
        train = train.lower()
        test = test.lower().replace("_test", "")
        mse_matrix.loc[train, test] = value["MSE"]
        r2_matrix.loc[train, test] = value["R2"]
        mean_bias_matrix.loc[train, test] = np.mean(value["Delta_z"])

    return (
        mse_matrix.astype(float).fillna(0),
        r2_matrix.astype(float).fillna(0),
        mean_bias_matrix.astype(float).fillna(0),
    )

def perform_statistical_tests(results):
    """Performs statistical tests and generates matrices for t-stats, p-values, and variance."""
    indices = ["high_density", "low_density", "random_sample"]
    delta_z_data = {
        key: np.array(value["Delta_z"])
        for key, value in results.items()
    }

    t_stat_matrix = pd.DataFrame(index=indices, columns=indices)
    p_value_matrix = pd.DataFrame(index=indices, columns=indices)
    variance_matrix = pd.DataFrame(index=indices, columns=indices)
    welch_t_stat_matrix = pd.DataFrame(index=indices, columns=indices)
    welch_p_value_matrix = pd.DataFrame(index=indices, columns=indices)
    bartlett_stat_matrix = pd.DataFrame(index=indices, columns=indices)
    bartlett_p_matrix = pd.DataFrame(index=indices, columns=indices)

    for train1 in indices:
        for train2 in indices:
            key1 = f"{train1}_train_{train1}_test"
            key2 = f"{train2}_train_{train2}_test"
            if key1 in delta_z_data and key2 in delta_z_data:
                delta_z1 = delta_z_data[key1]
                delta_z2 = delta_z_data[key2]

                # Variance difference
                variance_matrix.loc[train1, train2] = np.var(delta_z1) - np.var(delta_z2)

                # Regular t-test
                t_stat, p_value = ttest_ind(delta_z1, delta_z2, equal_var=False)
                t_stat_matrix.loc[train1, train2] = t_stat
                p_value_matrix.loc[train1, train2] = p_value

                # Welch's t-test (essentially the same as t-test above with equal_var=False)
                welch_t_stat, welch_p_value = ttest_ind(delta_z1, delta_z2, equal_var=False)
                welch_t_stat_matrix.loc[train1, train2] = welch_t_stat
                welch_p_value_matrix.loc[train1, train2] = welch_p_value

                # Bartlett's test for equal variances
                bartlett_stat, bartlett_p = bartlett(delta_z1, delta_z2)
                bartlett_stat_matrix.loc[train1, train2] = bartlett_stat
                bartlett_p_matrix.loc[train1, train2] = bartlett_p

    return (
        t_stat_matrix.astype(float),
        p_value_matrix.astype(float),
        variance_matrix.astype(float),
        bartlett_stat_matrix.astype(float),
        bartlett_p_matrix.astype(float),
        welch_t_stat_matrix.astype(float),
        welch_p_value_matrix.astype(float),
    )

def main(config_path):
    """Main function to process data and generate heatmaps and statistical results."""
    config = read_yaml_config(config_path)
    json_path = config["input"]["json_file"]
    output_dir = config["output"]["directory"]
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_json_results(json_path)

    # Create performance matrices
    mse_matrix, r2_matrix, mean_bias_matrix = create_performance_matrices(results)

    # Plot performance metrics
    plot_heatmap(mse_matrix, "MSE Heatmap", os.path.join(output_dir, "mse_heatmap.png"))
    plot_heatmap(r2_matrix, "R^2 Heatmap", os.path.join(output_dir, "r2_heatmap.png"))
    plot_heatmap(mean_bias_matrix, "Mean Bias Heatmap", os.path.join(output_dir, "mean_bias_heatmap.png"))

    # Perform statistical tests
    (
        t_stat_matrix,
        p_value_matrix,
        variance_matrix,
        bartlett_stat_matrix,
        bartlett_p_matrix,
        welch_t_stat_matrix,
        welch_p_value_matrix,
    ) = perform_statistical_tests(results)

    # Plot statistical results
    plot_heatmap(t_stat_matrix, "Student T-Statistic Heatmap", os.path.join(output_dir, "t_stat_heatmap.png"))
    plot_heatmap(p_value_matrix, "Student T-Test P-Value Heatmap", os.path.join(output_dir, "p_value_heatmap.png"))
    plot_heatmap(variance_matrix, "Variance Difference Heatmap", os.path.join(output_dir, "variance_heatmap.png"))
    plot_heatmap(bartlett_stat_matrix, "Bartlett Statistic Heatmap", os.path.join(output_dir, "bartlett_stat_heatmap.png"))
    plot_heatmap(bartlett_p_matrix, "Bartlett P-Value Heatmap", os.path.join(output_dir, "bartlett_p_heatmap.png"))
    plot_heatmap(welch_t_stat_matrix, "Welch T-Statistic Heatmap", os.path.join(output_dir, "welch_t_stat_heatmap.png"))
    plot_heatmap(welch_p_value_matrix, "Welch T-Test P-Value Heatmap", os.path.join(output_dir, "welch_p_heatmap.png"))

    # Save statistical results
    save_json(t_stat_matrix.to_dict(), os.path.join(output_dir, "t_stat_results.json"))
    save_json(p_value_matrix.to_dict(), os.path.join(output_dir, "p_value_results.json"))
    save_json(variance_matrix.to_dict(), os.path.join(output_dir, "variance_results.json"))
    save_json(bartlett_stat_matrix.to_dict(), os.path.join(output_dir, "bartlett_stat_results.json"))
    save_json(bartlett_p_matrix.to_dict(), os.path.join(output_dir, "bartlett_p_results.json"))
    save_json(welch_t_stat_matrix.to_dict(), os.path.join(output_dir, "welch_t_stat_results.json"))
    save_json(welch_p_value_matrix.to_dict(), os.path.join(output_dir, "welch_p_results.json"))

if __name__ == "__main__":
    main("/Users/r.kanaki/code/inlabru_nbody/config/RS_heatmap_plot_LSST_With_Noise.yml")