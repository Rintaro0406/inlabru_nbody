import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind, ttest_ind_from_stats, bartlett
import healpy as hp
import gc
import psutil
import time
from functools import wraps
import json
from matplotlib import use

use("Agg")

"""
Utility Functions
"""

def measure_time(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def check_memory(threshold=0.8):
    """Decorator to monitor memory usage."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)
            print(f"[{func.__name__}] Memory usage before: {memory_before:.2f} MB")
            result = func(*args, **kwargs)
            memory_after = process.memory_info().rss / (1024**2)
            print(f"[{func.__name__}] Memory usage after: {memory_after:.2f} MB")
            gc.collect()
            return result
        return wrapper
    return decorator

"""
Data Handling
"""

@check_memory()
@measure_time
def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

@check_memory()
@measure_time
def load_datasets(input_paths):
    return {key: pd.read_csv(path) for key, path in input_paths.items()}

"""
Statistical Functions
"""

def students_t_test(mean1, var1, n1, mean2, var2, n2):
    """Perform Student's t-test for means."""
    return ttest_ind_from_stats(mean1, np.sqrt(var1), n1, mean2, np.sqrt(var2), n2, equal_var=True)

def welchs_t_test(data1, data2):
    """Perform Welch's t-test for unequal variances."""
    return ttest_ind(data1, data2, equal_var=False)

def variance_test(data1, data2):
    """Perform variance test using Bartlett's test."""
    return bartlett(data1, data2)

"""
Visualization Functions
"""

def plot_scatter_true_vs_predicted(y_test, y_pred, output_path, train_key, test_key):
    """Scatter plot of true vs predicted redshift."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=0.5, label=f"{train_key} Train, {test_key} Test")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
    plt.xlabel("True Redshift $z$")
    plt.ylabel("Estimated Photometric Redshift $\hat{z}_\text{photo}$")
    plt.title("True vs Predicted Redshift")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/scatter_true_vs_predicted_{train_key}_train_{test_key}_test.png")
    plt.close()

def plot_histogram_delta_z(delta_z, output_path, train_key, test_key):
    """Histogram of redshift differences (Delta z)."""
    plt.figure(figsize=(8, 6))
    sns.histplot(delta_z, bins=100, kde=True, color="blue", alpha=0.7)
    plt.axvline(delta_z.mean(), color='r', linestyle='--', label=f"Mean Δz: {delta_z.mean():.4f}")
    plt.xlabel("Δz = z - $\hat{z}_\text{photo}$")
    plt.ylabel("Frequency")
    plt.title("Distribution of Δz")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/histogram_delta_z_{train_key}_train_{test_key}_test.png")
    plt.close()

def visualize_healpix_map(data, nside, title, output_path, unit="Residuals"):
    """Visualize HEALPix map."""
    npix = hp.nside2npix(nside)
    hp_map = np.zeros(npix)
    theta = np.radians(90 - data["dec_gal"])
    phi = np.radians(data["ra_gal"])
    pixel_indices = hp.ang2pix(nside, theta, phi)
    np.add.at(hp_map, pixel_indices, data["delta_z"])
    hp.mollview(hp_map, title=title, unit=unit, cmap="coolwarm")
    hp.graticule()
    plt.savefig(output_path)
    plt.close()

def plot_density_heatmap(y_test, y_pred, output_path, train_key, test_key):
    """
    Heatmap of predictions vs true values (density visualization).
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=y_test, y=y_pred, fill=True, cmap="Blues", alpha=0.6)
    plt.xlabel(r"True Redshift $z$")
    plt.ylabel(r"Estimated Photometric Redshift $\hat{z}_\text{photo}$")
    plt.title(f"Density of True vs Predicted Redshift ({train_key} Train, {test_key} Test)")
    plt.grid()
    plt.savefig(f"{output_path}/density_true_vs_predicted_{train_key}_train_{test_key}_test.png")
    plt.close()

"""
Model Training and Evaluation
"""

@measure_time
def train_random_forest(X_train, y_train, config):
    """Train Random Forest model."""
    rf = RandomForestRegressor(
        n_estimators=config["parameters"]["n_estimators"],
        max_depth=config["parameters"]["max_depth"],
        random_state=config["parameters"]["random_state"]
    )
    rf.fit(X_train, y_train)
    return rf

@measure_time
def evaluate_model(y_test, y_pred, test_data, nside, output_path, train_key, test_key):
    """Evaluate model and perform visualizations."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    delta_z = y_test - y_pred
    test_data["delta_z"] = delta_z

    # Scatter Plot
    plot_scatter_true_vs_predicted(y_test, y_pred, output_path, train_key, test_key)

    # Residual Histogram
    plot_histogram_delta_z(delta_z, output_path, train_key, test_key)

    # Density Heatmap
    plot_density_heatmap(y_test, y_pred, output_path, train_key, test_key)

    # HEALPix Map of Residuals
    visualize_healpix_map(test_data, nside, "Residual Map (Δz)", f"{output_path}/healpix_residuals_{train_key}_train_{test_key}_test.png")
    return mse, r2, delta_z

"""
Save Results and P-Values
"""

def save_results_with_pvalues(results, config, p_values, stats_results, output_dir):
    """Save results, p-values, and statistical test results."""
    formatted_results = {}
    for key, value in results.items():
        train_key, test_key = key
        formatted_key = f"{train_key}_train_{test_key}_test"
        formatted_results[formatted_key] = value
        formatted_results[formatted_key]["train_test_combination"] = f"{train_key.capitalize()} Train, {test_key.capitalize()} Test"

    results_to_save = {
        "hyperparameters": config["parameters"],
        "results": formatted_results,
        "p_values": p_values,
        "statistical_tests": stats_results
    }

    results_file = os.path.join(output_dir, "results_with_pvalues.json")
    with open(results_file, 'w') as file:
        json.dump(results_to_save, file, indent=4)
    print(f"Results and p-values saved to {results_file}")

    hyperparams_file = os.path.join(output_dir, "hyperparameters.yml")
    with open(hyperparams_file, 'w') as file:
        yaml.dump(config["parameters"], file, default_flow_style=False)
    print(f"Hyperparameters saved to {hyperparams_file}")

"""
Main Function
"""

@measure_time
def main():
    config = read_yaml_config("/Users/r.kanaki/code/inlabru_nbody/config/RS_Spatial_Forest_LSST_With_Noise.yml")
    input_paths = config["input_paths"]
    output_dir = config["output_paths"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)

    datasets = load_datasets(input_paths)
    features = ["des_asahi_full_g_true", "des_asahi_full_r_true", "des_asahi_full_i_true",
                "des_asahi_full_z_true", "des_asahi_full_y_true"]
    target = "z_cgal"
    nside = config["parameters"]["nside"]

    results = {}
    p_values = []
    stats_results = {"students_t_test": {}, "welchs_t_test": {}, "variance_test": {}}

    # Train and evaluate models
    for train_key, train_data in datasets.items():
        X_train = train_data[features]
        y_train = train_data[target]
        rf = train_random_forest(X_train, y_train, config)

        for test_key, test_data in datasets.items():
            X_test = test_data[features]
            y_test = test_data[target]
            y_pred = rf.predict(X_test)
            mse, r2, delta_z = evaluate_model(y_test, y_pred, test_data, nside, output_dir, train_key, test_key)
            results[(train_key, test_key)] = {"Delta_z": delta_z.tolist(), "MSE": mse, "R2": r2}

    # Perform statistical tests
    keys = list(results.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i + 1:]:
            delta_z1 = np.array(results[key1]["Delta_z"])
            delta_z2 = np.array(results[key2]["Delta_z"])
            mean1, var1, n1 = delta_z1.mean(), delta_z1.var(ddof=1), len(delta_z1)
            mean2, var2, n2 = delta_z2.mean(), delta_z2.var(ddof=1), len(delta_z2)

            t_stat_students, p_value_students = students_t_test(mean1, var1, n1, mean2, var2, n2)
            t_stat_welch, p_value_welch = welchs_t_test(delta_z1, delta_z2)
            t_stat_var, p_value_var = variance_test(delta_z1, delta_z2)

            comparison = f"{key1[0]}_train_{key1[1]}_test vs {key2[0]}_train_{key2[1]}_test"
            p_values.append({"comparison": comparison, "students_t_test": p_value_students, "welchs_t_test": p_value_welch, "variance_test": p_value_var})
            stats_results["students_t_test"][comparison] = {"t_stat": t_stat_students, "p_value": p_value_students}
            stats_results["welchs_t_test"][comparison] = {"t_stat": t_stat_welch, "p_value": p_value_welch}
            stats_results["variance_test"][comparison] = {"t_stat": t_stat_var, "p_value": p_value_var}

    # Save results
    save_results_with_pvalues(results, config, p_values, stats_results, output_dir)

if __name__ == "__main__":
    main()