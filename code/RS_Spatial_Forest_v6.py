import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import ttest_ind, ttest_ind_from_stats, bartlett
import healpy as hp
import gc
import psutil
import time
from functools import wraps
import json
from matplotlib import use
import sys

use("Agg")

"""
Utility Functions
"""

def measure_time(func):
    """
    A decorator to measure the execution time of a function.

    Parameters:
        func (function): The function to be measured.

    Returns:
        wrapper (function): The wrapped function with time measurement.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds.")
        return result

    return wrapper


def check_memory(threshold=0.8):
    """
    A decorator to monitor memory usage before and after a function call.
    Exits the program if memory usage exceeds the specified threshold.

    Parameters:
        threshold (float): The memory usage limit (e.g., 0.8 for 80%).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            total_memory = psutil.virtual_memory().total / (1024**2)  # Total system memory in MB

            print(f"Total system memory: {total_memory:.2f} MB")

            # Memory usage before function execution
            memory_before = process.memory_info().rss / (1024**2)  # Memory in MB
            print(f"[{func.__name__}] Memory before execution: {memory_before:.2f} MB")

            result = func(*args, **kwargs)

            # Memory usage after function execution
            memory_after = process.memory_info().rss / (1024**2)  # Memory in MB
            print(f"[{func.__name__}] Memory after execution: {memory_after:.2f} MB")

            # Force garbage collection
            gc.collect()
            memory_after_gc = process.memory_info().rss / (1024**2)  # Memory in MB
            print(f"[{func.__name__}] Memory after garbage collection: {memory_after_gc:.2f} MB")

            if psutil.virtual_memory().percent / 100.0 > threshold:
                print(f"[{func.__name__}] Memory usage exceeded {threshold * 100}%. Exiting.")
                sys.exit(1)

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
    print("Loading datasets...")
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

def plot_density_histogram(y_test, y_pred, output_path, train_key, test_key):
    """
    Plot histograms of true and predicted redshifts with statistical metrics.

    Parameters:
        y_test (np.ndarray): True redshift values.
        y_pred (np.ndarray): Predicted redshift values.
        output_path (str): Path to save the plot.
        train_key (str): Identifier for the training dataset.
        test_key (str): Identifier for the testing dataset.
    """
    # Compute residuals and statistics
    residuals = y_test - y_pred
    variance = np.var(residuals)
    skewness = np.mean(residuals) / np.std(residuals)
    kurtosis = np.mean(residuals ** 4) / variance ** 2

    # Create the histogram plot
    plt.figure(figsize=(10, 8))
    sns.histplot(y_test, bins=50, color="blue", alpha=0.6, label="True Redshift")
    sns.histplot(y_pred, bins=50, color="orange", alpha=0.6, label="Predicted Redshift")

    # Add labels and title
    plt.xlabel("Redshift $z$", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title("Histogram of True and Predicted Redshift", fontsize=16)

    # Add statistical annotations
    stats_text = (
        f"$\mathrm{{Variance}}$: {variance:.4f}\n"
        f"$\mathrm{{Skewness}}$: {skewness:.4f}\n"
        f"$\mathrm{{Kurtosis}}$: {kurtosis:.4f}"
    )
    plt.text(
        0.02, 0.98, stats_text, fontsize=12, ha="left", va="top", transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    # Add legend and grid
    plt.legend(fontsize=12)
    plt.grid(visible=True, which="major", linestyle="--", alpha=0.7)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_path}/histogram_redshift_{train_key}_train_{test_key}_test.png", dpi=300)
    plt.close()

def visualize_healpix_map(data, nside, title, output_path, unit="Residuals"):
    """Visualize HEALPix map."""
    npix = hp.nside2npix(nside)
    hp_map = np.zeros(npix)
    theta = np.radians(90 - data["dec"])
    phi = np.radians(data["ra"])
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

def plot_feature_importance(importances, features, output_path, train_key):
    """
    Feature importance plot.
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.xlabel("Feature Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.title(f"Feature Importance ({train_key} Train)", fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importance_{train_key}.png", dpi=300)
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
def train_1nn(X_train, y_train, config):
    """Train 1-Nearest Neighbor model."""
    nn = KNeighborsRegressor(n_neighbors=1, metric='euclidean')
    nn.fit(X_train, y_train)
    return nn

@measure_time
def evaluate_model(y_test, y_pred, test_data, nside, output_path, train_key, test_key):
    """Evaluate model and perform visualizations."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    delta_z = y_test - y_pred
    #test_data["delta_z"] = delta_z
    var= np.var(delta_z)
    skewness = np.mean(delta_z) / np.std(delta_z)
    kurtosis = np.mean(delta_z ** 4) / var ** 2
    # Density Histogram
    plot_density_histogram(y_test, y_pred, output_path, train_key, test_key)
    # Scatter Plot
    plot_scatter_true_vs_predicted(y_test, y_pred, output_path, train_key, test_key)

    # Residual Histogram
    plot_histogram_delta_z(delta_z, output_path, train_key, test_key)

    # Density Heatmap
    plot_density_heatmap(y_test, y_pred, output_path, train_key, test_key)

    # HEALPix Map of Residuals
    test_data["delta_z"] = delta_z
    visualize_healpix_map(test_data, nside, "Residual Map (Δz)", f"{output_path}/healpix_residuals_{train_key}_train_{test_key}_test.png")
    return mse, r2, delta_z, var, skewness, kurtosis

"""
Save Results and P-Values
"""
def save_results_with_pvalues(results, config, p_values, stats_results, output_dir, filename="results.json"):
    """Save results, p-values, and statistical test results."""
    formatted_results = {}

    # Iterate through models and train-test pairs
    for model_key, model_results in results.items():
        for key, value in model_results.items():
            # Ensure key is a tuple of (train_key, test_key, mode)
            if isinstance(key, tuple) and len(key) == 3:
                train_key, test_key, mode = key
            else:
                raise ValueError(f"Unexpected key format: {key}. Expected a tuple of (train_key, test_key, mode).")

            # Format key and store results
            formatted_key = f"{model_key}_{train_key}_train_{test_key}_test_{mode}"
            formatted_results[formatted_key] = value
            formatted_results[formatted_key]["train_test_combination"] = f"{train_key.capitalize()} Train, {test_key.capitalize()} Test ({mode.capitalize()})"

    # Prepare results for saving
    results_to_save = {
        "hyperparameters": config["parameters"],
        "results": formatted_results,
        "p_values": p_values,
        "statistical_tests": stats_results
    }

    # Save results as JSON
    results_file = os.path.join(output_dir, filename)
    with open(results_file, 'w') as file:
        json.dump(results_to_save, file, indent=4)
    print(f"Results and p-values saved to {results_file}")

    # Save hyperparameters as YAML
    hyperparams_file = os.path.join(output_dir, "hyperparameters.yml")
    with open(hyperparams_file, 'w') as file:
        yaml.dump(config["parameters"], file, default_flow_style=False)
    print(f"Hyperparameters saved to {hyperparams_file}")

def plot_histogram_true_vs_estimated(y_true, y_est, output_path, suffix=""):
    """
    Plot a histogram of true vs estimated redshifts within specified bins.
    """
    plt.figure(figsize=(10,8))
    sns.histplot(y_true, color='blue', alpha=0.6, label='True Redshift', bins=50)
    sns.histplot(y_est, color='orange', alpha=0.6, label='Estimated Redshift', bins=50)
    plt.xlabel('Redshift z')
    plt.ylabel('Frequency')
    plt.xlim(0.3, 1.2)
    plt.title('Histogram of True vs Estimated Redshift (Filtered by RF z_pred)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_path, f'histogram_redshift_nn_binned{suffix}.png'))
    plt.close()

def run_new_experiment(nn, test_data, y_pred_rf, train_data, features, target, output_dir, p_values, stats_results, nside, suffix="", train_key="exp", test_key="exp"):
    """
    Run the new experiment:
    1. Bin test_data by RF-predicted redshifts into [0.3,0.6), [0.6,0.9), [0.9,1.2).
    2. Apply 1NN model again on the filtered subset.
    3. Plot histograms and perform statistical tests.
    4. Store p-values and results in p_values and stats_results.
    """

    bins = [(0.3, 0.6), (0.6, 0.9), (0.9, 1.2)]
    filtered_data_list = []
    for (low, high) in bins:
        mask = (y_pred_rf >= low) & (y_pred_rf < high)
        filtered_data_list.append(test_data[mask])

    test_data2 = pd.concat(filtered_data_list)
    if len(test_data2) == 0:
        print("No data falls into specified bins for the new experiment.")
        return p_values, stats_results

    # Prepare test_data2
    X_test2 = (test_data2[features] - train_data[features].mean()) / train_data[features].std()
    y_test2 = test_data2[target]

    # Predict with 1NN again
    y_pred_nn2 = nn.predict(X_test2)

    # Plot histogram of the new subset
    plot_histogram_true_vs_estimated(y_test2, y_pred_nn2, output_dir, suffix=suffix)

    # If you want other plots (scatter, delta_z histogram, etc.), you can use evaluate_model or replicate logic:
    # We'll use evaluate_model here to generate similar plots:
    # Note: evaluate_model requires train_key and test_key, we pass generic ones
    mse_new, r2_new, delta_z_new, var_new, skew_new, kurt_new = evaluate_model(
        y_test2, y_pred_nn2, test_data2.copy(), nside, output_dir, train_key, test_key
    )

    # Perform a Welch's t-test between y_test2 and y_pred_nn2 to see if distributions differ
    t_stat, p_val = ttest_ind(y_test2, y_pred_nn2, equal_var=False, nan_policy='omit')
    # Store p-value
    experiment_key = f"new_experiment_{train_key}_{test_key}"
    p_values.append({"experiment": experiment_key, "p_value": p_val})
    stats_results["welchs_t_test"][experiment_key] = {"t_stat": t_stat, "p_value": p_val}

    return p_values, stats_results

@measure_time
def main():
    config = read_yaml_config("/Users/r.kanaki/code/inlabru_nbody/config/RS_Spatial_Forest_micecat1_error.yml")
    input_paths = config["input_paths"]
    base_output_dir = config["output_paths"]["base_dir"]

    datasets = load_datasets(input_paths)

    features_true = [
        "g_des_true", "r_des_true", "i_des_true", "z_des_true", "y_des_true",
        "g_des_true_error", "r_des_true_error", "i_des_true_error", "z_des_true_error", "y_des_true_error"
    ]
    features_realization = [
        "g_des_realization", "r_des_realization", "i_des_realization", "z_des_realization", "y_des_realization",
        "g_des_realization_error", "r_des_realization_error", "i_des_realization_error", "z_des_realization_error", "y_des_realization_error"
    ]
    target_true = "z"
    target_realization = "z"

    nside = config["parameters"]["nside"]

    results = {"random_forest": {}, "1nn": {}}
    p_values = []
    stats_results = {"students_t_test": {}, "welchs_t_test": {}, "variance_test": {}}

    for mode in ["true", "realization"]:
        print(f"Processing {mode} case")
        features = features_true if mode == "true" else features_realization
        target = target_true if mode == "true" else target_realization

        mode_output_dir = os.path.join(base_output_dir, mode)
        rf_output_dir = os.path.join(mode_output_dir, "random_forest")
        nn_output_dir = os.path.join(mode_output_dir, "1nn")
        os.makedirs(rf_output_dir, exist_ok=True)
        os.makedirs(nn_output_dir, exist_ok=True)

        for train_key, train_data in datasets.items():
            print(f"Processing training dataset: {train_key} ({mode})")

            X_train = (train_data[features] - train_data[features].mean()) / train_data[features].std()
            y_train = train_data[target]

            rf = train_random_forest(X_train, y_train, config)
            nn = train_1nn(X_train, y_train, config)

            # Feature importance
            plot_feature_importance(rf.feature_importances_, features, rf_output_dir, train_key)

            for test_key, test_data in datasets.items():
                print(f"Processing testing dataset: {test_key} ({mode})")

                X_test = (test_data[features] - train_data[features].mean()) / train_data[features].std()
                y_test = test_data[target]

                # Evaluate RF
                rf_test_output_dir = os.path.join(rf_output_dir, f"{train_key}_train_{test_key}_test")
                os.makedirs(rf_test_output_dir, exist_ok=True)
                y_pred_rf = rf.predict(X_test)
                rf_mse, rf_r2, rf_delta_z, rf_var, rf_skew, rf_kurto = evaluate_model(
                    y_test, y_pred_rf, test_data.copy(), nside, rf_test_output_dir, train_key, test_key
                )
                results["random_forest"][(train_key, test_key, mode)] = {
                    "Delta_z": rf_delta_z.tolist(), "MSE": rf_mse, "R2": rf_r2,
                    "Variance": rf_var, "Skewness": rf_skew, "Kurtosis": rf_kurto
                }

                # Evaluate 1NN
                nn_test_output_dir = os.path.join(nn_output_dir, f"{train_key}_train_{test_key}_test")
                os.makedirs(nn_test_output_dir, exist_ok=True)
                y_pred_nn = nn.predict(X_test)
                nn_mse, nn_r2, nn_delta_z, nn_var, nn_skew, nn_kurto = evaluate_model(
                    y_test, y_pred_nn, test_data.copy(), nside, nn_test_output_dir, train_key, test_key
                )
                results["1nn"][(train_key, test_key, mode)] = {
                    "Delta_z": nn_delta_z.tolist(), "MSE": nn_mse, "R2": nn_r2,
                    "Variance": nn_var, "Skewness": nn_skew, "Kurtosis": nn_kurto
                }
                save_results_with_pvalues(results, config, p_values, stats_results, base_output_dir, filename="results_old_experiment.json")

    # Run the new experiment on all previously processed data
                # Run the new experiment
                p_values, stats_results = run_new_experiment(
                    nn, test_data.copy(), y_pred_rf, train_data, features, target,
                    nn_test_output_dir, p_values, stats_results, nside,
                    suffix="_" + mode, train_key=train_key, test_key=test_key
                )

    save_results_with_pvalues(results, config, p_values, stats_results, base_output_dir, filename="results_new_experiment.json")

if __name__ == "__main__":
    main()