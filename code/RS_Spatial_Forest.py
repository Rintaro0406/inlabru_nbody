import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind
import healpy as hp
import gc
import psutil
import time
from functools import wraps
import json

# Use Agg backend for non-interactive plotting
from matplotlib import use

use("Agg")

"""
Utility Functions
"""


# Time Measurement
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper


# Memory Monitoring
def check_memory(threshold=0.8):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024**2)  # MB
            print(
                f"[{func.__name__}] Memory usage before execution: {memory_before:.2f} MB"
            )
            result = func(*args, **kwargs)
            memory_after = process.memory_info().rss / (1024**2)  # MB
            print(
                f"[{func.__name__}] Memory usage after execution: {memory_after:.2f} MB"
            )
            gc.collect()
            return result

        return wrapper

    return decorator


"""
Data Handling
"""


@check_memory(threshold=0.8)
@measure_time
def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@check_memory(threshold=0.8)
@measure_time
def load_datasets(input_paths):
    return {
        key: pd.read_csv(path, compression="bz2") for key, path in input_paths.items()
    }


"""
Visualization Functions
"""


def visualize_healpix_map(data, nside, title, output_path, unit="Residuals"):
    """Visualizes a HEALPix map."""
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


def plot_scatter_true_vs_predicted(y_test, y_pred, output_path, train_key, test_key):
    """
    Scatter plot of true vs predicted redshift.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        y_test, y_pred, alpha=0.5, s=0.5, label=f"{train_key} Train, {test_key} Test"
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Perfect Prediction",
    )
    plt.xlabel(r"True Redshift $z$")
    plt.ylabel(r"Estimated Photometric Redshift $\hat{z}_\text{photo}$")
    plt.title("True vs Predicted Redshift")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{output_path}/scatter_true_vs_predicted_{train_key}_train_{test_key}_test.png"
    )
    plt.close()


def plot_histogram_delta_z(delta_z, output_path, train_key, test_key):
    """
    Histogram of redshift differences (Delta z).
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(delta_z, bins=100, kde=True, color="blue", alpha=0.7)
    plt.axvline(
        delta_z.mean(),
        color="r",
        linestyle="--",
        label=f"Mean Δz: {delta_z.mean():.4f}",
    )
    plt.xlabel(r"$\Delta z = z - \hat{z}_\text{photo}$")
    plt.ylabel("Frequency")
    plt.title(r"Distribution of $\Delta z$")
    plt.legend()
    plt.grid()
    plt.savefig(
        f"{output_path}/histogram_delta_z_{train_key}_train_{test_key}_test.png"
    )
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


def plot_density_heatmap(y_test, y_pred, output_path, train_key, test_key):
    """
    Heatmap of predictions vs true values (density visualization).
    """
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=y_test, y=y_pred, fill=True, cmap="Blues", alpha=0.6)
    plt.xlabel(r"True Redshift $z$")
    plt.ylabel(r"Estimated Photometric Redshift $\hat{z}_\text{photo}$")
    plt.title(
        f"Density of True vs Predicted Redshift ({train_key} Train, {test_key} Test)"
    )
    plt.grid()
    plt.savefig(
        f"{output_path}/density_true_vs_predicted_{train_key}_train_{test_key}_test.png"
    )
    plt.close()


"""
Random Forest Training and Evaluation
"""


@measure_time
def train_random_forest(X_train, y_train, config):
    rf = RandomForestRegressor(
        n_estimators=config["parameters"]["n_estimators"],
        max_depth=config["parameters"]["max_depth"],
        random_state=config["parameters"]["random_state"],
    )
    rf.fit(X_train, y_train)
    return rf


@measure_time
def evaluate_model(y_test, y_pred, test_data, nside, output_path, train_key, test_key):
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
    visualize_healpix_map(
        test_data,
        nside,
        "Residual Map (Δz)",
        f"{output_path}/healpix_residuals_{train_key}_train_{test_key}_test.png",
    )

    return mse, r2, delta_z


# Utility to save results, including p-values and train-test combinations
def save_results_with_pvalues(results, config, p_values, output_dir):
    # Convert results keys to descriptive strings
    formatted_results = {}
    for key, value in results.items():
        train_key, test_key = key
        formatted_key = f"{train_key}_train_{test_key}_test"
        formatted_results[formatted_key] = value
        formatted_results[formatted_key]["train_test_combination"] = (
            f"{train_key.capitalize()} Train, {test_key.capitalize()} Test"
        )

    # Prepare final output structure
    results_to_save = {
        "hyperparameters": config["parameters"],
        "results": formatted_results,
        "p_values": p_values,
    }

    # Save to JSON
    results_file = os.path.join(output_dir, "results_with_pvalues.json")
    with open(results_file, "w") as file:
        json.dump(results_to_save, file, indent=4)
    print(f"Results and p-values saved to {results_file}")

    # Save hyperparameters as a YAML file
    hyperparams_file = os.path.join(output_dir, "hyperparameters.yml")
    with open(hyperparams_file, "w") as file:
        yaml.dump(config["parameters"], file, default_flow_style=False)
    print(f"Hyperparameters saved to {hyperparams_file}")


# Main Function
@measure_time
def main():
    config = read_yaml_config(
        "/Users/r.kanaki/code/inlabru_nbody/config/RS_Spatial_Forest_LSST_With_Noise.yml"
    )
    input_paths = config["input_paths"]
    output_dir = config["output_paths"]["base_dir"]
    os.makedirs(output_dir, exist_ok=True)

    datasets = load_datasets(input_paths)
    features = [
        "des_asahi_full_g_abs_mag",
        "des_asahi_full_r_abs_mag",
        "des_asahi_full_i_abs_mag",
        "des_asahi_full_z_abs_mag",
        "des_asahi_full_y_abs_mag",
    ]
    target = "z_cgal"
    nside = config["parameters"]["nside"]

    results = {}
    p_values = []

    # Train and evaluate models
    for train_key, train_data in datasets.items():
        print(f"Training on {train_key} dataset...")
        X_train = (train_data[features] - train_data[features].mean()) / train_data[
            features
        ].std()
        y_train = train_data[target]

        rf = train_random_forest(X_train, y_train, config)

        # Plot feature importance
        plot_feature_importance(
            rf.feature_importances_, features, output_dir, train_key
        )

        # Evaluate on each test dataset
        for test_key, test_data in datasets.items():
            print(f"Testing on {test_key} dataset...")
            X_test = (test_data[features] - test_data[features].mean()) / test_data[
                features
            ].std()
            y_test = test_data[target]

            y_pred = rf.predict(X_test)
            mse, r2, delta_z = evaluate_model(
                y_test, y_pred, test_data, nside, output_dir, train_key, test_key
            )
            results[(train_key, test_key)] = {
                "train_test_combination": f"{train_key.capitalize()} Train, {test_key.capitalize()} Test",
                "MSE": mse,
                "R2": r2,
                "Delta_z": delta_z.tolist(),  # Convert numpy array to list for JSON compatibility
            }

    # Statistical Tests
    keys = list(results.keys())
    for i, key1 in enumerate(keys):
        for key2 in keys[i + 1 :]:
            delta_z1 = np.array(results[key1]["Delta_z"])
            delta_z2 = np.array(results[key2]["Delta_z"])
            _, p_value = ttest_ind(delta_z1, delta_z2)
            p_values.append(
                {
                    "comparison": f"{key1[0]}_train_{key1[1]}_test vs {key2[0]}_train_{key2[1]}_test",
                    "p_value": p_value,
                }
            )
            print(f"T-test between {key1} and {key2}: p-value = {p_value:.4f}")

    # Save results and p-values
    save_results_with_pvalues(results, config, p_values, output_dir)


if __name__ == "__main__":
    main()
