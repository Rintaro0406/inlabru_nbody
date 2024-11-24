import os
import yaml
import pandas as pd
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from functools import wraps
import gc
import time
import psutil
import sys
from matplotlib import use
import seaborn as sns

# Use non-interactive backend
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
Part 1: Input and Data Handling
"""


@check_memory(threshold=0.8)
def read_yaml_config(file_path):
    """
    Reads a YAML configuration file.

    Parameters:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


@check_memory(threshold=0.8)
def read_catalog(file_path, index_col=None, chunksize=10000):
    """
    Reads a CSV catalog file in chunks and concatenates into a single DataFrame.

    Parameters:
        file_path (str): Path to the CSV file.
        index_col (str, optional): Column to use as the index.
        chunksize (int, optional): Number of rows per chunk.

    Returns:
        pd.DataFrame: Loaded catalog data.
    """
    dtype_dict = {
        "unique_gal_id": "int64",
        "ra_gal": "float64",
        "dec_gal": "float64",
        "z_cgal": "float64",
        "des_asahi_full_g_abs_mag": "float32",
        "des_asahi_full_r_abs_mag": "float32",
        "des_asahi_full_i_abs_mag": "float32",
        "des_asahi_full_z_abs_mag": "float32",
        "des_asahi_full_y_abs_mag": "float32"
    }
    chunks = []
    for chunk in pd.read_csv(file_path, sep=",", index_col=index_col, comment="#",
                             na_values=r"\N", compression="bz2", chunksize=chunksize,
                             dtype=dtype_dict):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


"""
Part 2: HEALPix Map Generation and Visualization
"""


def create_healpix_map(data, nside):
    """
    Generates a HEALPix map of galaxy density, excluding boundary pixels and near-boundary pixels.

    Parameters:
        data (pd.DataFrame): Input data containing 'ra_gal' and 'dec_gal'.
        nside (int): HEALPix resolution parameter.

    Returns:
        np.ndarray: HEALPix map of galaxy density.
    """
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - data["dec_gal"])  # Convert to colatitude
    phi = np.radians(data["ra_gal"])  # Convert to radians
    pixel_indices = hp.ang2pix(nside, theta, phi)
    
    # Create density map
    density_map = np.bincount(pixel_indices, minlength=npix)
    
    # Filter out boundary pixels (RA/Dec between 0 and 90 degrees)
    valid_ra_dec_mask = (data["ra_gal"] >= 0) & (data["ra_gal"] <= 90) & \
                        (data["dec_gal"] >= 0) & (data["dec_gal"] <= 90)
    valid_pixels = pixel_indices[valid_ra_dec_mask]
    valid_density_map = np.bincount(valid_pixels, minlength=npix)
    
    return valid_density_map


def plot_healpix_map(healpix_map, title, output_path):
    """
    Plots and saves a HEALPix map, excluding invalid pixels.

    Parameters:
        healpix_map (np.ndarray): HEALPix map to visualize.
        title (str): Title for the map.
        output_path (str): Path to save the map image.
    """
    hp.mollview(healpix_map, title=title, unit="Galaxy Density", cmap="viridis")
    plt.savefig(output_path)
    plt.close()


"""
Part 3: Data Stratification
"""
def stratify_data(data, healpix_map, nside):
    """
    Stratifies data into high-density, low-density, and random subsets, ensuring no boundary pixels are included.

    Parameters:
        data (pd.DataFrame): Input data containing galaxy information.
        healpix_map (np.ndarray): HEALPix map of galaxy density.
        nside (int): HEALPix resolution parameter.

    Returns:
        tuple: (high_density_data, low_density_data, random_sample_data)
    """
    # Filter out boundary pixels by ensuring RA/Dec are within 0 to 90 degrees
    valid_ra_dec_mask = (data["ra_gal"] >= 0) & (data["ra_gal"] <= 90) & \
                        (data["dec_gal"] >= 0) & (data["dec_gal"] <= 90)
    data_filtered = data[valid_ra_dec_mask]

    # Create HEALPix map with filtered data
    filtered_healpix_map = create_healpix_map(data_filtered, nside)
    
    # Determine density threshold for high-density pixels
    threshold = np.percentile(filtered_healpix_map[filtered_healpix_map > 0], 90)
    high_density_pixels = np.where(filtered_healpix_map > threshold)[0]
    low_density_pixels = np.where((filtered_healpix_map <= threshold) & (filtered_healpix_map > 0))[0]

    # Convert RA/Dec to HEALPix pixel indices for filtered data
    theta = np.radians(90.0 - data_filtered["dec_gal"])
    phi = np.radians(data_filtered["ra_gal"])
    pixel_indices = hp.ang2pix(nside, theta, phi)

    # Filter data into high-density and low-density subsets
    necessary_columns = ["unique_gal_id", "ra_gal", "dec_gal", "z_cgal",
                         "des_asahi_full_g_abs_mag", "des_asahi_full_r_abs_mag",
                         "des_asahi_full_i_abs_mag", "des_asahi_full_z_abs_mag",
                         "des_asahi_full_y_abs_mag"]

    high_density_data = data_filtered[data_filtered.index.isin(high_density_pixels)][necessary_columns]
    low_density_data = data_filtered[data_filtered.index.isin(low_density_pixels)][necessary_columns]

    # Determine the minimum size among high-density, low-density, and random subsets
    min_size = min(len(high_density_data), len(low_density_data), len(data_filtered) // 3)

    # Downsample each subset to the minimum size
    high_density_data = high_density_data.sample(n=min_size, random_state=42)
    low_density_data = low_density_data.sample(n=min_size, random_state=42)
    random_sample_data = data_filtered.sample(n=min_size, random_state=42)[necessary_columns]

    return high_density_data, low_density_data, random_sample_data

def save_datasets(high_data, low_data, random_data, output_paths):
    """
    Saves stratified datasets in compressed CSV format.

    Parameters:
        high_data (pd.DataFrame): High-density data.
        low_data (pd.DataFrame): Low-density data.
        random_data (pd.DataFrame): Randomly sampled data.
        output_paths (dict): Paths to save the datasets.
    """
    high_data.to_csv(output_paths["high"], index=False, compression="bz2")
    low_data.to_csv(output_paths["low"], index=False, compression="bz2")
    random_data.to_csv(output_paths["random"], index=False, compression="bz2")
"""
4. Additional Plot Functions
"""

def plot_magnitude_distributions(data, features, output_path, subset_name="all"):
    """
    Plot distributions of observed magnitudes.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        features (list): List of magnitude feature columns.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
    """
    plt.figure(figsize=(10, 6))
    for feature in features:
        sns.histplot(data[feature], bins=50, kde=True, label=feature, alpha=0.6)
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Observed Magnitudes ({subset_name.capitalize()})")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/magnitude_distributions_{subset_name}.png")
    plt.close()


def plot_pairwise_relationships(data, features, target, output_path, subset_name="all"):
    """
    Create a pairplot of magnitudes and redshift.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        features (list): List of magnitude feature columns.
        target (str): Column name of the redshift target.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
    """
    sns.pairplot(data, vars=features + [target], diag_kind="kde", corner=True, height=2.5)
    plt.suptitle(f"Pairwise Relationships Between Magnitudes and Redshift ({subset_name.capitalize()})", y=1.02)
    plt.savefig(f"{output_path}/pairwise_relationships_{subset_name}.png")
    plt.close()


def plot_true_redshift_distribution(data, target, output_path, subset_name="all"):
    """
    Plot histogram of true redshifts.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        target (str): Column name of the redshift target.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(data[target], bins=50, kde=True, color="blue", alpha=0.7)
    plt.xlabel("True Redshift (z_cgal)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of True Redshifts ({subset_name.capitalize()})")
    plt.grid()
    plt.savefig(f"{output_path}/true_redshifts_{subset_name}.png")
    plt.close()


def plot_density_histogram(healpix_map, threshold, output_path):
    """
    Plot a normalized histogram of galaxy densities with high/low density boundaries.

    Parameters:
        healpix_map (np.ndarray): HEALPix map of galaxy density.
        threshold (float): Density threshold separating high/low density regions.
        output_path (str): Path to save the plot.
    """
    # Remove zero values for normalization
    non_zero_values = healpix_map[healpix_map > 0]

    plt.figure(figsize=(10, 6))
    # Normalize the histogram by setting `stat="density"` in sns.histplot
    sns.histplot(non_zero_values, bins=100, color="gray", alpha=0.7, label="Galaxy Density", stat="density")
    plt.axvline(threshold, color="red", linestyle="--", label=f"High/Low Density Threshold ({threshold:.2f})")
    plt.xlabel("Galaxy Density")
    plt.ylabel("Probability Density (Log scale)")
    plt.title("Normalized Galaxy Density Distribution with High/Low Boundary")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/density_histogram_normalized.png")
    plt.close()


"""
Main Function
"""


@measure_time
def main():
    """
    Main script execution.
    """
    config = read_yaml_config("/Users/r.kanaki/code/inlabru_nbody/config/RS_Spatial_Selection_data_prep.yml")
    catalog_path = config["input_path"]
    output_paths = config["output_path"]
    nside = config["nside"]

    # Feature and target columns for plotting
    magnitude_features = [
        "des_asahi_full_g_abs_mag",
        "des_asahi_full_r_abs_mag",
        "des_asahi_full_i_abs_mag",
        "des_asahi_full_z_abs_mag",
        "des_asahi_full_y_abs_mag"
    ]
    redshift_target = "z_cgal"

    # Read the catalog
    data = read_catalog(catalog_path)

    # Create HEALPix map and plot density histogram
    healpix_map = create_healpix_map(data, nside)
    density_threshold = np.percentile(healpix_map, 90)
    plot_density_histogram(healpix_map, density_threshold, output_paths["plots"])

    # Stratify the data
    high_data, low_data, random_data = stratify_data(data, healpix_map, nside)

    # Save stratified datasets
    save_datasets(high_data, low_data, random_data, output_paths)

    # Plot magnitude distributions
    plot_magnitude_distributions(data, magnitude_features, output_paths["plots"], subset_name="all")
    plot_magnitude_distributions(high_data, magnitude_features, output_paths["plots"], subset_name="high_density")
    plot_magnitude_distributions(low_data, magnitude_features, output_paths["plots"], subset_name="low_density")
    plot_magnitude_distributions(random_data, magnitude_features, output_paths["plots"], subset_name="random")

    # Plot pairwise relationships
    plot_pairwise_relationships(data, magnitude_features, redshift_target, output_paths["plots"], subset_name="all")
    plot_pairwise_relationships(high_data, magnitude_features, redshift_target, output_paths["plots"], subset_name="high_density")
    plot_pairwise_relationships(low_data, magnitude_features, redshift_target, output_paths["plots"], subset_name="low_density")
    plot_pairwise_relationships(random_data, magnitude_features, redshift_target, output_paths["plots"], subset_name="random")

    # Plot true redshift distributions
    plot_true_redshift_distribution(data, redshift_target, output_paths["plots"], subset_name="all")
    plot_true_redshift_distribution(high_data, redshift_target, output_paths["plots"], subset_name="high_density")
    plot_true_redshift_distribution(low_data, redshift_target, output_paths["plots"], subset_name="low_density")
    plot_true_redshift_distribution(random_data, redshift_target, output_paths["plots"], subset_name="random")


if __name__ == "__main__":
    main()