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

use("Agg")


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
        print(
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper


def check_memory(threshold=0.8):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            total_memory = psutil.virtual_memory().total / (
                1024**2
            )  # Total system memory in MB

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
            print(
                f"[{func.__name__}] Memory after garbage collection: {memory_after_gc:.2f} MB"
            )

            if psutil.virtual_memory().percent / 100.0 > threshold:
                print(
                    f"[{func.__name__}] Memory usage exceeded {threshold * 100}%. Exiting."
                )
                sys.exit(1)

            return result

        return wrapper

    return decorator


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


@check_memory(threshold=0.8)
def read_yaml_config(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@check_memory(threshold=0.8)
def read_catalog(file_path, index_col=None, chunksize=10000):
    dtype_dict = {
        "id": "int64",
        "ra": "float64",
        "dec": "float64",
        "z": "float64",
        "z_v": "float64",
        "r_des_true": "float32",
        "r_des_true_error": "float32",
        "r_des_realization": "float32",
        "r_des_realization_error": "float32",
        "g_des_true": "float32",
        "g_des_true_error": "float32",
        "g_des_realization": "float32",
        "g_des_realization_error": "float32",
        "i_des_true": "float32",
        "i_des_true_error": "float32",
        "i_des_realization": "float32",
        "i_des_realization_error": "float32",
        "z_des_true": "float32",
        "z_des_true_error": "float32",
        "z_des_realization": "float32",
        "z_des_realization_error": "float32",
        "y_des_true": "float32",
        "y_des_true_error": "float32",
        "y_des_realization": "float32",
        "y_des_realization_error": "float32",
        "zb": "float32",
        "z_ml": "float32",
        "z_s": "float32",
    }
    chunks = []
    for chunk in pd.read_csv(
        file_path,
        sep=",",
        index_col=index_col,
        comment="#",
        na_values=r"\N",
        compression="bz2",
        chunksize=chunksize,
        dtype=dtype_dict,
    ):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def create_healpix_map(data, nside):
    npix = hp.nside2npix(nside)
    theta = np.radians(90.0 - data["dec"])
    phi = np.radians(data["ra"])
    pixel_indices = hp.ang2pix(nside, theta, phi)
    density_map = np.bincount(pixel_indices, minlength=npix)
    valid_ra_dec_mask = (
        (data["ra"] >= 0)
        & (data["ra"] <= 90)
        & (data["dec"] >= 0)
        & (data["dec"] <= 90)
    )
    valid_pixels = pixel_indices[valid_ra_dec_mask]
    valid_density_map = np.bincount(valid_pixels, minlength=npix)
    return valid_density_map


def plot_healpix_map(healpix_map, title, output_path):
    hp.mollview(healpix_map, title=title, unit="Galaxy Density", cmap="viridis")
    plt.savefig(output_path)
    plt.close()


def stratify_data(data, healpix_map, nside):
    magnitude_columns = [
        "g_des_true",
        "r_des_true",
        "i_des_true",
        "z_des_true",
        "y_des_true",
        "g_des_realization",
        "r_des_realization",
        "i_des_realization",
        "z_des_realization",
        "y_des_realization",
    ]
    magnitude_cut_mask = data["i_des_realization"] < 23.5
    data = data[magnitude_cut_mask]

    # Add error columns
    error_columns = [
        "g_des_true_error",
        "r_des_true_error",
        "i_des_true_error",
        "z_des_true_error",
        "y_des_true_error",
        "g_des_realization_error",
        "r_des_realization_error",
        "i_des_realization_error",
        "z_des_realization_error",
        "y_des_realization_error",
    ]

    valid_ra_dec_mask = (
        (data["ra"] >= 0)
        & (data["ra"] <= 90)
        & (data["dec"] >= 0)
        & (data["dec"] <= 90)
    )
    data_filtered = data[valid_ra_dec_mask]

    theta = np.radians(90.0 - data_filtered["dec"])
    phi = np.radians(data_filtered["ra"])
    pixel_indices = hp.ang2pix(nside, theta, phi)

    data_filtered = data_filtered.copy()
    data_filtered["pixel_index"] = pixel_indices

    filtered_healpix_map = healpix_map[healpix_map > 350]
    threshold = np.percentile(filtered_healpix_map, 95)
    print(f"Density threshold for high-density pixels: {threshold:.2f}")
    high_density_pixels = np.where(filtered_healpix_map > threshold)[0]
    low_density_pixels = np.where(filtered_healpix_map <= threshold)[0]

    necessary_columns = ["id", "ra", "dec", "z"] + magnitude_columns + error_columns

    high_density_data = data_filtered[
        data_filtered["pixel_index"].isin(high_density_pixels)
    ][necessary_columns]
    low_density_data = data_filtered[
        data_filtered["pixel_index"].isin(low_density_pixels)
    ][necessary_columns]

    min_size = min(len(high_density_data), len(low_density_data), len(data_filtered))
    print(f"High-density subset size: {len(high_density_data)}")
    print(f"Low-density subset size: {len(low_density_data)}")
    print(f"Random subset size: {len(data_filtered)}")
    print(f"Minimum size for each subset: {min_size}")

    high_density_data = high_density_data.sample(n=min_size, random_state=42)
    low_density_data = low_density_data.sample(n=min_size, random_state=42)
    random_sample_data = data_filtered.sample(n=min_size, random_state=42)[
        necessary_columns
    ]

    return high_density_data, low_density_data, random_sample_data


def save_datasets(high_data, low_data, random_data, output_paths):
    high_data.to_csv(output_paths["high"], index=False, compression="bz2")
    low_data.to_csv(output_paths["low"], index=False, compression="bz2")
    random_data.to_csv(output_paths["random"], index=False, compression="bz2")


def plot_magnitude_distributions_i(
    data,
    feature,
    output_path,
    subset_name="all",
    magnitude_cut=23.5,
    redshift_column="z",
):
    """
    Plot distribution of the observed i-band magnitude with a vertical line for the magnitude cut
    and optional redshift overlay.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        feature (str): The magnitude feature column to plot.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
        magnitude_cut (float): The magnitude cut value to highlight on the plot.
        redshift_column (str): Column name of the redshift to include in the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the magnitude distribution
    sns.histplot(
        data[feature],
        bins=50,
        kde=True,
        label=f"{feature} (i-band)",
        alpha=0.6,
        ax=ax1,
        color="blue",
    )
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Frequency", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Add a vertical line to indicate the magnitude cut
    ax1.axvline(
        magnitude_cut, color="red", linestyle="--", label=f"Mag. Cut ({magnitude_cut})"
    )
    ax1.legend(loc="upper left")

    # Title and grid
    plt.title(
        f"Distribution of {feature} Magnitude with {redshift_column} ({subset_name.capitalize()})"
    )
    plt.grid()

    # Save the plot
    plot_filename = f"{output_path}/magnitude_distributions_i_band_{feature}_{subset_name}_{redshift_column}.png"
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")


def plot_magnitude_distributions(
    data, features, output_path, subset_name="all", realization=False
):
    """
    Plot distributions of observed magnitudes for true and realization values.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        features (list): List of magnitude feature columns.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
        realization (bool): If True, uses realization magnitudes instead of true magnitudes.
    """
    # Select magnitude features based on whether it's realization or true
    selected_features = (
        [feature for feature in features if "realization" in feature]
        if realization
        else [feature for feature in features if "true" in feature]
    )

    # Plot distributions
    plt.figure(figsize=(10, 6))
    for feature in selected_features:
        sns.histplot(data[feature], bins=50, kde=True, label=feature, alpha=0.6)
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    realization_text = "Realization" if realization else "True"
    plt.title(
        f"Distribution of {realization_text} Magnitudes ({subset_name.capitalize()})"
    )
    plt.legend()
    plt.grid()

    # Save the plot
    realization_suffix = "_realization" if realization else "_true"
    plot_filename = (
        f"{output_path}/magnitude_distributions{realization_suffix}_{subset_name}.png"
    )
    plt.savefig(plot_filename)
    plt.close()
    print(f"Saved plot: {plot_filename}")


def plot_color_color_diagram(
    data, output_path, subset_name="all", redshift_column="z", realization=False
):
    """
    Plot color-color diagram of galaxy color indices (e.g., g-r, r-i) with a color bar for redshifts.
    Handles both true and realization values.

    Parameters:
        data (pd.DataFrame): The dataset containing galaxy magnitudes and redshifts.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
        redshift_column (str): Column name of the redshift to use for color-coding (e.g., "z", "z_v").
        realization (bool): If True, uses realization magnitudes instead of true magnitudes.
    """
    # Select magnitude columns based on whether it's realization or true
    g_col = "g_des_realization" if realization else "g_des_true"
    r_col = "r_des_realization" if realization else "r_des_true"
    i_col = "i_des_realization" if realization else "i_des_true"
    z_col = "z_des_realization" if realization else "z_des_true"
    y_col = "y_des_realization" if realization else "y_des_true"

    # Calculate color indices
    data["g-r"] = data[g_col] - data[r_col]
    data["r-i"] = data[r_col] - data[i_col]
    data["i-z"] = data[i_col] - data[z_col]
    data["z-y"] = data[z_col] - data[y_col]

    # Define color pairs to plot
    color_pairs = [("g-r", "r-i"), ("r-i", "i-z"), ("i-z", "z-y")]

    # Plot each color-color diagram
    for x_color, y_color in color_pairs:
        plt.figure(figsize=(8, 6))

        # Scatter plot with redshift as the color
        scatter = plt.scatter(
            x=data[x_color],
            y=data[y_color],
            c=data[redshift_column],  # Use specified redshift column for color mapping
            cmap="plasma",
            s=1,  # Point size
            alpha=0.7,
        )
        plt.xlabel(f"{x_color} Color Index")
        plt.ylabel(f"{y_color} Color Index")
        realization_text = "Realization" if realization else "True"
        plt.title(
            f"Color-Color Diagram ({x_color} vs {y_color}) - {realization_text} ({subset_name.capitalize()})"
        )
        plt.colorbar(scatter, label=f"Redshift ({redshift_column})")  # Add color bar
        plt.grid()

        # Save the plot
        realization_suffix = "_realization" if realization else "_true"
        plot_filename = f"{output_path}/color_color_diagram_{x_color}_vs_{y_color}_{subset_name}_{redshift_column}{realization_suffix}.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot: {plot_filename}")


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
    sns.histplot(data[target], bins=50, kde=None, color="blue", alpha=0.7)
    plt.xlabel("True Redshift (z_cgal)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of True Redshifts ({subset_name.capitalize()})")
    plt.grid()
    plt.savefig(f"{output_path}/true_redshifts_{subset_name}.png")
    plt.close()


def plot_density_histogram_squaredegree(healpix_map, threshold, output_path, nside=32):
    """
    Plot a normalized histogram of galaxy densities with high/low density boundaries.

    Parameters:
        healpix_map (np.ndarray): HEALPix map of galaxy density.
        threshold (float): Density threshold separating high/low density regions.
        output_path (str): Path to save the plot.
    """
    # Remove zero values for normalization
    # Calculate the area of a HEALPix pixel in square degrees
    npix = hp.nside2npix(nside)
    sky_area = 41253  # Total sky area in square degrees
    pixel_area = sky_area / npix  # Area of each pixel in square degrees
    # Convert galaxy density to galaxies per square degree
    healpix_map_deg2 = healpix_map / pixel_area * 256.0
    threshold = threshold / pixel_area * 256.0
    plt.figure(figsize=(10, 6))
    # Normalize the histogram by setting `stat="density"` in sns.histplot
    sns.histplot(
        healpix_map_deg2,
        bins=100,
        color="gray",
        alpha=0.7,
        label="Galaxy Density (per square degree)",
        stat="density",
    )
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"High/Low Density Threshold ({int(threshold)})",
    )
    plt.xlabel("Galaxy Density (per square degree)")
    plt.ylabel("Probability Density")
    plt.title("Normalized Galaxy Density Distribution with High/Low Boundary")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/density_histogram_normalized.png")
    plt.close()

def plot_density_histogram(healpix_map, threshold, output_path, nside=32):
    """
    Plot a histogram of galaxy densities with high/low density boundaries.
    Not in square degrees.

    Args:
        healpix_map (_type_): _description_
        threshold (_type_): _description_
        output_path (_type_): _description_
        nside (int, optional): _description_. Defaults to 32.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(
        healpix_map,
        bins=100,
        color="gray",
        alpha=0.7,
        label="Galaxy Density",
    )
    plt.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"High/Low Density Threshold ({int(threshold)})",
    )
    plt.xlabel("Galaxy Density")
    plt.ylabel("Frequency")
    plt.title("Galaxy Density Distribution with High/Low Boundary")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_path}/density_histogram.png")
    plt.close()

def plot_pairwise_true_realization(
    data, true_features, realization_features, target, output_path, subset_name="all"
):
    """
    Plot pairwise relationships for true and realization features along with the target,
    saving them as separate plots.

    Parameters:
        data (pd.DataFrame): The dataset to plot.
        true_features (list): List of true magnitude feature columns.
        realization_features (list): List of realization magnitude feature columns.
        target (str): Column name of the redshift target.
        output_path (str): Path to save the plot.
        subset_name (str): Name of the subset (e.g., "all", "high_density").
    """
    # Pairwise plot for true features
    sns.pairplot(
        data, vars=true_features + [target], diag_kind="kde", corner=True, height=2.5
    )
    plt.suptitle(
        f"Pairwise Relationships for True Features ({subset_name.capitalize()})", y=1.02
    )
    true_plot_file = f"{output_path}/pairwise_true_features_{subset_name}.png"
    plt.savefig(true_plot_file)
    plt.close()
    print(f"Saved pairwise plot for true features: {true_plot_file}")

    # Pairwise plot for realization features
    sns.pairplot(
        data,
        vars=realization_features + [target],
        diag_kind="kde",
        corner=True,
        height=2.5,
    )
    plt.suptitle(
        f"Pairwise Relationships for Realization Features ({subset_name.capitalize()})",
        y=1.02,
    )
    realization_plot_file = (
        f"{output_path}/pairwise_realization_features_{subset_name}.png"
    )
    plt.savefig(realization_plot_file)
    plt.close()
    print(f"Saved pairwise plot for realization features: {realization_plot_file}")


@measure_time
def main():
    config = read_yaml_config(
        "/Users/r.kanaki/code/inlabru_nbody/config/RS_micecat1_data_debug.yml"
    )
    # config = read_yaml_config("/Users/r.kanaki/code/inlabru_nbody/config/RS_micecat1_data_full_prep.yml")
    catalog_path = config["input_path"]
    output_paths = config["output_path"]
    nside = config["nside"]

    true_features = [
        "g_des_true",
        "r_des_true",
        "i_des_true",
        "z_des_true",
        "y_des_true",
    ]
    realization_features = [
        "g_des_realization",
        "r_des_realization",
        "i_des_realization",
        "z_des_realization",
        "y_des_realization",
    ]
    redshift_target = "z"

    data = read_catalog(catalog_path)
    healpix_map = create_healpix_map(data, nside)
    healpix_map_valid = healpix_map[healpix_map > 350]
    density_threshold = np.percentile(healpix_map_valid, 95)
    plot_density_histogram(healpix_map_valid, density_threshold, output_paths["plots"])
    plot_density_histogram_squaredegree(healpix_map_valid, density_threshold, output_paths["plots"])

    high_data, low_data, random_data = stratify_data(data, healpix_map_valid, nside)
    save_datasets(high_data, low_data, random_data, output_paths)

    ensure_directory_exists(output_paths["plots"])
    # Plot magnitude distributions for i-band
    plot_magnitude_distributions_i(
        data,
        "i_des_realization",
        output_paths["plots"],
        subset_name="all",
        magnitude_cut=23.5,
        redshift_column="z",
    )
    # Plot magnitude distributions
    plot_magnitude_distributions(
        data,
        true_features,
        output_paths["plots"],
        subset_name="all",
        realization=False,
    )
    plot_magnitude_distributions(
        high_data,
        true_features,
        output_paths["plots"],
        subset_name="high_density",
        realization=False,
    )
    plot_magnitude_distributions(
        low_data,
        true_features,
        output_paths["plots"],
        subset_name="low_density",
        realization=False,
    )
    plot_magnitude_distributions(
        random_data,
        true_features,
        output_paths["plots"],
        subset_name="random",
        realization=False,
    )
    plot_magnitude_distributions(
        data,
        realization_features,
        output_paths["plots"],
        subset_name="all",
        realization=True,
    )
    plot_magnitude_distributions(
        high_data,
        realization_features,
        output_paths["plots"],
        subset_name="high_density",
        realization=True,
    )
    plot_magnitude_distributions(
        low_data,
        realization_features,
        output_paths["plots"],
        subset_name="low_density",
        realization=True,
    )
    plot_magnitude_distributions(
        random_data,
        realization_features,
        output_paths["plots"],
        subset_name="random",
        realization=True,
    )

    # Plot pairwise relationships
    plot_pairwise_true_realization(
        data,
        true_features,
        realization_features,
        redshift_target,
        output_paths["plots"],
        subset_name="all",
    )
    plot_pairwise_true_realization(
        high_data,
        true_features,
        realization_features,
        redshift_target,
        output_paths["plots"],
        subset_name="high_density",
    )
    plot_pairwise_true_realization(
        low_data,
        true_features,
        realization_features,
        redshift_target,
        output_paths["plots"],
        subset_name="low_density",
    )
    plot_pairwise_true_realization(
        random_data,
        true_features,
        realization_features,
        redshift_target,
        output_paths["plots"],
        subset_name="random",
    )
    # Plot color-color diagrams for true values
    plot_color_color_diagram(
        data,
        output_path=output_paths["plots"],
        subset_name="all",
        redshift_column="z",
        realization=False,
    )
    plot_color_color_diagram(
        high_data,
        output_path=output_paths["plots"],
        subset_name="high_density",
        redshift_column="z",
        realization=False,
    )
    plot_color_color_diagram(
        low_data,
        output_path=output_paths["plots"],
        subset_name="low_density",
        redshift_column="z",
        realization=False,
    )
    plot_color_color_diagram(
        random_data,
        output_path=output_paths["plots"],
        subset_name="random",
        redshift_column="z",
        realization=False,
    )
    # Plot color-color diagrams for realization values
    plot_color_color_diagram(
        data,
        output_path=output_paths["plots"],
        subset_name="all",
        redshift_column="z",
        realization=True,
    )
    plot_color_color_diagram(
        high_data,
        output_path=output_paths["plots"],
        subset_name="High density",
        redshift_column="z",
        realization=True,
    )
    plot_color_color_diagram(
        low_data,
        output_path=output_paths["plots"],
        subset_name="Low density",
        redshift_column="z",
        realization=True,
    )
    plot_color_color_diagram(
        random_data,
        output_path=output_paths["plots"],
        subset_name="Random Sampling",
        redshift_column="z",
        realization=True,
    )
    # Plot redshift distributions
    plot_true_redshift_distribution(
        data, redshift_target, output_paths["plots"], subset_name="All"
    )
    plot_true_redshift_distribution(
        high_data, redshift_target, output_paths["plots"], subset_name="High density"
    )
    plot_true_redshift_distribution(
        low_data, redshift_target, output_paths["plots"], subset_name="Low density"
    )
    plot_true_redshift_distribution(
        random_data,
        redshift_target,
        output_paths["plots"],
        subset_name="Random Sampling",
    )


if __name__ == "__main__":
    main()
