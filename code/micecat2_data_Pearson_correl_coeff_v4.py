import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time
import psutil
import gc
from functools import wraps
import sys
from scipy.optimize import curve_fit
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving files
"""
Utility functions
"""


def measure_time(func):
    """
    Decorator to measure the execution time of a function.
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
            f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


# Decorator to monitor memory usage
def check_memory(threshold=0.8):
    """
    A decorator to check memory usage before and after a function call.
    Exits the program if memory usage exceeds the specified threshold.

    Parameters:
        threshold (float): The memory usage limit (e.g., 0.8 for 80%).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            total_memory = psutil.virtual_memory().total / (
                1024**2
            )  # Total system memory in MB

            # Print total system memory
            print(f"Total system memory: {total_memory:.2f} MB")

            # Memory usage percentage before the function call
            memory_before = process.memory_info().rss / (1024**2)  # In MB
            memory_percent_before = (memory_before / total_memory) * 100
            print(
                f"[{func.__name__}] Memory before execution: {memory_percent_before:.2f}%"
            )

            # Execute the original function
            result = func(*args, **kwargs)

            # Memory usage percentage after the function call
            memory_after = process.memory_info().rss / (1024**2)  # In MB
            memory_percent_after = (memory_after / total_memory) * 100
            print(
                f"[{func.__name__}] Memory after execution: {memory_percent_after:.2f}%"
            )

            # Force garbage collection to release memory
            gc.collect()
            memory_after_gc = process.memory_info().rss / (1024**2)  # In MB
            memory_percent_after_gc = (memory_after_gc / total_memory) * 100
            print(
                f"[{func.__name__}] Memory after garbage collection: {memory_percent_after_gc:.2f}%"
            )

            # Exit the program if memory usage exceeds the threshold
            if psutil.virtual_memory().percent / 100.0 > threshold:
                print(
                    f"[{func.__name__}] Memory usage exceeded {threshold * 100}%. Exiting."
                )
                sys.exit(1)

            return result

        return wrapper

    return decorator


"""
Part 1: Read the input data
"""


@check_memory(threshold=0.8)
def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a Python dictionary.
    Parameters:
        file_path (str): The path to the YAML file to be read.
    Returns:
        dict: The contents of the YAML file as a dictionary.
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


@check_memory(threshold=0.8)
def read_pandas(catalog_filename, index_col=None, chunksize=10000):
    """
    Reads a CSV file into a pandas DataFrame in chunks.
    The code is adapted from the following source: https://cosmohub.pic.es/help
    parameters:
        catalog_filename (str): The path to the CSV file to be read.
        index_col (str): The name of the column to be used as the index.
        chunksize (int): The number of rows to read at a time.
    Returns:
        chunk: The chunk of the DataFrame
    """
    dtype_dict = {
        "col1": "int32",
        "col2": "float32",
    }  # Define appropriate types for each column
    for chunk in pd.read_csv(
        catalog_filename,
        sep=",",
        index_col=index_col,
        comment="#",
        na_values=r"\N",
        compression="bz2",
        chunksize=chunksize,
        dtype=dtype_dict,
    ):
        # print(chunk.head())  # Process or use data here
        return chunk


"""
Part 2: Split the data into redshift bins and generate the Healpix maps
"""


def split_data(data_gal, z, num_bins):
    """
    Splits the input data into redshift bins.
    Parameters:
        data_gal (DataFrame): The input data.
        z (array): The redshift values.
        num_bins (int): The number of redshift bins.
    Returns:
        binned_data (dict): A dictionary containing the data split into redshift bins.
    """
    z_bins = pd.cut(z, bins=num_bins, labels=False)

    # Split the data for each redshift bin
    binned_data = {}
    for i in range(num_bins):
        binned_data[i] = data_gal[z_bins == i]
        # print(f"Bin {i}: {len(binned_data[i])} entries")
    return binned_data


def Healpix_map(binned_data, bin_num, nside=64):
    """
    Generates a HEALPix map of the number of galaxies in each pixel for a given redshift bin.
    Parameters:
        binned_data (dict): A dictionary containing the data split into redshift bins.
        bin_num (int): The redshift bin number.
        nside (int): The HEALPix resolution parameter.
    Returns:
        Healpix_map (array): A HEALPix map of the number of galaxies in each pixel.
    """
    ra = binned_data[bin_num]["ra_gal"]
    dec = binned_data[bin_num]["dec_gal"]
    # Convert the ra and dec to theta and phi
    theta = np.pi / 2 - np.deg2rad(dec)
    phi = np.deg2rad(ra)

    # Get the pixel number for each ra, dec pair
    pixels = hp.ang2pix(nside, theta, phi)

    # Create a HEALPix map with the number of galaxies in each pixel
    Healpix_map = np.bincount(pixels, minlength=hp.nside2npix(nside))
    return Healpix_map


@check_memory(threshold=0.8)
def Healpix_map_z_cgal(data, nsides):
    """
    Generates a HEALPix map where each pixel contains the average redshift of galaxies within that pixel.
    Parameters:
        data (DataFrame): The input data containing 'ra_gal', 'dec_gal', and 'z_cgal'.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
    Returns:
        Healpix_map (array): A HEALPix map with the average redshift of galaxies in each pixel.
    """
    Healpix_maps = {}
    ra = data["ra_gal"]
    dec = data["dec_gal"]
    z_cgal = data["z_cgal"]

    # Convert the ra and dec to theta and phi
    theta = np.pi / 2 - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    for nside in nsides:
        # Create a HEALPix map with the average redshift of galaxies in each pixel
        # Get the pixel number for each ra, dec pair
        pixels = hp.ang2pix(nside, theta, phi)
        # Create an array to store the sum of redshifts and the count of galaxies in each pixel
        sum_z = np.zeros(hp.nside2npix(nside))
        count = np.zeros(hp.nside2npix(nside))
        # Sum the redshifts and count the number of galaxies in each pixel
        np.add.at(sum_z, pixels, z_cgal)
        np.add.at(count, pixels, 1)
        # Calculate the average redshift for each pixel
        avg_z = np.zeros_like(sum_z)
        nonzero_pixels = count > 0
        avg_z[nonzero_pixels] = sum_z[nonzero_pixels] / count[nonzero_pixels]
        Healpix_maps[nside] = avg_z  # Store the HEALPix map for the given nside
        # Clear the memory to free up space
        del pixels, sum_z, count, avg_z
        gc.collect()
    return Healpix_maps


def binned_Healpix_maps(binned_data, num_bins, nsides):
    """
    Generates HEALPix maps for each redshift bin.
    Parameters:
        binned_data (dict): A dictionary containing the data split into redshift bins.
        num_bins (int): The number of redshift bins.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
    Returns:
        Healpix_maps (dict): A dictionary containing the HEALPix maps for each redshift bin.
    """
    Healpix_maps = {}
    for i in range(num_bins):
        Healpix_maps[i] = {}  # Initialize a dictionary for each redshift bin
        for nside in nsides:
            print(f"Processing bin {i} with nside {nside}")
            Healpix_maps[i][nside] = Healpix_map(binned_data, i, nside)
            print("Healpix map max: ", np.max(Healpix_maps[i][nside]))
            print("Healpix map min: ", np.min(Healpix_maps[i][nside]))
    return Healpix_maps


"""
Part 3: Calculate the Statistical Values(Mean, Variance, Pearson correlation coefficient) of input data
"""


def calculate_mean_variance(healpix_map):
    """
    Calculate the mean and variance of the given map data.

    Parameters:
    map_data (numpy.ndarray): The map data.
    nside (int): The nside parameter for HEALPix.

    Returns:
    tuple: mean and variance of the map data.
    """
    mean = np.mean(healpix_map[healpix_map > 0])
    variance = np.var(healpix_map[healpix_map > 0])
    return mean, variance

def calculate_dispersion(healpix_map, nsides):
    """
    Calculate the dispersion statistic (variance-to-mean ratio) for each nside.

    Parameters:
    healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
    nsides (list): List of nside values.

    Returns:
    numpy.ndarray: Array of dispersion statistics for each nside.
    """
    dispersion_stats = []
    
    for nside in nsides:
        print(f"Calculating dispersion statistic for nside: {nside}")
        map_data = healpix_map[nside]
        mean_value = np.mean(map_data[map_data > 0])
        variance_value = np.var(map_data[map_data > 0])
        
        # Calculate the dispersion statistic (variance-to-mean ratio)
        dispersion = variance_value / mean_value
        print(f"Dispersion statistic (Var/Mean): {dispersion:.4f}")
        dispersion_stats.append(dispersion)

    return np.array(dispersion_stats)

def get_pixel_neighbor_pairs(healpix_map, nside):
    """
    Generate pairs of pixel values and their neighboring pixel values.

    Parameters:
    healpix_map (numpy.ndarray): The HEALPix map data.
    nside (int): The nside parameter for HEALPix.

    Returns:
    numpy.ndarray: Array of pairs of pixel values and their neighboring pixel values.
    """
    npix = hp.nside2npix(nside)
    neighbors = hp.get_all_neighbours(nside, np.arange(npix))

    pairs = []
    for i in range(npix):
        pixel_value = healpix_map[i]
        if pixel_value == 0:  # Skip empty pixels
            continue
        neighbor_values = healpix_map[neighbors[:, i]]
        valid_neighbors = neighbor_values[
            neighbor_values > 0
        ]  # Filter out invalid neighbors

        for neighbor_value in valid_neighbors:
            pairs.append([pixel_value, neighbor_value])
    return np.array(pairs)


def calculate_pearson_correlation(pairs):
    """
    Calculate the Pearson correlation coefficient using pairs of pixel values.

    Parameters:
    pairs (numpy.ndarray): Array of pairs of pixel values and their neighboring pixel values.

    Returns:
    float: Pearson correlation coefficient.
    """
    if len(pairs) == 0:  # Check if there are no valid pairs
        return np.nan

    pixel_values = pairs[:, 0]
    neighbor_values = pairs[:, 1]
    correlation = np.corrcoef(pixel_values, neighbor_values)[0, 1]
    return correlation


def make_list_Pearson_correlation(healpix_map, nsides, output_filename):
    """
    Calculate the Pearson correlation coefficient for each HEALPix map.

    Parameters:
    healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
    nsides (list): A list of HEALPix resolution parameters for each redshift bin.

    Returns:
    numpy.ndarray: Array of Pearson correlation coefficients for each HEALPix map.
    """
    correlations = []
    for nside in nsides:
        print(f"Calculating the Pearson correlation coefficient for nside: {nside}")
        correlation_bin = []
        for bin_num in range(len(healpix_map)):
            print(f"Calculating the Pearson correlation coefficient for bin: {bin_num}")
            Healpix_map_i = healpix_map[bin_num][nside]
            pairs = get_pixel_neighbor_pairs(Healpix_map_i, nside)
            print(f"Number of pairs: {len(pairs)}")
            print(f"pairs: {pairs}")
            visualize_pairs(pairs, nside, bin_num, output_filename)
            correlation = calculate_pearson_correlation(pairs)
            print(f"Pearson correlation coefficient: {correlation}")
            correlation_bin.append(correlation)
        correlations.append(correlation_bin)
    correlations = np.array(correlations)
    return correlations


@check_memory(threshold=0.8)
def make_list_Pearson_correlation_zgal(healpix_map, nsides, output_filename):
    """
    Calculate the Pearson correlation coefficient for each HEALPix map.

    Parameters:
    healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
    nsides (list): A list of HEALPix resolution parameters for each redshift bin.
    output_filename (str): The path to the output file.

    Returns:
    numpy.ndarray: Array of Pearson correlation coefficients for each HEALPix map.
    """
    correlations = []
    for nside in nsides:
        print(f"Calculating the Pearson correlation coefficient for nside: {nside}")
        Healpix_map_i = healpix_map[nside]
        pairs = get_pixel_neighbor_pairs(Healpix_map_i, nside)
        visualize_pairs_cgal(pairs, nside, output_filename)
        print(f"Number of pairs: {len(pairs)}")
        print(f"pairs: {pairs}")
        # visualize_pairs(pairs, nside, bin_num, output_filename)
        correlation = calculate_pearson_correlation(pairs)
        print(f"Pearson correlation coefficient: {correlation}")
        correlations.append(correlation)
    correlations_np = np.array(correlations)
    # free up memory
    del pairs, Healpix_map_i, correlations, correlation
    gc.collect()
    return correlations_np


def exponential_model(r, r0, gamma):
    """
    Power law model function: g(r) = 1 + (r / r0)^{-gamma}
    """
    return (r / r0) ** (-gamma)


@check_memory(threshold=0.8)
def fit_power_law(pixel_areas, correlations):
    """
    Fit the Pearson correlation data to the power law model.

    Parameters:
    pixel_areas (numpy.ndarray): Array of pixel areas in square degrees.
    correlations (numpy.ndarray): Array of Pearson correlation coefficients.

    Returns:
    tuple: (r0, gamma)
        - r0 (float): The correlation length.
        - gamma (float): The power law exponent.
    """
    # Convert pixel area to approximate angular separation
    separations = np.sqrt(pixel_areas)

    # Filter out NaN values
    valid_indices = ~np.isnan(correlations)
    separations = separations[valid_indices]
    correlations = correlations[valid_indices]

    # Initial guesses for r0 and gamma
    initial_guess = [0.1, 1.0]

    # Fit the power law model
    try:
        popt, pcov = curve_fit(
            exponential_model, separations, correlations, p0=initial_guess
        )
        r0, gamma = popt
        print(f"Fitted parameters: r0 = {r0:.4f}, gamma = {gamma:.4f}")
    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        return None, None

    return r0, gamma


"""
Part 4: Visualize the results
"""


def visualize_healpix(healpix_map, nsides, output_filename):
    """
    Visualizes a HEALPix map.
    Parameters:
        healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
        output_filename (str): The path to the output file.
    """
    # print(f"Length of healpix_map: {len(healpix_map)}")
    # print(f"nsides: {nsides}")
    output_filename = os.path.join(
        output_filename, "Healpix_maps/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)
    for i in range(len(healpix_map)):
        print(f"Processing galaxy map bin {i}")
        print(
            f"healpix_map[{i}] keys: {list(healpix_map[i].keys())}"
        )  # Check the keys in the map
        for nside in nsides:
            print(f"Processing galaxy map nside {nside}")
            plt.figure()
            print(f"healpix map max: {np.max(healpix_map[i][nside])}")
            print(f"healpix map min: {np.min(healpix_map[i][nside])}")
            hp.mollview(
                healpix_map[i][nside],
                title=f"HEALPix Map for Bin {i} (nside={nside})",
                cmap="viridis",
                min=np.amin(healpix_map[i][nside]),
                max=np.amax(healpix_map[i][nside]),
            )
            hp.graticule()
            plt.show()
            plt.savefig(f"{output_filename}galaxy_counts_map_bin_{i}_nside_{nside}.png")
            plt.close()


def visualize_healpix_gnom_view(
    healpix_map, nsides, output_filename, ra_min=0, ra_max=10, dec_min=0, dec_max=10
):
    """
    Visualizes a HEALPix map with a zoomed-in region of interest based on RA and Dec range.
    Parameters:
        healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
        output_filename (str): The path to the output file.
        ra_min (float): Minimum right ascension (RA) for the region of interest.
        ra_max (float): Maximum right ascension (RA) for the region of interest.
        dec_min (float): Minimum declination (Dec) for the region of interest.
        dec_max (float): Maximum declination (Dec) for the region of interest.
    """
    output_filename = os.path.join(output_filename, "Healpix_maps_gnomview/")
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)

    # Calculate the center and size of the RA-Dec box
    central_ra = (ra_min + ra_max) / 2
    central_dec = (dec_min + dec_max) / 2
    dec_extent = dec_max - dec_min  # Angular size in degrees
    ra_extent = ra_max - ra_min  # Angular size in degrees

    # Convert RA-Dec to the coordinates expected by gnomview (rot)
    central_lon = central_ra  # In degrees, directly used as lon
    central_lat = central_dec  # In degrees, directly used as lat
    xsize = int(ra_extent * 7.5)  # Convert angular size to pixel size (rough estimate)

    for i in range(len(healpix_map)):
        print(f"Processing galaxy map bin {i}")
        for nside in nsides:
            print(f"Processing galaxy map nside {nside}")
            plt.figure()
            print(f"healpix map max: {np.max(healpix_map[i][nside])}")
            print(f"healpix map min: {np.min(healpix_map[i][nside])}")
            hp.gnomview(
                healpix_map[i][nside],
                rot=(central_lon, central_lat),
                xsize=xsize,  # Controls the zoom level based on RA extent
                reso=dec_extent,  # Adjust the resolution/zoom according to the declination range
                title=f"HEALPix Map for Bin {i} (nside={nside}, RA {ra_min}-{ra_max}, Dec {dec_min}-{dec_max})",
                cmap="viridis",
                min=np.amin(healpix_map[i][nside]),
                max=np.amax(healpix_map[i][nside]),
            )
            hp.graticule()
            plt.show()
            plt.savefig(
                f"{output_filename}galaxy_counts_map_bin_{i}_nside_{nside}_RA_{ra_min}-{ra_max}_Dec_{dec_min}-{dec_max}.png"
            )
            plt.close()


def plot_histogram_galaxies(healpix_map, nsides, output_filename):
    """
    Plots a histogram of the number of galaxies in each HEALPix pixel.
    Parameters:
        healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
        output_filename (str): The path to the output file.
    """
    output_filename = os.path.join(
        output_filename, "Histogram_counts/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)
    for i in range(len(healpix_map)):
        print(f"Processing Histogram bin {i}")
        print(
            f"healpix_map[{i}] keys: {list(healpix_map[i].keys())}"
        )  # Check the keys in the map
        for nside in nsides:
            print(f"Processing Histogram nside {nside}")
            plt.figure(figsize=(10, 6))
            mean_count = np.mean(healpix_map[i][nside][healpix_map[i][nside] > 0])
            variance_count = np.var(healpix_map[i][nside][healpix_map[i][nside] > 0])
            plt.axvline(
                mean_count + np.sqrt(variance_count),
                color="green",
                linestyle="dashed",
                linewidth=1.5,
                label=f"$\sigma$: {np.sqrt(variance_count):.2f}",
            )
            plt.axvline(
                mean_count - np.sqrt(variance_count),
                color="green",
                linestyle="dashed",
                linewidth=1.5,
            )
            plt.axvline(
                mean_count,
                color="red",
                linestyle="dashed",
                linewidth=1.5,
                label=f"$\mu$: {mean_count:.2f}",
            )
            plt.legend()
            plt.hist(
                healpix_map[i][nside][healpix_map[i][nside] > 0],
                bins=30,
                color="blue",
                alpha=0.7,
                edgecolor="black",
                linewidth=1.2,
            )
            plt.xlabel("Counts per Healpix Pixel")
            plt.ylabel("Frequency")
            plt.title("Histogram of Counts per Healpix Pixel")
            plt.grid(True)
            plt.show()
            plt.savefig(
                f"{output_filename}histoghram_galaxy_counts_bin_{i}_nside_{nside}.png"
            )
            plt.close()


def visulaize_mean_variance(healpix_map, nsides, output_filename):
    """
    Visualizes the mean and variance of the galaxy counts in each HEALPix pixel.
    Parameters:
        healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
        output_filename (str): The path to the output file.
    """
    output_filename = os.path.join(
        output_filename, "mean_and_variance/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    mean_values = []
    error_values = []

    for nside in nsides:
        print(f"Processing nside {nside}")
        mean_values_nside = []
        error_values_nside = []
        for i in range(len(healpix_map)):
            mean, variance = calculate_mean_variance(healpix_map[i][nside])
            mean_values_nside.append(mean)
            error_values_nside.append(variance)
        mean_values.append(mean_values_nside)
        error_values.append(error_values_nside)
    means = np.array(mean_values)
    errors = np.array(error_values)
    plt.figure(figsize=(10, 6))
    for bin_num in range(len(healpix_map)):
        plt.errorbar(
            nsides,
            means[:, bin_num],
            yerr=np.sqrt(errors[:, bin_num]),
            label=f"Bin {bin_num}",
            capsize=5,
        )
    plt.xlabel("Nside")
    plt.ylabel("Mean Galaxy Count")
    plt.title("Mean Galaxy Count with Error Bars for Different Redshift Bins")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}mean_galaxy_count.png")
    plt.close()


def visualize_mean_variance_zgal(healpix_map, nsides, output_filename):
    """
    Visualizes the mean and variance of the galaxy counts in each HEALPix pixel.
    Parameters:
        healpix_map (dict): A dictionary containing the HEALPix maps for each redshift bin.
        nsides (list): A list of HEALPix resolution parameters for each redshift bin.
        output_filename (str): The path to the output file.
    """
    output_filename = os.path.join(
        output_filename, "mean_and_variance_zgal/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    mean_values = []
    error_values = []

    for nside in nsides:
        print(f"Processing nside {nside}")
        mean, variance = calculate_mean_variance(healpix_map[nside])
        mean_values.append(mean)
        error_values.append(
            np.sqrt(variance)
        )  # standard deviation as error not variance
    means = np.array(mean_values)
    errors = np.array(error_values)
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        nsides,
        means,
        yerr=np.sqrt(errors),
        capsize=5,
    )
    plt.xlabel("Nside")
    plt.ylabel("Mean Redshift")
    plt.title("Mean Redshift with Error Bars for z_cgal")
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}mean_redshift_zgal.png")
    plt.close()


def visualize_pearson_correlation_vs_nside(
    correlations, nsides, num_bins, output_filename
):
    """
    Visualize the Pearson correlation coefficient against nside and compare the curve across multiple redshift bins.

    Parameters:
    correlations (numpy.ndarray): Array of Pearson correlation coefficients for different nsides and redshift bins.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Peasron_correlation_nside/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)
    plt.figure(figsize=(10, 6))
    for bin_num in range(num_bins):
        plt.plot(nsides, correlations[:, bin_num], label=f"Bin {bin_num}")

    plt.xlabel("Nside")
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation vs Nside for Different Redshift Bins")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}Pearson_correlation_vs_nside.png")
    plt.close()


def visualize_pearson_correlation_vs_arcdegree(
    correlations, nsides, num_bins, output_filename
):
    """
    Visualize the Pearson correlation coefficient against arcdegree and compare the curve across multiple redshift bins.

    Parameters:
    correlations (numpy.ndarray): Array of Pearson correlation coefficients for different nsides and redshift bins.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Peasron_correlation_arcdegree/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)
    arcdegree = [hp.nside2resol(nside, arcmin=True) / 60 for nside in nsides]
    plt.figure(figsize=(10, 6))
    for bin_num in range(num_bins):
        plt.plot(arcdegree, correlations[:, bin_num], label=f"Bin {bin_num}")

    plt.xlabel("Arcdegree")
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation vs Arcdegree for Different Redshift Bins")
    plt.legend()
    plt.grid(True)
    # plt.xscale('log')
    plt.show()
    plt.savefig(f"{output_filename}Pearson_correlation_vs_arcdegree.png")
    plt.close()


def visualize_pairs(pairs, nside, num_bins, output_filename):
    """
    Visualize the pairs of pixel values and their neighboring pixel values.

    Parameters:
    pairs (numpy.ndarray): Array of pairs of pixel values and their neighboring pixel values.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Pairs/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    plt.figure(figsize=(10, 6))
    plt.scatter(pairs[:, 0], pairs[:, 1], s=1, label=f"Bin {num_bins} nside {nside}")
    plt.xlabel(r"Galaxy counts, $N_i$ (within each pixel)")
    plt.ylabel(r"Galaxy counts, $N_j$ (in neighboring pixels)")
    plt.title(
        f"Pairs of galaxy counts and Their Neighboring Mean Redshifts in nside={nside} and Bin {num_bins}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f"{output_filename}{nside}_{num_bins}pairs.png")
    plt.close()


def visualize_pearson_correlation_vs_square_degree(
    correlations, nsides, num_bins, output_filename
):
    """
    Visualize the Pearson correlation coefficient against the pixel area in square degrees
    and compare the curve across multiple redshift bins.

    Parameters:
    correlations (numpy.ndarray): Array of Pearson correlation coefficients for different nsides and redshift bins.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Pearson_correlation_square_degree/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    # Calculate pixel area in square degrees for each nside
    pixel_area_sqdeg = [hp.nside2pixarea(nside, degrees=True) for nside in nsides]

    plt.figure(figsize=(10, 6))
    for bin_num in range(num_bins):
        plt.plot(pixel_area_sqdeg, correlations[:, bin_num], label=f"Bin {bin_num}")

    plt.xlabel("Pixel Area (Square Degrees)")
    plt.ylabel("Pearson Correlation")
    plt.title(
        "Pearson Correlation vs Pixel Area (Square Degrees) for Different Redshift Bins"
    )
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}Pearson_correlation_vs_square_degree.png")
    plt.close()


def visualize_pairs_cgal(pairs, nside, output_filename):
    """
    Visualize the pairs of pixel values and their neighboring pixel values.

    Parameters:
    pairs (numpy.ndarray): Array of pairs of pixel values and their neighboring pixel values.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Pairs_cgal/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    plt.figure(figsize=(10, 6))
    plt.scatter(pairs[:, 0], pairs[:, 1], s=1, label=f"nside {nside}")
    plt.xlabel(r"Mean Redshift, $z_i$ (within each pixel)")
    plt.ylabel(r"Mean Redshift, $z_j$ (in neighboring pixels)")
    plt.title(
        f"Pairs of Mean Redshifts and Their Neighboring Mean Redshifts in nside={nside}"
    )
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(f"{output_filename}{nside}pairs.png")
    plt.close()


def visualize_healpix_map_z_cgal(healpix_maps, nsides, output_filename):
    """
    Visualizes a HEALPix map of average redshift using mollview.
    Parameters:
        healpix_map (array): The HEALPix map of average redshift.
        nsides (list): List of nside values.
        output_filename (str): The path to the output file.
    """
    output_filename = os.path.join(output_filename, "Healpix_map_z_cgal/")
    if not os.path.exists(output_filename):
        os.makedirs(output_filename)
    for nside in nsides:
        healpix_maps_nside = healpix_maps[nside]
        plt.figure()
        hp.mollview(
            healpix_maps_nside,
            title=f"HEALPix Map of Average Redshift (nside={nside})",
            cmap="viridis",
            min=np.amin(healpix_maps_nside),
            max=np.amax(healpix_maps_nside),
            unit="Average Redshift",
        )
        hp.graticule()
        plt.show()
        plt.savefig(f"{output_filename}healpix_map_z_cgal_nside_{nside}.png")
        plt.close()


def visualize_pearson_correlation_vs_square_degree_zgal(
    correlations, nsides, output_filename
):
    """
    Visualize the Pearson correlation coefficient against the pixel area in square degrees
    and compare the curve across multiple redshift bins.

    Parameters:
    correlations (numpy.ndarray): Array of Pearson correlation coefficients for different nsides and redshift bins.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Pearson_correlation_square_degree_zgal/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    # Calculate pixel area in square degrees for each nside
    pixel_area_sqdeg = [hp.nside2pixarea(nside, degrees=True) for nside in nsides]

    plt.figure(figsize=(10, 6))
    plt.plot(pixel_area_sqdeg, correlations)
    plt.xlabel("Pixel Area (Square Degrees)")
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation vs Pixel Area (Square Degrees) for z_cgal")
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}Pearson_correlation_vs_square_degree_zgal.png")
    plt.close()


def visualize_pearson_correlation_vs_nsides_zgal(correlations, nsides, output_filename):
    """
    Visualize the Pearson correlation coefficient against the pixel area in square degrees
    and compare the curve across multiple redshift bins.

    Parameters:
    correlations (numpy.ndarray): Array of Pearson correlation coefficients for different nsides and redshift bins.
    nsides (list): List of nside values.
    num_bins (int): Number of redshift bins.
    """
    output_filename = os.path.join(
        output_filename, "Pearson_correlation_nsides_zgal/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)

    plt.figure(figsize=(10, 6))
    plt.plot(nsides, correlations)
    plt.xlabel("Nside")
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation vs Nside for z_cgal")
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_filename}Pearson_correlation_vs_nsides_zgal.png")
    plt.close()


def plot_fitted_correlation(pixel_areas, correlations, r0, gamma, output_filename):
    """
    Plot the Pearson correlation coefficients and the fitted power law model.

    Parameters:
    pixel_areas (numpy.ndarray): Array of pixel areas in square degrees.
    correlations (numpy.ndarray): Array of Pearson correlation coefficients.
    r0 (float): Fitted correlation length.
    gamma (float): Fitted power law exponent.
    """
    separations = np.sqrt(pixel_areas)
    fitted_correlations = exponential_model(separations, r0, gamma)

    plt.figure(figsize=(10, 6))
    plt.scatter(separations, correlations, color="blue", label="Observed Correlations")
    plt.plot(
        separations,
        fitted_correlations,
        color="red",
        linestyle="--",
        label=f"Fitted Model: $r_0={r0:.4f}$, $\\gamma={gamma:.4f}$",
    )
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel("Angular Separation (radians)")
    plt.ylabel("Pearson Correlation Coefficient")
    plt.title("Fitted Pearson Correlation Coefficient vs Angular Separation")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_filename}fitted_pearson_correlation.png")
    plt.show()
    plt.close()

def visualize_dispersion_test_vs_degree(dispersion_stats, nsides, output_filename):
    """
    Visualize the dispersion statistic (variance-to-mean ratio) against angular scale (in degrees).

    Parameters:
    dispersion_stats (numpy.ndarray): Array of dispersion statistics for each nside.
    nsides (list): List of nside values.
    output_filename (str): Path to save the output plot.
    """
    output_directory = os.path.join(output_filename, "Dispersion_Test/")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Calculate the angular scale (in degrees) as the square root of the pixel area
    pixel_area_sqdeg = [hp.nside2pixarea(nside, degrees=True) for nside in nsides]
    angular_scale_deg = np.sqrt(pixel_area_sqdeg)

    plt.figure(figsize=(10, 6))
    plt.plot(angular_scale_deg, dispersion_stats, marker='o', linestyle='-', color='blue')
    #plt.axhline(y=1, color='red', linestyle='--', label='Poisson Process (Var/Mean = 1)')
    plt.xlabel("Angular Scale (Degrees)")
    plt.ylabel("Variance-to-Mean Ratio (Dispersion Statistic)")
    plt.title("Dispersion Test: Variance-to-Mean Ratio vs Angular Scale")
    plt.legend()
    plt.grid(True)
    plt.xscale("log")
    plt.show()
    plt.savefig(f"{output_directory}dispersion_test_vs_degree.png")
    plt.close()

"""
Main function
"""


@check_memory(threshold=0.8)
def main():
    """
    The main function of the script.
    """
    # Part 1: Read the input data
    plt.ioff()
    yaml_file = "/Users/r.kanaki/code/inlabru_nbody/config/micecat2_data_Pearson_correl_coeff_100.yml"
    config = read_yaml(yaml_file)
    catalog_filename = config["data"]["input_file"]
    output_filename = config["data"]["output_file"]
    nsides = config["parameters"]["nsides"]

    data_gal = measure_time(read_pandas)(
        catalog_filename, ["unique_gal_id"], chunksize=200_000_000
    )
    z = data_gal["z_cgal"]

    # Part 2: Generate the Healpix maps
    Healpix_maps_zgal = measure_time(Healpix_map_z_cgal)(data_gal, nsides)
    #visualize_healpix_map_z_cgal(Healpix_maps_zgal, nsides, output_filename)
    del data_gal, z
    gc.collect()

    # Part 3: Calculate the Pearson correlation coefficient
    # measure_time(visualize_mean_variance_zgal)(Healpix_maps_zgal, nsides, output_filename)
    dispersion_stats = measure_time(calculate_dispersion)(Healpix_maps_zgal, nsides)
    measure_time(visualize_dispersion_test_vs_degree)(dispersion_stats, nsides, output_filename)
    Pearson_correlation_zgal = measure_time(make_list_Pearson_correlation_zgal)(Healpix_maps_zgal, nsides, output_filename)

    # Part 4: Fit the Power Law Model
    pixel_area_sqdeg = [hp.nside2pixarea(nside, degrees=True) for nside in nsides]
    r0, gamma = fit_power_law(pixel_area_sqdeg, Pearson_correlation_zgal)

    # Part 5: Visualize the Fitted Model
    # if r0 is not None and gamma is not None:
    plot_fitted_correlation(pixel_area_sqdeg, Pearson_correlation_zgal, r0, gamma, output_filename)

    # Part 6: Additional Visualizations
    measure_time(visualize_pearson_correlation_vs_square_degree_zgal)(Pearson_correlation_zgal, nsides, output_filename)
    measure_time(visualize_pearson_correlation_vs_nsides_zgal)(Pearson_correlation_zgal, nsides, output_filename)
    




if __name__ == "__main__":
    main = measure_time(main)
    main()