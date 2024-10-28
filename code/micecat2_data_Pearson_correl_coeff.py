import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import time

"""
Part 1: Read the input data
"""


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
    plt.xlabel("Pairs")
    plt.ylabel("Pixel Value")
    plt.title("Pairs of Pixel Values and Their Neighboring Pixel Values")
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


"""
Main function
"""


def main():
    """
    The main function of the script.
    """
    # Part 1: Read the input data

    # Read the YAML file
    yaml_file = (
        "/home/rintaro/code/inlabru_nbody/config/micecat2_data_Pearson_correl_coeff.yml"
    )
    config = read_yaml(yaml_file)
    # Read the input and output filenames from the YAML file
    catalog_filename = config["data"]["input_file"]
    output_filename = config["data"]["output_file"]
    # Read the parameters from the YAML file
    nsides = config["parameters"]["nsides"]
    print("nsides", nsides)
    num_bins = config["parameters"]["num_bins"]
    # Read the data from the input file
    data_gal = measure_time(read_pandas)(
        catalog_filename, ["unique_gal_id"], chunksize=200_000_000
    )
    z = data_gal["z_cgal"]  # Spectroscopic redshift

    # Part 2: Split the data into redshift bins and generate the Healpix maps
    # Split the data into redshift bins
    binned_data = measure_time(split_data)(data_gal, z, num_bins)
    # Generate the HEALPix maps for each redshift bin
    Healpix_maps = measure_time(binned_Healpix_maps)(binned_data, num_bins, nsides)
    # Part 3: Calculate the Statistical Values(Mean, Variance, Pearson correlation coefficient) of input data
    # measure_time(visulaize_mean_variance)(Healpix_maps, nsides, output_filename)
    # measure_time(visualize_healpix)(Healpix_maps, nsides, output_filename)
    # measure_time(visualize_healpix_gnom_view)(Healpix_maps, nsides, output_filename)
    # measure_time(plot_histogram_galaxies)(Healpix_maps, nsides, output_filename)
    # Calculate the Pearson correlation coefficient for each HEALPix map
    Pearson_correlation = measure_time(make_list_Pearson_correlation)(
        Healpix_maps, nsides, output_filename
    )
    # Part 4: Visualize the results
    measure_time(visualize_pearson_correlation_vs_nside)(
        Pearson_correlation, nsides, num_bins, output_filename
    )
    measure_time(visualize_pearson_correlation_vs_arcdegree)(
        Pearson_correlation, nsides, num_bins, output_filename
    )
    measure_time(visualize_pearson_correlation_vs_square_degree)(
        Pearson_correlation, nsides, num_bins, output_filename
    )


if __name__ == "__main__":
    main = measure_time(main)
    main()
