import yaml
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp

"""
Part 1: Read the input data
"""


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a Python dictionary.
    Args:
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
    Args:
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
        print(chunk.head())  # Process or use data here
        return chunk


"""
Part 2: Split the data into redshift bins and generate the Healpix maps
"""


def split_data(data_gal, z, num_bins):
    """
    Splits the input data into redshift bins.
    Args:
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

    # Print the number of entries in each bin
    for i in range(num_bins):
        print(f"Bin {i}: {len(binned_data[i])} entries")
    return binned_data


def Healpix_map(binned_data, bin_num, nside=64):
    """
    Generates a HEALPix map of the number of galaxies in each pixel for a given redshift bin.
    Args:
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
    Args:
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
    return Healpix_maps


"""
Part 3: Calculate the Statistical Values(Mean, Variance, Pearson correlation coefficient) of input data
"""
"""
Part 4: Visualize the results
"""


def visualize_healpix(healpix_map, nsides, output_filename):
    """
    Visualizes a HEALPix map.
    """
    # print(f"Length of healpix_map: {len(healpix_map)}")
    # print(f"nsides: {nsides}")
    output_filename = os.path.join(
        output_filename, "Healpix_maps/"
    )  # Create a directory to store the output images
    if not os.path.exists(output_filename):  # Check if the directory exists
        os.makedirs(output_filename)
    for i in range(len(healpix_map)):
        print(f"Processing bin {i}")
        print(
            f"healpix_map[{i}] keys: {list(healpix_map[i].keys())}"
        )  # Check the keys in the map
        for nside in nsides:
            print(f"Processing nside {nside}")
            plt.figure()
            hp.mollview(
                healpix_map[i][nside],
                title=f"HEALPix Map for Bin {i} (nside={nside})",
                cmap="viridis",
            )
            hp.graticule()
            plt.show()
            plt.savefig(f"{output_filename}galaxy_counts_map_bin_{i}_nside_{nside}.png")
            plt.close()


def main():
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
    data_gal = read_pandas(catalog_filename, ["unique_gal_id"])
    z = data_gal["z_cgal"]  # Spectroscopic redshift

    # Part 2: Split the data into redshift bins and generate the Healpix maps
    # Split the data into redshift bins
    binned_data = split_data(data_gal, z, num_bins)
    # Generate the HEALPix maps for each redshift bin
    Healpix_maps = binned_Healpix_maps(binned_data, num_bins, nsides)
    # visualize_healpix(Healpix_maps, nsides, output_filename)


if __name__ == "__main__":
    main()
