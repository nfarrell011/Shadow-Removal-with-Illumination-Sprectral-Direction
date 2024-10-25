"""
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Project 2: Color Constancy, Shadow Manipulation, or Intrinsic Imaging  

Run from commandline with "folder_#" as arg. 
Processes all images in 'done' subdirectory and uses matching annotations CSV from the provided folder.
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Modules
from utils import *

# Constants
epsilon = 1e-10
anchor_point = 10.8

def find_csv_file(data_folder: Path) -> Path:
    """Finds the annotation CSV file dynamically within the data folder."""
    csv_files = list(data_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {data_folder}.")
    return csv_files[0]  # Assume only one relevant CSV exists

def process_image(img_file: Path, annotations_df: pd.DataFrame, data_folder: Path):
    """Processes a single image and saves the results."""
    img_16bit = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
    img_16bit = cv2.cvtColor(img_16bit, cv2.COLOR_BGR2RGB)

    # Create 8-bit image
    img_8bit = convert_16bit_to_8bit(img_16bit)

    # Generate chromaticity image
    chromaticity_image = convert_img_to_rg_chromaticity(img_16bit, rg_only=True)

    # Get coordinates for shadow/lit pixels from annotations
    img_name = img_file.name
    lit_pixels, shadow_pixels = get_lit_shadow_pixel_coordinates(annotations_df, img_name)

    # Convert image to log space
    log_rgb = convert_img_to_log_space(img_16bit)

    # Compute the ISD estimate
    isd = compute_isd(log_rgb, lit_pixels, shadow_pixels)

    # Project the log image onto the chromaticity plane
    projected_log_rgb = project_to_plane(log_rgb, isd, np.full(3, anchor_point))
    projected_rgb = log_to_linear(projected_log_rgb)

    # Save the processed image
    processed_dir = data_folder / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_img_path = processed_dir / f"{img_name.split('.')[0]}_lsc.tiff"
    cv2.imwrite(str(processed_img_path), projected_rgb)

    # Plot image comparisons
    fig, axes = plt.subplots(3, 1, figsize=(5, 8))
    ax = axes.ravel()

    ax[0].imshow(img_8bit)
    ax[0].axis("off")
    ax[0].set_title("Original")

    ax[1].imshow(chromaticity_image)
    ax[1].axis("off")
    ax[1].set_title("Standard Chromaticity")

    projected_rgb_8bit = convert_16bit_to_8bit(projected_rgb)
    ax[2].imshow(projected_rgb_8bit)
    ax[2].axis("off")
    ax[2].set_title("Log Space Chromaticity")

    # Save the plot in 'chromaticity_results'
    results_dir = Path.cwd() / "chromaticity_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f"{img_name.split('.')[0]}_results.png"
    plt.savefig(str(results_file))
    plt.close()

def main(data_folder: Path):
    """Main function to process all images in the 'done' subdirectory."""
    # Find the relevant CSV file and 'done' folder dynamically
    annotations_csv = find_csv_file(data_folder)
    done_folder = data_folder / "done"

    # Load the annotations CSV
    annotations_df = pd.read_csv(annotations_csv, index_col="filename")

    # Process each image in the 'done' folder
    for img_file in done_folder.glob("*.tif"):
        print(f"Processing {img_file.name} ...")
        process_image(img_file, annotations_df, data_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images with dynamic folder and CSV naming.")
    parser.add_argument("data_folder", type=Path, help="The folder containing the 'done' subdirectory and annotations CSV.")
    args = parser.parse_args()

    main(args.data_folder)