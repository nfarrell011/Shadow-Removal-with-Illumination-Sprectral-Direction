""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains a class that will organize the processed image and the isd maps in preparation for 
data loading.

The class center crop and divide into 4 equal regions
"""
##########################################################################################################################################
# Packages
from PIL import Image # I swtiched to cv2 for reading in images currently
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import os  
from pathlib import Path
import sys
from einops import rearrange

##########################################################################################################################################
# Class
##########################################################################################################################################
class DataFileReorganizer:
    """ 
    Reorganizes data files in preparation for data loading.

    Methods:
        * reorganize_data_files
        * get_channel_wise_mean_and_std
        * get_min_max_dims
        * center_crop_and_split

    First reorganize the files. This aggregates all the training data into two folders, images and isd_maps.
    If you want the mean and standard deviation compute that next using the aggregated folder.
    If you need to check the dims before cropping, do that next.
    Finally, crop the images, if needed.
    """
    def __init__(self) -> None:
        pass

    def reorganize_data_files(parent_data_folder_name:str, images_folder_list:list, image_quality_list:list) -> None:
        """ 
        Loops over the data folder extracting the original images and their corresponding isd_maps.
        This is needed because the isd_maps folder contains isd_maps for all the processed images and
        we will only use a subset in training, i.e., high quality.

        It generate a new folder called training_data two sub-folders, images and isd_maps
        """
        # Set the dir to the project home
        cwd = Path.cwd()
        project_root_folder = cwd.parent
        
        # Set the destination folders; create if needed
        
        # Root
        training_data_root = str(project_root_folder / "training_data")
        os.makedirs(training_data_root, exist_ok = True)
        
        # Img destination
        training_images_folder = str(project_root_folder / "training_data" / "training_images")
        os.makedirs(training_images_folder, exist_ok = True)
        
        # Img destination
        training_isd_maps_folder = str(project_root_folder / "training_data" / "training_isds")
        os.makedirs(training_isd_maps_folder, exist_ok = True)

        # Go through each of the image folders; i.e., folder_1, folder_2
        for img_folder in images_folder_list:

            # Go through each of the quality folders; i.e, high_quality, low_quality
            for image_quality in image_quality_list:
                    
                    # Set folder paths
                    image_folder = project_root_folder / parent_data_folder_name / img_folder / f"processed" / image_quality
                    isd_map_folder = project_root_folder / parent_data_folder_name / img_folder / f"processed" / "isd_maps"
                    image_folder_str = str(image_folder)
                    isd_map_folder_str = str(isd_map_folder)
                    print(image_folder_str)
                    print(isd_map_folder_str)

                    # Get the images and the isd maps so we can move them to the training folder; make copies
                    for image_file in os.listdir(image_folder_str):
                        if image_file.lower().endswith((".tif", ".tiff")):

                            # Step # 1: Get the image of interest
                            img = cv2.imread(os.path.join(image_folder_str, image_file), cv2.IMREAD_UNCHANGED)

                            # Step 2: Get the corresponding ISD map
                            img_name_str = image_file.split(".", 1)[0]
                            img_isd_map_str = img_name_str + "_isd.png"


                            for isd_file in os.listdir(isd_map_folder_str):
                                if isd_file.lower().endswith((".png")):
                                    if isd_file == img_isd_map_str:
                                        isd = cv2.imread(os.path.join(isd_map_folder_str, isd_file), cv2.IMREAD_UNCHANGED)

                                        # Set the save names
                                        img_out_file_name = image_file
                                        isd_out_file_name = img_isd_map_str

                                        cv2.imwrite(training_images_folder + "/" + img_out_file_name, img)
                                        cv2.imwrite(training_isd_maps_folder + "/" + isd_out_file_name, isd)
                                        break

    def get_channel_wise_mean_and_std(self, folder_name: str) -> tuple:
        """ 
        Computes the scaled mean and std of each channel in the training set.
        Used for normalization.
        """
        cwd = Path.cwd()
        project_home_dir = cwd.parent
        image_folder = str(project_home_dir / "training_data" / folder_name)

        # Initialize trackers
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_squared_sum = np.zeros(3, dtype=np.float64)
        num_pixels = 0

        # Iterate over all the images in the training dir
        for file in os.listdir(image_folder):
            if file.lower().endswith((".tif", ".tiff")):
                img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)

                # Scale
                img = img / 65535.0 

                # Update trackers
                channel_sum += np.sum(img, axis = (0, 1))  # For mean
                channel_squared_sum += np.sum(img**2, axis= (0, 1))  # For std
                num_pixels += img.shape[0] * img.shape[1]

        # Compute mean and std
        channel_mean = channel_sum / num_pixels
        channel_std = np.sqrt(
            np.maximum(channel_squared_sum / num_pixels - channel_mean ** 2, 1e-8)  # Avoid dividing by zero??
        )

        return channel_mean, channel_std
    
    def get_min_max_dims(folder_name:str) -> tuple:
        """ 
        Get the min and max dims out of all the images in the training data.
        Used to decide on crop sizes.
        """
        # Pathing
        cwd = Path.cwd()
        project_home_dir = cwd.parent
        image_folder = str(project_home_dir / "training_data" / folder_name)

        # Trackers
        max_height = 0
        max_width = 0
        min_height = np.inf
        min_width = np.inf

        # Iterate over all the images in the training dir
        for file in os.listdir(image_folder):
            if file.lower().endswith((".tif", ".tiff")):

                # Get an image
                img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)

                # Get img dims
                H, W, C = img.shape 

                # Update trackers
                max_height = max(max_height, H)
                max_width = max(max_width, W)
                min_height = min(min_height, H)
                min_width = min(min_width, W)

        return max_width, max_height, min_width, min_height

    def center_crop_and_split(folder_name: str, output_folder: str, crop_size:int, sub_crop_size:int):
        """
        Center crop all images and then divide the center crop into four 224x224 sub-crops.
        """
        # Pathing
        cwd = Path.cwd()
        project_home_dir = cwd.parent
        image_folder = str(project_home_dir / "training_data" / folder_name)
        output_dir = Path(project_home_dir / "training_data" / output_folder)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)

        # Iterate over all the images in the folder
        for file in os.listdir(image_folder):
            if file.lower().endswith((".tif", ".tiff", ".png")):
                # Read the image
                img = cv2.imread(os.path.join(image_folder, file), cv2.IMREAD_UNCHANGED)

                # Get image dimensions
                H, W = img.shape[:2]

                # Calculate center crop
                start_x = (W - crop_size) // 2
                start_y = (H - crop_size) // 2
                end_x = start_x + crop_size
                end_y = start_y + crop_size

                # Extract center crop
                center_crop = img[start_y:end_y, start_x:end_x]

                # Divide the center crop into 4 (224x224) sub-crops
                for i in range(2):  # Rows (top and bottom)
                    for j in range(2):  # Columns (left and right)
                        x_start = j * sub_crop_size
                        y_start = i * sub_crop_size
                        x_end = x_start + sub_crop_size
                        y_end = y_start + sub_crop_size

                        # Extract sub-crop
                        sub_crop = center_crop[y_start:y_end, x_start:x_end]

                        # Determine save path
                        if file.lower().endswith(".png"):
                            sub_crop_filename = f"{file.split('.')[0]}_{i}_{j}.png"
                        else:
                            sub_crop_filename = f"{file.split('.')[0]}_{i}_{j}.tif"
                        cv2.imwrite(os.path.join(output_dir_str, sub_crop_filename), sub_crop)

if __name__ == "__main__":
    pass