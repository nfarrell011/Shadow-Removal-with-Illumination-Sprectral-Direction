""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

This file contains a class generate a dataset used for training a model
"""
####################################################################################################################
# Packages
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split

####################################################################################################################
# Class
####################################################################################################################
class ImageDatasetGenerator(Dataset):
    def __init__(self, image_folder:str, guidance_folder:str, split:str = None, val_size:float = 0.2,
                 random_seed:int = 42, transform_images:callable= None, transform_guidance:callable = None, use_mean: bool = False):
        """
        Generates a dataset of images and guidance ISDs, with optional train/val splitting 
        Uses SKLearn for splitting.

        Args:
            image_folder: (str)             - Folder name of images
            guidance_folder: (str)          - Folder name of ISD maps
            split: (str)                    - Dataset split [train, val]. If None, uses all data.
            test_size: (float)              - Proportion of the dataset to use as val set. Default is 0.2.
            random_seed: (int)              - Random seed for reproducibility; used in SKlearn train-test split.
            transform_images: (callable):   - Function/transform to apply to the images. Optional, default = None.
            transform_guidance: (callable): - Function/transform to apply to the ISD maps. Optional, default = None.
            use_mean: (bool)                - Sets the output for guidance image to mean isd if True or pixel-wise if False. Optional, default = False
        """
        # Set the transformation functions
        self.transform_images = transform_images
        self.transform_guidance = transform_guidance

        # Set the ISD to mean or pixelwise
        self.use_mean = use_mean

        # Set paths to images and ISDs using glob
        self.image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")) + glob.glob(os.path.join(image_folder, "*.tiff")))
        self.guidance_paths = sorted(glob.glob(os.path.join(guidance_folder, "*.png")))

        # Check that the number of images and ISD maps match
        assert len(self.image_paths) == len(self.guidance_paths), "Assertion failed!! Num images and ISDs do not match."
        print("Assertion Passed!!! We have the same number of images and ISD maps.")

        # Splits the the data using SKlearn
        if split in ["train", "val"]:
            train_img, val_img, train_guid, val_guid = train_test_split(
                self.image_paths, self.guidance_paths, test_size=val_size, random_state=random_seed, shuffle = True
            )
            # Sets attribute to return correct set
            if split == "train":
                self.image_paths, self.guidance_paths = train_img, train_guid
            elif split == "val":
                self.image_paths, self.guidance_paths = val_img, val_guid

        # If split is not valid option, raise error
        elif split is not None:
            raise ValueError("Invalid splitter. Must be 'train', 'val', or 'None.")
        
    def __len__(self):
        """ 
        Returns the number of image pairs. 
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """ 
        Get the primary image and its corresponding guidance image.
        """
        # Load primary image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  
        image = image.astype(np.float32) / 65535.0

        # Load guidance image
        guidance_path = self.guidance_paths[idx]
        guidance_image = cv2.imread(guidance_path, cv2.IMREAD_UNCHANGED)
        guidance_image = guidance_image.astype(np.float32) / 255.0
        if self.use_mean:
            guidance_image = np.mean(guidance_image, axis=(0,1))


        # Perform transforms
        if self.transform_images:
            image = self.transform_images(image)
            guidance_image = self.transform_guidance(guidance_image)

        # Return the pair of images
        return image, guidance_image
    
####################################################################################################################
# End Class
####################################################################################################################
if __name__ == "__main__":
    pass