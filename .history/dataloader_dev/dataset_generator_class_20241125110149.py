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

####################################################################################################################
# Class
####################################################################################################################
class ImageDatasetGenerator(Dataset):
    def __init__(self, image_folder:str, guidance_folder:str, transform_images = None, transform_guidance = None):
        """
        Generates a data set of images and guidance ISDs

        Args:
            image_folder: (str)             - Folder name of images
            guidance_folder: (str)          - Folder name of isd maps
            transform_images: (callable):   - Function/transform to apply to the images. Optional, default = None.
            transform_guidance: (callable): - Function/transform to apply to the images. Optional, default = None.
        """
        self.transform_images = transform_images
        self.transform_guidance = transform_guidance

        # Set paths to images
        self.image_paths = [
            os.path.join(image_folder, file)
            for file in os.listdir(image_folder)
            if file.lower().endswith(('.tif', '.tiff'))
        ]

        # Set paths to isds
        self.guidance_paths = [
            os.path.join(guidance_folder, file)
            for file in os.listdir(guidance_folder)
            if file.lower().endswith(".png")
        ]
        
        # Sort to ensure they match
        self.image_paths.sort()
        self.guidance_paths.sort()

        # Check they are same length
        assert len(self.image_paths) == len(self.guidance_paths)
        print("Assertion Passed!!! We have the number of images and isd maps")

        self.transform_images = transform_images
        self.transform_guidance = transform_guidance

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