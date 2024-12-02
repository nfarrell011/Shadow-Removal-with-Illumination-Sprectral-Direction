"""
This file will test the model's predictions by taking an image as an input, 
predicting the ISD map with the trained model, and processing the image. 
"""
import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt
import torch
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.CvT_model import CvT
from models.cnnViT_model import VisionTransformer

class ModelTesterLogChromaticity:
    """
    Generates a log chromaticity image
    """
    def __init__(self, use_variables:bool = True, use_mean_isd:bool = False) -> None:
        """
        """
        self.anchor_point= None
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        # Dirs
        self.images_dir = None

        # Map matrices
        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None
        self.mean_isd = None

        # Usage flags
        self.use_variables = use_variables
        self.use_mean_isd = use_mean_isd

        
    def set_images_dir(self, images_dir):
        """
        Set the directory to the images, used when reading in an image from the file.
        """
        self.images_dir = images_dir

    def set_img_rbg(self, image_name) -> None:
        """
        Sets the RGB image form the directory
        """
        image_path = os.path.join(self.images_dir, image_name)
        self.rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        return None

    def set_anchor_point(self, anchor_point):
        """
        Set the anchor point; controls brightness
        """
        self.anchor_point = np.array(anchor_point)
    
    def set_isd_map(self, isd_map):
        isd_map = cv2.imread(isd_map)
        isd_map = isd_map.astype(np.float32) / 255.0
        self.isd_map = isd_map
        
    def set_img_rbg_from_variable(self, rgb_img):
        """
        Set the RGB image froma a variable.
        """
        self.rgb_img = rgb_img

    def set_isd_map_from_variable(self, isd_map):
        """
        Sets the ISD map from a variable
        """
        self.isd_map = isd_map

    def set_mean_isd_from_variable(self, mean_isd):
        """
        Sets the mean ISD from a variable.
        """
        self.mean_isd = mean_isd

    def convert_img_to_log_space(self) -> None:
        """
        Converts a 16-bit linear image to log space, setting linear 0 values to 0 in log space.

        Parameters:
        -----------
        img : np.array
            Input 16-bit image as a NumPy array.

        Returns:
        --------
        log_img : np.array
            Log-transformed image with 0 values preserved.
        """
        log_img = np.zeros_like(self.rgb_img, dtype = np.float32)
        log_img[self.rgb_img != 0] = np.log(self.rgb_img[self.rgb_img != 0])

        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        self.log_img = log_img

        return None

    def project_to_plane_locally(self) -> None:
        """
        Projects each pixel to a plane orthogonal to the ISD that is closest to that pixel. 
        """
        shifted_log_rgb = self.log_img - self.anchor_point
        dot_product_map = np.einsum('ijk,ijk->ij', shifted_log_rgb, self.isd_map)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * self.isd_map

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        projected_rgb += self.anchor_point

        self.img_chroma = projected_rgb

    def project_to_plane(self, anchor_point: np.array = (10.8, 10.8, 10.8)) -> np.array:
        """
        Projects the log RGB values onto a plane defined by a normal vector,
        with the plane anchored at the given anchor point.

        Parameters:
        -----------
        log_rgb : np.array
            Log-transformed RGB values with shape (H, W, 3).
        mean_isd : np.array
            The normal vector of the plane.
        anchor_point : np.array
            The point where the plane is anchored in log space.

        Returns:
        --------
        projected_rgb : np.array
            Log RGB values projected onto the plane.
        """
        # Shift the log RGB values relative to the anchor point
        shifted_log_rgb = self.log_img - anchor_point

        # Normalize the isd vector
        norm_isd = self.mean_isd / np.linalg.norm(self.mean_isd)

        # Perform the projection for each pixel
        #projection = np.einsum('ijk,k->ij', shifted_log_rgb, mean_isd)[:, :, np.newaxis] * mean_isd

        # Computes a dot product along the last dimension of the RGB pixel data with the ISD vector
        # where, i: Height dimension, j: Width dimension, k: RGB channels (3 channels)
        # sum over the shared k dimension: RGB and ISD
        # contracts the k dimension, resulting array will have shape (H, W), indicated by ij in the output.
        dot_product = np.einsum('ijk,k->ij', shifted_log_rgb, norm_isd)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * norm_isd

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        projected_rgb += anchor_point

        self.img_chroma = projected_rgb

        return None
    
    def log_to_linear(self) -> None:
        """
        Converts log transformed image back to linear space in 16 bits.

        Parameters:
        -----------
        log_rgb : np.array
            Log-transformed image with values between 0 and 11.1.

        Returns:
        --------
        vis_img : np.array
            Visualization-ready 8-bit image.
        """

        linear_img = np.exp(self.img_chroma)
        self.linear_converted_log_chroma = linear_img
        return None
    
    def convert_16bit_to_8bit(self) -> None:
        """
        Converts a 16-bit image to 8-bit by normalizing pixel values.

        Parameters:
        -----------
        img : np.array
            Input image array in 16-bit format (dtype: np.uint16).
        
        Returns:
        --------
        img_8bit : np.array
            Output image array converted to 8-bit (dtype: np.uint8).
        """
        img_normalized = np.clip(self.linear_converted_log_chroma / 255, 0, 255)
        img_8bit = np.uint8(img_normalized)
        self.linear_converted_log_chroma_8bit = img_8bit
    
    def display_image(self):
        """ 
        """
        cv2.imshow("Result", self.linear_converted_log_chroma_8bit)
        cv2.waitKey(0)
    
    def display_original_image(self):
        """ 
        """
        cv2.imshow("Orig", self.rgb_img)
        cv2.waitKey(0)

    def process_img(self, images_dir, image_name, isd, anchor_point) -> np.array:
        """ 
        """
        if self.use_variables:
            if self.use_mean_isd:
                self.set_img_rbg_from_variable(image_name)
                self.set_mean_isd_from_variable(isd)
                self.convert_img_to_log_space()
                self.set_anchor_point(anchor_point)
                self.project_to_plane()
                self.log_to_linear()
                self.convert_16bit_to_8bit()
            else:
                self.set_img_rbg_from_variable(image_name)
                self.set_isd_map_from_variable(isd)
                self.convert_img_to_log_space()
                self.set_anchor_point(anchor_point)
                self.project_to_plane_locally()
                self.log_to_linear()
                self.convert_16bit_to_8bit()
        
        else:
            self.set_images_dir(images_dir)
            self.set_img_rbg(image_name)
            self.display_original_image()
            self.convert_img_to_log_space()
            self.set_isd_map(isd)
            self.set_anchor_point(anchor_point)
            self.project_to_plane_locally()
            self.log_to_linear()
            self.convert_16bit_to_8bit()
            self.display_image()

        return self.linear_converted_log_chroma_8bit

############################################################################################################################################

def main():
    """
    """
    # Define paths
    path = Path.cwd()
    path_to_home = str(path.parent)

    # Get the images and isds; the isds here are from file ~ for testing purposes
    image_path = os.path.join(path_to_home, "training_data/training_images_cropped/Adari_Girish_000_0_0.tif")
    isd_path = os.path.join(path_to_home, "training_data/training_isds_cropped/Adari_Girish_000_isd_0_0.png")
    
    # Check if files exist
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
    if not os.path.exists(isd_path):
        print("Error: ISD file not found!")
    
    # Read images
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    isd = cv2.imread(isd_path)
    
    if image is None:
        print("Error: Failed to load image file!")
    if isd is None:
        print("Error: Failed to load ISD file!")

    # Set Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    anchor_point = [10.8, 10.8, 10.8] # THIS IS A DEFAULT VALUE! The real anchor point for an image is in the XML doc.
    model = "CvT"
    model_state = os.path.join(path_to_home, "results/test_run_01/model_states/checkpoint_epoch_49.pth")

    # This is isd from file, to use it we need to divide by 255.0 and set to a float
    isd = isd.astype(np.float32) / 255.0

    # The model is trained with scaled images; this will prepare the image for the model
    image_for_model = image.astype(np.float32) / 65535.0

    # Displays the original image
    cv2.imwrite(path_to_home + '/' + "original_image.png", image)

    # Select model and load states
    if model == "CvT":
        model = CvT(embed_dim = 64)
        model = model.to(device)
        checkpoint = torch.load(model_state, weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        model = VisionTransformer()
        model = model.to(device)
        checkpoint = torch.load(model_state, weights_only = True)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Set up single image tensor to pass to the model
    image_tensor = torch.tensor(image_for_model, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    print(f"Image Tensor Shape: {image_tensor.shape}")

    # Execute model and get the prediction
    with torch.no_grad():
        isd_pred = model(image_tensor).cpu().numpy().squeeze()

    # Processes image with the mdoel output and save the results
    isd_pred = np.transpose(isd_pred, (1, 2, 0))
    processor = ModelTesterLogChromaticity(use_variables = True, use_mean_isd = False)
    img = processor.process_img(None, image, isd_pred, anchor_point)
    cv2.imwrite(path_to_home + '/' + "predicted.png", img)

    # Process the image with actaul isd vals
    processor = ModelTesterLogChromaticity(use_variables = True, use_mean_isd = False)
    img = processor.process_img(None, image, isd, anchor_point)
    cv2.imwrite(path_to_home + '/' + "actual.png", img)


if __name__ == "__main__":
    main()