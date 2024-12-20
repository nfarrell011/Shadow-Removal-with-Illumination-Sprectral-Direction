"""
IMPORTANT!!!

README:

This file can be used to check that the saved ISD maps can be used to recreate the shadow-free image.
It is currently fast and dirty and the anchor point is simply set to [10.8, 10.8, 10.8]; this is NOT true for all images. 

Get true anchor point for a specific image from the XML; or UPDATE THIS CODE! 
"""
import numpy as np 
import cv2
import os

class ImageTesterLogChromaticity:
    """
    """
    def __init__(self, use_variables:bool = False) -> None:
        """
        """
        self.anchor_point= None
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        # Map matrices
        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None

        # Usage Flag
        self.use_variables = use_variables
        
    def set_images_dir(self, images_dir):
        """ 
        """
        self.images_dir = images_dir

    def set_img_rbg(self, image_name) -> None:
        """ 
        """
        image_path = os.path.join(self.images_dir, image_name)
        self.rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) 
        return None

    def set_anchor_point(self, anchor_point):
        """ 
        """
        self.anchor_point = np.array(anchor_point)
    
    def set_isd_map(self, isd_map):
        isd_map = cv2.imread(isd_map)
        isd_map = isd_map.astype(np.float32) / 255.0
        self.isd_map = isd_map
        
    def set_img_rbg_and_isd_map_from_variable(self, rgb_img, isd_map):
        """ 
        """
        self.rgb_img = rgb_img
        self.isd_map = isd_map

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

    def process_img(self, images_dir, image_name, isd_map, anchor_point) -> np.array:
        """ 
        """
        if self.use_variables:
            self.set_img_rbg_and_isd_map_from_variable(image_name, isd_map)
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
            self.set_isd_map(isd_map)
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
    # Update these to check an image
    image_name = "Adari_Girish_000_0_0.tif"
    images_dir = "/Users/nelsonfarrell/Documents/Northeastern/7180/projects/spectral_ratio/training_data/training_images_cropped/"
    isd_map_for_image_png = "/Users/nelsonfarrell/Documents/Northeastern/7180/projects/spectral_ratio/training_data/training_isds_cropped/Adari_Girish_000_isd_0_0.png"
    anchor_point = [10.8, 10.8, 10.8] # THIS IS A DEFAULT VALUE! The real anchor point for an image is in the XML doc.

    top_left = images_dir + image_name
    top_left_image = cv2.imread(top_left, cv2.IMREAD_UNCHANGED)
    print("Image Values", top_left_image)

    isd_map_for_image_png = cv2.imread(isd_map_for_image_png)
    isd_map_for_image_png = isd_map_for_image_png.astype(float) / 255.0
    print(isd_map_for_image_png)
    

    import matplotlib.pyplot as plt

    # Processes image and displays results
    processor = ImageTesterLogChromaticity(use_variables = True)
    img = processor.process_img(None, top_left_image, isd_map_for_image_png, anchor_point)
    plt.imshow(img)
    plt.show()

    # Processes image and displays results
    # processor = ImageTesterLogChromaticity()
    # processor.process_img(images_dir, image_name, isd_map_for_image_png, anchor_point)

if __name__ == "__main__":
    main()