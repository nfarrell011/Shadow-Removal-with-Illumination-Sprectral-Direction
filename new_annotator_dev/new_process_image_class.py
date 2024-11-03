import numpy as np
import cv2 

class LogChromaticity:
    """
    """
    def __init__(self) -> None:
        """
        """
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        self.lit_pixels = None
        self.shadow_pixels = None

        self.mean_isd = None

    def set_img_rbg(self, image_path) -> None:
        """ 
        """
        self.rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        print("Type", self.rgb_img.dtype)  
        return None
    
    def set_lit_shadow_pixels_list(self, clicks) -> None:
        """ 
        """
        self.lit_pixels = [(pair[0], pair[1]) for pair in clicks]
        self.shadow_pixels = [(pair[2], pair[3]) for pair in clicks]
        return None

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

        # Prepare a mask to keep zero-valued pixels unchanged
        zero_mask = (self.rgb_img == 0)

        # Apply the log transformation
        log_img = np.log(self.rgb_img)  

        # Set log-transformed values to 0 where the original pixels were 0
        log_img[zero_mask] = 0

        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        self.log_img = log_img

        # Checking values and type
        print(f"Max Log Image: {np.max(log_img)}")
        print(f"Min Log Image: {np.min(log_img)}")
        print(f"DType Log Image: {log_img.dtype}")

        return None
    
    def compute_isd(self) -> np.array:
        """
        Computes the Illumination Subspace Direction (ISD) vector 
        by comparing pixel values between lit and shadow regions in log RGB space.

        Parameters:
        -----------
        log_rgb : np.array
            The log transformed image.
        lit_pixels : list of tuples
            List of (x, y) coordinates representing pixels in the lit region.
        
        shadow_pixels : list of tuples
            List of (x, y) coordinates representing pixels in the shadow region.

        Returns:
        --------
        isd : np.array
            A 3-element vector representing the unit-normalized Illumination Subspace Direction (ISD).
            This vector indicates the direction of change in RGB values caused by illumination.

        Explanation:
        ------------
        - The ISD vector captures the differences between lit and shadow regions.
        - This function uses the mean RGB difference between these regions to find
        the vector in log RGB space that represents the change in illumination.
        - It normalizes the mean difference to ensure the ISD is a unit vector.

        Notes:
        ------
        - The log RGB values (log_rgb) must already be computed prior to calling this function.
        - Normalization ensures that the ISD vector is scale-independent, which is 
        essential for proper projection in chromaticity-based models.
        """
        # Unpack the tuples of pixel coordinates
        lit_x, lit_y = zip(*self.lit_pixels) 
        shadow_x, shadow_y = zip(*self.shadow_pixels)

        lit_values = self.log_img[lit_x, lit_y]
        shadow_values = self.log_img[shadow_x, shadow_y]

        # Get pixel diffs
        pixel_diff = lit_values  - shadow_values
        #isd = pixel_diff / np.linalg.norm(pixel_diff)

        # Use equation #4 from Bruce's RoadVision Paper
        mean_diff = np.mean(pixel_diff, axis=0)
        isd = mean_diff / np.linalg.norm(mean_diff)
        
        self.mean_isd = isd
        return None

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
        Converts log transofrmed image back to linear space in 16 bits.

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
        img_normalized = cv2.normalize(self.linear_converted_log_chroma, None, 0, 255, cv2.NORM_MINMAX)
        img_8bit = np.uint8(img_normalized)
        self.linear_converted_log_chroma_8bit = img_8bit
        return None
    
    def display_processed_img(self) -> None:
        """ 
        """
        cv2.imshow("Processed Image", self.linear_converted_log_chroma_8bit)
        cv2.resizeWindow("Processed Image", 600, 600)

    def process_img(self, image_path, clicks) -> None:
        """
        """
        self.set_img_rbg(image_path)
        self.convert_img_to_log_space()
        self.set_lit_shadow_pixels_list(clicks)
        self.compute_isd()
        self.project_to_plane()
        self.log_to_linear()
        self.convert_16bit_to_8bit()
        self.display_processed_img()

if __name__== "__main__":
    pass




    