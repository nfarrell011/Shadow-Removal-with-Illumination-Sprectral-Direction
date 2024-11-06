import numpy as np
import cv2 

class LogChromaticity:
    """
    """
    def __init__(self) -> None:
        """
        """
        self.method = 'mean'
        self.anchor_point= np.array([10.8, 10.8, 10.8])
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        # Default patch size set to single pixel value
        self.patch_size = (1, 1)

        self.lit_pixels = None
        self.shadow_pixels = None

        self.mean_isd = None

        # Map matrices
        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None

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
    
    ###################################################################################################################
    # Methods for Pre Processing Images
    ###################################################################################################################

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
    
    ###################################################################################################################
    # Methods for using Mean
    ###################################################################################################################

    def get_patch_mean(self, center, patch_size):
        """
        Extracts a patch from a NumPy array centered at a specific pixel location 
        and returns the mean of the patch.

        Parameters:
        - img: np.array, the input array (can be 2D or 3D).
        - center: tuple, the (y, x) coordinates of the center pixel.
        - patch_size: tuple, the (height, width) of the patch.

        Returns:
        - mean_value: float, the mean of the extracted patch.
        """
        y, x = center
        patch_height, patch_width = patch_size

        # Calculate the start and end indices, ensuring they don't go out of bounds
        start_y = max(y - patch_height // 2, 0)
        end_y = min(y + patch_height // 2 + 1, self.log_img.shape[0])
        start_x = max(x - patch_width // 2, 0)
        end_x = min(x + patch_width // 2 + 1, self.log_img.shape[1])

        # Extract the patch and return its mean value
        patch = self.log_img[start_y:end_y, start_x:end_x]
        # print(f"Patch: {patch}")
        # print(f"Patch shape: {patch.shape}")
        mean_value = np.mean(patch, axis=(0, 1))
        # print(f"Patch mean: {mean_value}")

        return mean_value
    
    
    def compute_isd(self) -> np.array:
        """
        Computes the Illumitant Spectral Direction (ISD) vector 
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
            A 3-element vector representing the unit-normalized Illumitant Spectral Direction (ISD).
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
        # Ensure `log_img` exists and lit/shadow pixels are defined
        if self.log_img is None or not self.lit_pixels or not self.shadow_pixels:
            raise ValueError("Log image and pixel coordinates must be set before computing ISD.")

        # Compute mean patch values for each lit and shadow pixel
        lit_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.lit_pixels])
        shadow_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.shadow_pixels])

        # Calculate mean difference between lit and shadow regions
        mean_diff = np.mean(lit_means - shadow_means, axis=0)

        # Normalize the mean difference to get the ISD vector
        isd = mean_diff / np.linalg.norm(mean_diff)
        self.mean_isd = isd


    def project_to_plane(self) -> np.array:
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
        shifted_log_rgb = self.log_img - self.anchor_point

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
        projected_rgb += self.anchor_point

        self.img_chroma = projected_rgb

        return None
    
    ###################################################################################################################
    # Methods for using Weighted Mean
    ###################################################################################################################
    def get_annotation_weights(self) -> None:
        """
        Computes a weight for each annotation for each pixel based on its distance to each pixel.
        weight_i = dist_i / sum_dist_i^n
        """
        """
        Computes a weight for each annotation for each pixel based on its distance to each pixel.
        weight_i = dist_i / sum_dist_i^n
        """
        # Calculate midpoints between the lit and shadow pairs
        midpoints = np.array([((x1 + x2) / 2, (y1 + y2) / 2) 
                              for (x1, y1), (x2, y2) in zip(self.lit_pixels, self.shadow_pixels)])

        # Get the number of annotations
        num_isds = midpoints.shape[0]

        # Get the image dims
        height, width = self.rgb_img.shape[0], self.rgb_img.shape[1]

        # Initialize weight map matrix (Height x Width x Num_ISDs)
        annotation_weight_map = np.zeros((height, width, num_isds), dtype=np.float32)
        
        # Generate a distance map for each annotation to each pixel
        for i in range(num_isds):

            # Extract center point of the annotation line and round down coordinates to integer values
            y, x = np.floor(midpoints[i][1]).astype(int), np.floor(midpoints[i][0]).astype(int)

            # Initialize a binary image with all ones
            binary_image = np.ones((height, width), dtype=np.uint8)
            binary_image = binary_image * 255 # make image all white

            # Set one pixel to black (the annotation point)
            if 0 <= y < height and 0 <= x < width:
                binary_image[y, x] = 0

            # Perform distance transform from each pixel to the annotation center point
            dist = cv2.distanceTransform(binary_image, cv2.DIST_L2, 0)

            # Normalize the distance transform output to [0, 1]
            dist_output = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

            # Add the distances for this annotation to the annotation_map
            annotation_weight_map[:, :, i] = dist_output

            # Display the distance transform result
            #cv2.imshow("Distance Transform", dist_output)

        # Transforms distances to proportions to use as the weights
        annotation_weight_map = annotation_weight_map / annotation_weight_map.sum(axis = 2, keepdims = True)
        self.annotation_weight_map = annotation_weight_map

    def compute_weighted_mean_isd_per_pixel(self) -> None:
        """
        Computes the ISD for each pixel using the weighted mean of ISDs from each
        annotation.
        Weight is computed using the distance to each annotation.
        """
        # Ensure `log_img` exists and lit/shadow pixels are defined
        if self.log_img is None or not self.lit_pixels or not self.shadow_pixels:
            raise ValueError("Log image and pixel coordinates must be set before computing ISD.")

        # Compute mean patch values for each lit and shadow pixel
        lit_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.lit_pixels])
        shadow_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.shadow_pixels])

        # Calculate mean difference between lit and shadow regions
        pixel_diff = lit_means - shadow_means
        
        norms = np.linalg.norm(pixel_diff, axis = 1, keepdims = True)
        isds = pixel_diff / norms

        # Expand the annotation map for broadcasting
        weight_map_expanded = self.annotation_weight_map[:, :, :, np.newaxis]

        # Element multiplication between the pixel-specific weights and the ISD values; the weights sum to 1.
        weighted_isds = weight_map_expanded * isds

        # Sum the values across the IDSs to get the weigthed mean 
        weighted_mean_isds = np.sum(weighted_isds, axis = 2)

        # Update isd_map attribute
        self.isd_map = weighted_mean_isds

    def project_to_plane_locally(self, anchor_point: np.array = np.array([10.8, 10.8, 10.8])) -> None:
        """
        Projects each pixel to a plane orthogonal to the ISD that is closest to that pixel. 
        """
        shifted_log_rgb = self.log_img - anchor_point
        dot_product_map = np.einsum('ijk,ijk->ij', shifted_log_rgb, self.isd_map)

        # Reshape the dot product to (H, W, 1) for broadcasting
        dot_product_reshaped = dot_product_map[:, :, np.newaxis]

        # Multiply with the ISD vector to get the projected RGB values
        projection = dot_product_reshaped * self.isd_map

        # Subtract the projection from the shifted values to get plane-projected values
        projected_rgb = shifted_log_rgb - projection

        # Shift the values back by adding the anchor point
        projected_rgb += anchor_point

        self.img_chroma = projected_rgb
    
    ###################################################################################################################
    # Methods for Post Processing Images
    ###################################################################################################################
    
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
    
    ###################################################################################################################
    # Methods for using GUI Controls
    ###################################################################################################################

    def update_patch_size(self, size):
        """
        """
        self.patch_size = size
        print(f"Updated patch_size: {self.patch_size}")

        # Re-run processing steps that depend on isd
        self.compute_isd()
        self.project_to_plane()

        self.log_to_linear()
        self.convert_16bit_to_8bit()
        return self.linear_converted_log_chroma_8bit

    def update_anchor_point(self, val):
        """
        """
        # Update the anchor point's first component based on trackbar position
        point = val / 10.0  # Scale value to range 0.0 to 11.1
        self.anchor_point = np.array([point, point, point])
        print(f"Updated anchor_point: {self.anchor_point}")

        # Re-run processing steps that depend on anchor_point
        if self.method == 'mean':
            self.project_to_plane()

        elif self.method == 'weighted':
            self.project_to_plane_locally()

        self.log_to_linear()
        self.convert_16bit_to_8bit()

        return self.linear_converted_log_chroma_8bit
    
    def set_isd_method(self, val):
        """
        """
        if val == 0:
            self.method = "mean"
            print("ISD set to Global Mean")
            self.compute_isd()
            self.project_to_plane()
        elif val == 1:
            self.method = "weighted"
            print("ISD set to Weighted Mean")
            self.get_annotation_weights()
            self.compute_weighted_mean_isd_per_pixel()
            self.project_to_plane_locally()
        
        self.log_to_linear()
        self.convert_16bit_to_8bit()

        return self.linear_converted_log_chroma_8bit
        


    
    ###################################################################################################################
    # Methods for  Execution
    ###################################################################################################################


    def process_img(self, image_path, clicks) -> np.array:

            self.set_img_rbg(image_path)
            self.convert_img_to_log_space()
            self.set_lit_shadow_pixels_list(clicks)
            
            if self.method == 'mean':
                self.compute_isd()
                self.project_to_plane()

            elif self.method == 'weighted':
                self.get_annotation_weights()
                self.compute_weighted_mean_isd_per_pixel()
                self.project_to_plane_locally()
            
            self.log_to_linear()
            self.convert_16bit_to_8bit()

            return self.linear_converted_log_chroma_8bit

    if __name__== "__main__":
        pass




    