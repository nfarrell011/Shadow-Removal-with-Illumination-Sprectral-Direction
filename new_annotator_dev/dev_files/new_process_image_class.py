import numpy as np
import cv2 

class LogChromaticity:
    """

    """
    def __init__(self, method = "weighted") -> None:
        """
        """
        self.method = method
        self.rgb_img = None
        self.log_img = None
        self.img_chroma = None
        self.linear_converted_log_chroma = None
        self.linear_converted_log_chroma_8bit = None

        self.lit_pixels = None
        self.shadow_pixels = None

        self.mean_isd = None

        # Map matrices
        self.closest_annotation_map = None
        self.annotation_weight_map = None
        self.isd_map = None

    ###################################################################################################################
    # Setters
    ###################################################################################################################
    def set_img_rbg(self, image_path) -> None:
        """
        Reads in an image using cv2 IMREAD_UNCHANGED flag and sets the corresponding attribute.

        Params:
            * image_path: (str) - The path the image of interest.
        """
        self.rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return None
    
    def set_lit_shadow_pixels_list(self, clicks) -> None:
        """
        Generates 2 lists of pixel coordinates; lit_pixels and shadow_pixels and sets the 
        corresponding attributes.

        Params:
            * clicks: (list) - The annotations from AnnotatorManager
        """
        self.lit_pixels = [(pair[0], pair[1]) for pair in clicks]
        self.shadow_pixels = [(pair[2], pair[3]) for pair in clicks]
        return None
    
    def convert_img_to_log_space(self) -> None:
        """
        Converts a 16-bit linear image to log space, setting linear 0 values to 0 in log space.
        Sets the corresponding attribute

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

        # Check values are in expected range
        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        # Set attribute
        self.log_img = log_img

        return None

    ###################################################################################################################
    # Methods for using Nearest Neighbor
    ###################################################################################################################
    def find_closest_annotation(self) -> None:
        """
        Find the closest annotation to each pixel.
        Uses the midpoint of the annotation line to find the distance.
        """
        # Calculate midpoints between the lit and shadow pairs
        midpoints = np.array([((x1 + x2) / 2, (y1 + y2) / 2) 
                              for (x1, y1), (x2, y2) in zip(self.lit_pixels, self.shadow_pixels)])

        # Initialize closest annotation mapper
        closest_annotation_map = np.zeros_like(self.rgb_img[:, :, 0], dtype=int)
        
        # Find the closest midpoint for each pixel in the image
        for i in range(self.rgb_img.shape[0]):
            for j in range(self.rgb_img.shape[1]):
                point_of_interest = np.array([i, j])
                distances = np.linalg.norm(midpoints - point_of_interest, axis=1)
                min_dist = np.argmin(distances)
                closest_annotation_map[i, j] = min_dist
        
        # Update the map attribute
        self.closest_annotation_map = closest_annotation_map

    def compute_localized_isd(self) -> None:
        """
        Compute the ISD per pixel using the closest annotation
        """
        # Unpack the tuples of pixel coordinates
        lit_x, lit_y = zip(*self.lit_pixels) 
        shadow_x, shadow_y = zip(*self.shadow_pixels)

        # Get the pixel values
        lit_values = self.log_img[lit_x, lit_y]
        shadow_values = self.log_img[shadow_x, shadow_y]

        # Compute the isd
        pixel_diff = lit_values - shadow_values
        norms = np.linalg.norm(pixel_diff, axis = 1, keepdims = True)
        isds = pixel_diff / norms

        self.isd_map = isds[self.closest_annotation_map]

    def display_image_with_regions(self):
        """
        Displays an image that highlights what ISD is being used for each pixel
        """
        # Assign random colors to each region based on the number of midpoints
        num_regions = len(np.unique(self.closest_annotation_map))
        region_colors = np.random.randint(0, 65535, (num_regions, 3), dtype=np.uint16)

        # Create a blank color overlay
        overlay = np.zeros_like(self.rgb_img, dtype=np.uint16)

        # Assign each region its color
        for region_idx in range(num_regions):
            overlay[self.closest_annotation_map == region_idx] = region_colors[region_idx]

        alpha = 0.6
        beta = (1.0 - alpha)
        blended_image = cv2.addWeighted(self.rgb_img, alpha, overlay, beta, 0)

        # Display the result
        cv2.imshow("Image with Regions", blended_image)

    ###################################################################################################################
    # Methods for using Weighted Mean
    ###################################################################################################################
    def get_annotation_weights(self) -> None:
        """
        Computes a weight for each annotation for each pixel based on its distance to each pixel.
        weight_i = dist_i / sum_dist_i^n
        """
        # Calculate midpoints between the lit and shadow pairs
        midpoints = np.array([((x1 + x2) / 2, (y1 + y2) / 2) 
                              for (x1, y1), (x2, y2) in zip(self.lit_pixels, self.shadow_pixels)])

        # Get the number of annotations
        num_isds = midpoints.shape[0]

        # Initialize closest annotation mapper
        annotation_weight_map = np.zeros((self.rgb_img.shape[0], self.rgb_img.shape[1], num_isds), dtype=np.float32)
        
        # Find the distances from each pixel to each annotation midpoint
        for i in range(self.rgb_img.shape[0]):
            for j in range(self.rgb_img.shape[1]):

                # Get the point of interest
                point_of_interest = np.array([i, j])

                # Compute Euclidean distance between the point and the midpoints
                distances = np.linalg.norm(midpoints - point_of_interest, axis = 1)

                # Avoid division by zero for the point and itself.
                distances[distances == 0] = 1e-6

                # Invert distances to get higher values closer the midpoint
                inverted_distances = 1 / distances

                # Compute the weights ~ the proportion of the total distance.
                weights = inverted_distances / np.sum(inverted_distances)

                # Assign the weights to the current pixel in annotation_weight_map
                annotation_weight_map[i, j, :] = weights
        
        # Update the map attribute
        print(annotation_weight_map[0, 0, :])
        print(np.sum(annotation_weight_map[0, 0, :]))
        self.annotation_weight_map = annotation_weight_map

    def compute_weighted_mean_isd_per_pixel(self) -> None:
        """
        Computes the ISD for each pixel using the weighted mean of ISDs from each
        annotation.
        Weight is computed using the distance to each annotation.
        """
        # Unpack the tuples of pixel coordinates
        lit_x, lit_y = zip(*self.lit_pixels) 
        shadow_x, shadow_y = zip(*self.shadow_pixels)

        # Get the pixel values
        lit_values = self.log_img[lit_x, lit_y]
        shadow_values = self.log_img[shadow_x, shadow_y]

        # Compute the isd
        pixel_diff = lit_values - shadow_values
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

    def get_annotation_weights_cv(self) -> None:
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

    ###################################################################################################################
    # Methods for using Overall Mean
    ###################################################################################################################
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

    ###################################################################################################################
    # General Methods
    ###################################################################################################################
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
        Displays the processed image
        """
        cv2.imshow("Processed Image", self.linear_converted_log_chroma_8bit)
        cv2.resizeWindow("Processed Image", 600, 600)

    def process_img(self, image_path, clicks) -> None:
        """
        Execution function
        """
        self.set_img_rbg(image_path)
        self.convert_img_to_log_space()
        self.set_lit_shadow_pixels_list(clicks)

        # Using weight mean ISD
        if self.method == "weighted":
            print(f"Using weighted mean...")
            self.get_annotation_weights_cv()
            self.compute_weighted_mean_isd_per_pixel()
            self.project_to_plane_locally()
            self.log_to_linear()
            self.convert_16bit_to_8bit()
            self.display_processed_img()

        # Using nearest neighbor ISD
        elif self.method == "nearest_neighbor":
            self.find_closest_annotation()
            self.compute_localized_isd()
            self.project_to_plane_locally()
            self.log_to_linear()
            self.convert_16bit_to_8bit()
            self.display_processed_img()

        # Using overall mean ISD
        else:
            self.compute_isd()
            self.project_to_plane()
            self.log_to_linear()
            self.convert_16bit_to_8bit()
            self.display_processed_img()

if __name__== "__main__":
    pass




    