"""
This file will fix the ISD map issue. 
"""
import xmltodict
import pandas as pd
import numpy as np 
import cv2
import os

def read_in_xml_data_as_dict(xml_file_path):
    """
    Reads in an XML doc as a dict
    """
    # Read the XML file
    with open(xml_file_path, "r") as file:
        xml_content = file.read()

    # Parse the XML content
    data = xmltodict.parse(xml_content)
    return data

def extract_image_data(data, image_name):
    """
    Extracts the image data for one image from the XML data dict 
    """
    try:
        annotations = data['root']['dataset']['annotations']
        lit_coords = []
        shadow_coords = []
        patch_size = None
        anchor_point = None
        
        # Get the image annotations
        for annotation in annotations:
            image = annotation['image']
            if image['@name'] == image_name:
                patch_size = image['@patch_size']
                anchor_point = image['@anchor_point']
                clicks = image['click']
                
                # Handle single click or list of clicks
                if isinstance(clicks, dict):
                    clicks = [clicks]
                
                for click in clicks:
                    lit_coords.append((int(click['lit']['@row']), int(click['lit']['@col'])))
                    shadow_coords.append((int(click['shadow']['@row']), int(click['shadow']['@col'])))
        
        # Update the dtype of patch_size and anchor_point; currently they are strings
        patch_size = tuple(map(int, patch_size.strip("()").split(", ")))
        anchor_point = list(map(float, anchor_point.strip("[]").split()))
    except Exception:
        print(f"{image_name} NOT FOUND IN XML")

    return lit_coords, shadow_coords, patch_size, anchor_point

class LogChromaticity:
    """
    """
    def __init__(self) -> None:
        """
        """
        self.method = 'mean'
        self.anchor_point= None
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

        # Directoires
        self.isd_map_dir = None
        self.images_dir = None

    def set_isd_map_dir(self, isd_map_dir):
        """ 
        """
        os.makedirs(isd_map_dir, exist_ok = True)
        self.isd_map_dir = isd_map_dir
        
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
    
    def set_lit_shadow_pixels_list(self, lit_pixels, shadow_pixels) -> None:
        """ 
        """
        self.lit_pixels = lit_pixels
        self.shadow_pixels = shadow_pixels
        return None   
    
    def set_patch_size(self, patch_size):
        """ 
        """
        self.patch_size = patch_size

    def set_anchor_point(self, anchor_point):
        """ 
        """
        self.anchor_point = np.array(anchor_point)
    
    def get_patch_size(self):
        """
        Returns patch size.
        """
        return self.patch_size
    
    def get_isd_pixel_map(self):
        """
        Returns ISD pixel map as numpy array.
        """
        return self.isd_map
    
    def get_anchor_point(self):
        """
        Returns anchor point as np.array.
        """
        return self.anchor_point

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

        log_img = np.zeros_like(self.rgb_img, dtype = np.float32)
        log_img[self.rgb_img != 0] = np.log(self.rgb_img[self.rgb_img != 0])

        assert np.min(log_img) >= 0 and np.max(log_img) <= 11.1

        self.log_img = log_img

        return None
    
    ###################################################################################################################
    # Methods for using Weighted Mean
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
        patch_height, patch_width = patch_size[0], self.patch_size[1]

        # Calculate the start and end indices, ensuring they don't go out of bounds
        start_y = max(y - patch_height // 2, 0)
        end_y = min(y + patch_height // 2 + 1, self.log_img.shape[0])
        start_x = max(x - patch_width // 2, 0)
        end_x = min(x + patch_width // 2 + 1, self.log_img.shape[1])

        # Extract the patch and return its mean value
        patch = self.log_img[start_y:end_y, start_x:end_x]
        mean_value = np.mean(patch, axis=(0, 1))
        return mean_value
    
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
                              for (x1, y1), (x2, y2) in zip(self.lit_pixels, self.shadow_pixels) if x2 is not None and y2 is not None])

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
            #dist_output = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)

            # Add the distances for this annotation to the annotation_map
            annotation_weight_map[:, :, i] = dist

            # Display the distance transform result
            #cv2.imshow("Distance Transform", dist_output)

        # Transforms distances to proportions to use as the weights
        # annotation_weight_map = annotation_weight_map / annotation_weight_map.sum(axis = 2, keepdims = True)
        epsilon = 1e-10
        annotation_weight_map = annotation_weight_map / (annotation_weight_map.sum(axis = 2, keepdims = True) + epsilon)
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
        lit_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.lit_pixels if x is not None and y is not None])
        shadow_means = np.array([self.get_patch_mean((y, x), self.patch_size) for y, x in self.shadow_pixels if x is not None and y is not None])

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
    
    ###################################################################################################################
    # Methods for Post Processing Images
    ###################################################################################################################
    
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
        #img_normalized = cv2.normalize(self.linear_converted_log_chroma, None, 0, 255, cv2.NORM_MINMAX) # divide by 255, clip to 0-255, then convert to unit8
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

    def display_image_with_annotations(self):
        """ 
        Displays the image with annotations: circles for lit/shadow pixels and lines connecting them.
        """
        # Ensure the image is not None
        if self.rgb_img is None:
            raise ValueError("Image has not been loaded or set.")

        # Create a copy of the image to draw on
        annotated_image = self.rgb_img.copy()

        # Draw circles around lit pixels (green)
        for lit_y, lit_x in self.lit_pixels:
            cv2.circle(annotated_image, (lit_x, lit_y), radius=5, color=(0, 40000, 0), thickness=3)

        # Draw circles around shadow pixels (red)
        for sha_y, sha_x in self.shadow_pixels:
            cv2.circle(annotated_image, (sha_x, sha_y), radius=5, color=(0, 0, 40000), thickness=3)

        # Draw lines between corresponding lit and shadow pixels (blue)
        for (lit_y, lit_x), (sha_y, sha_x) in zip(self.lit_pixels, self.shadow_pixels):
            cv2.line(annotated_image, (lit_x, lit_y), (sha_x, sha_y), color=(40000, 0, 0), thickness=2)

        # Display the annotated image
        cv2.imshow("Annotations with Lines", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    ###################################################################################################################
    # Methods for using GUI Controls
    ###################################################################################################################
    def save_isd_map_to_png(self, image_name):
        """ 
        """
        isd_map = self.isd_map * 255
        image_name_no_ext = os.path.splitext(image_name)[0]
        save_path = os.path.join(self.isd_map_dir, f"{image_name_no_ext}_isd.png")
        cv2.imwrite(save_path, isd_map)


    def update_patch_size(self, size):
        """
        """
        self.patch_size = size
        print(f"Updated patch_size: {self.patch_size}")

        self.get_annotation_weights()
        self.compute_weighted_mean_isd_per_pixel()
        self.project_to_plane_locally()
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

        self.project_to_plane_locally()
        self.log_to_linear()
        self.convert_16bit_to_8bit()

        return self.linear_converted_log_chroma_8bit
    
    ###################################################################################################################
    # Methods for  Execution
    ###################################################################################################################
    def process_img(self, image_name, lit_pixels, shadow_pixels, images_dir, isd_map_dir, patch_size, anchor_point) -> np.array:
        """ 
        """
        self.set_images_dir(images_dir)
        self.set_img_rbg(image_name)
        self.set_isd_map_dir(isd_map_dir)
        #self.display_original_image()
        self.convert_img_to_log_space()
        self.set_lit_shadow_pixels_list(lit_pixels, shadow_pixels)
        #self.display_image_with_annotations()
        self.set_anchor_point(anchor_point)
        self.set_patch_size(patch_size)
        self.get_annotation_weights()
        self.compute_weighted_mean_isd_per_pixel()
        self.project_to_plane_locally()
        self.log_to_linear()
        self.convert_16bit_to_8bit()
        #self.display_image()
        self.save_isd_map_to_png(image_name)

        return self.linear_converted_log_chroma_8bit

def reprocess_all_the_images_in_a_folder(xml_data_dict, images_dir, isd_map_destination_dir):
    """
    """
    for file in os.listdir(images_dir):
        if file.endswith(".tif"):
            image_name = file

            # The image names are lower case in the XML doc.
            image_name_lower = image_name.lower()

            # Extract the data from the XML doc
            lit_pixels, shadow_pixels, patch_size, anchor_point = extract_image_data(xml_data_dict, image_name_lower)
            try:
                processor = LogChromaticity()
                processor.process_img(image_name, lit_pixels, shadow_pixels, images_dir, isd_map_destination_dir, patch_size, anchor_point)
            except Exception:
                print("Proceeding to next image...")

def main():
    """ 
    Here you can update for isd maps for an image folder.


    UPDATE THE VARIABLES!!!
    """
    # Set the path to XML file; this contains all the processed image data for all the images
    xml_file_path = "/Users/nelsonfarrell/Documents/Northeastern/7180/projects/spectral_ratio/annotations_folder_1_and_3.xml"

    # Set the path to where you want to store the isd maps; this also does not need to be changed for each image folder; high, low, dup.
    # But it does need to changed for outer image folder; i.e., folder_1, folder_3, ... 
    isd_map_destination_dir = "/Users/nelsonfarrell/Documents/Northeastern/7180/projects/spectral_ratio/data/folder_3/processed_folder_3/isd_maps"

    # Images dir are the processed image folders, high quality, low quality, duplicates. 
    # All need to processed separately.
    # The other variables can remain the same and it the results will follow our current setup.
    images_dir = "/Users/nelsonfarrell/Documents/Northeastern/7180/projects/spectral_ratio/data/folder_3/processed_folder_3/high_quality"

    # Get the XML data
    xml_data_dict = read_in_xml_data_as_dict(xml_file_path)

    # Process the images saving the isd_maps
    reprocess_all_the_images_in_a_folder(xml_data_dict, images_dir, isd_map_destination_dir)

if __name__ == "__main__":
    main()



