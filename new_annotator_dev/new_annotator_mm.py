""" 
Image Annotator
"""
import cv2
import os
import csv
import shutil
import numpy as np
from pathlib import Path

from new_process_image_class_mm import LogChromaticity

# Class to manage the annotation state for each image
class AnnotationManager:
    """ 
    """
    def __init__(self, image_folder):
        """ 
        """
        # Image click data
        self.click_count = 0
        self.clicks = []  # List to store (lit_row, lit_col, shad_row, shad_col) pairs

        # Directories
        self.image_folder = image_folder
        self.processed_folder = None
        self.image_error_folder = None
        self.bad_image_folder = None

        # CSV file
        self.csv_file = None

        # Image of interest
        self.img = None
        self.image_path = None # Used to read a UNCHANAGED version for processing
        self.processed_img = None

        # Processing class object
        self.img_processor = None

    def set_directories(self, processed_folder, image_error_folder, bad_image_folder):
        """
        Create the target directories if they do not exist, set to attributes.  
        """
        # Create folders if they don't exist
        os.makedirs(processed_folder, exist_ok=True)
        os.makedirs(image_error_folder, exist_ok=True)
        os.makedirs(bad_image_folder, exist_ok=True)

        # Set to object attributes
        self.processed_folder = processed_folder
        self.image_error_folder = image_error_folder
        self.bad_image_folder = bad_image_folder
        return None
    
    def set_cvs_file(self, csv_file_path) -> None:
        """
        Create the cvs file if it does not exist; to csv_file attribute
        """
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                self.csvwriter = csv.writer(csvfile)
        self.csv_file = csv_file_path
        return None

    def reset(self):
        """
        Reset the annotation state for the current image.
        """
        self.click_count = 0
        self.clicks = []

    def add_click(self, row, col):
        """
        Add a click (light or shadow) based on the click count.
        """
        if self.click_count % 2 == 0:
            # Add a new lit point
            self.clicks.append((row, col, None, None))  # Placeholder for shadow
        else:
            # Add shadow point to the last lit point
            self.clicks[-1] = (self.clicks[-1][0], self.clicks[-1][1], row, col)
        self.click_count += 1
    def remove_click(self):
        """
        Removes the last recorded click.
        - If the last click was a shadow point, it removes just the shadow point.
        - If the last click was a lit point without a shadow point, it removes the entire lit point entry.
        """
        if self.click_count == 0:
            print("No clicks to remove.")
            return
        
        # If the last click was a shadow point
        if self.click_count % 2 == 1:
            # Remove the shadow point (set to None) but keep the lit point
            self.clicks[-1] = (self.clicks[-1][0], self.clicks[-1][1], None, None)
        else:
            # If the last click was a lit point (without shadow), remove the entire entry
            self.clicks.pop()

    def is_complete(self):
        """
        Check if a minimum of 6 pairs are annotated (12 clicks minimum).
        """
        return (self.click_count >= 12) and (self.click_count % 2 == 0)

    def show_message(self):
        """
        Show a message indicating which point is expected next.
        """
        if self.click_count < 12:
            pair_num = (self.click_count // 2) + 1
            if self.click_count % 2 == 0:
                print(f'Click lit patch for pair {pair_num}')
            else:
                print(f'Click shadow patch for pair {pair_num}')


    def write_to_csv(self, image_name):
        """ 
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [image_name.lower()] + [item for pair in self.clicks for item in pair]
            writer.writerow(row)

    def move_image(self, image_path, destination_folder):
        """
        """
        try:
            shutil.move(image_path, destination_folder)
            print(f"Moved image {os.path.basename(image_path)} to {destination_folder}.")
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")


    def process_image(self) -> None:
        """
        """
        self.img_processor = LogChromaticity()
        self.processed_img = self.img_processor.process_img(self.image_path, self.clicks)

    def update_patch(self, size) -> None:
        """
        """
        if self.img_processor is not None:
            self.processed_img = self.img_processor.update_patch_size((size, size))
            self.display_images()

    def update_anchor(self, val) -> None:
        """
        """
        if self.img_processor is not None:
            self.processed_img = self.img_processor.update_anchor_point(val)
            self.display_images()

    def isd_method(self) -> None:
        """
        """
        pass
    
    def display_images(self, clickable=False) -> None:
        """ 
        """
        if self.processed_img is not None:
            combined_image = np.hstack((self.img, self.processed_img))
        else:
            combined_image = np.hstack((self.img, np.zeros_like(self.img)))

        cv2.imshow("Original Image ---------------------------------------------------------------- Processed Image", combined_image)
        cv2.resizeWindow("Original Image ---------------------------------------------------------------- Processed Image", 600, 600)

        # if self.processed_img is not None:
        #     cv2.createTrackbar("Anchor Point", "Original Image -- Processed Image", 0, 111, self.update_anchor)
        
        if clickable:
            cv2.setMouseCallback('Original Image ---------------------------------------------------------------- Processed Image', self.click_event)

    
    # Mouse event callback function
    def click_event(self, event, x, y, flags, params):
        """
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            row, col = y, x  # Convert (x, y) to (row, col)
            self.add_click(row, col)

            # Alternate between two colors for the circles ~ Green for light, Red for shadow
            color = (0, 0, 255) if self.click_count % 2 == 0 else (0, 255, 0)
            cv2.circle(self.img, (x, y), 5, color, 1)

            # Draw a line between every pair of clicks
            if self.click_count % 2 == 0:
                cv2.line(self.img, (self.clicks[-1][1], self.clicks[-1][0]),
                         (self.clicks[-1][3], self.clicks[-1][2]), (255, 255, 255), 1)

            # cv2.imshow('image', self.img)
            self.display_images()

            # Process image after every pair of click have been made
            if self.click_count >= 2 and self.click_count % 2 == 0:
                print("Entering process image!")
                self.process_image()
                self.display_images()
            self.show_message()

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.remove_click()

            # Redraw the image to reflect the removed click
            self.img = cv2.imread(self.image_path)  # Reload original image to clear previous annotations
            for idx, (lit_row, lit_col, shad_row, shad_col) in enumerate(self.clicks):
                lit_color = (0, 255, 0)  # Green for lit points
                shad_color = (0, 0, 255)  # Red for shadow points
                cv2.circle(self.img, (lit_col, lit_row), 5, lit_color, 1)

                if shad_row is not None and shad_col is not None:
                    cv2.circle(self.img, (shad_col, shad_row), 5, shad_color, 1)
                    cv2.line(self.img, (lit_col, lit_row), (shad_col, shad_row), (255, 255, 255), 1)        
            self.display_images()


    def annotate_images(self):
        """ 
        """
        for image_name in os.listdir(self.image_folder):
            if image_name.endswith(('tif', 'tiff')):
                self.image_path = os.path.join(self.image_folder, image_name)

                try:
                    self.img = cv2.imread(self.image_path)

                    if self.img is None:
                        print(f"Cannot open image: {image_name}. Moving to wrong folder.")
                        self.move_image(self.image_path, self.image_error_folder)
                        continue

                    self.reset() # Sets click count to zero
                    self.display_images(clickable=True)
                    cv2.namedWindow("Log Space Widget")
                    cv2.createTrackbar("Anchor Point", "Log Space Widget", 0, 111, self.update_anchor)
                    cv2.createTrackbar("Patch Size", "Log Space Widget", 1, 31, self.update_patch)
                    cv2.createTrackbar("ISD", "Log Space Widget", 0, 1, self.isd_method)
                    cv2.resizeWindow("Anchor Point Adjustment", 200, 200)

                    print("Ensure at least 6 pairs have been selected.")
                    print("Press: \n'k' to keep image annotations \n'r' to redo the annotations \n'd' to drop the image for quality reasons. \n'q' to quit the annotator.")
                    
                    while True:

                        self.display_images()
                        key = cv2.waitKey(0)

                        if key == ord('c') and self.is_complete(): 
                            self.write_to_csv(image_name)
                            self.move_image(self.image_path, self.processed_folder)
                            break  # Move to the next image
                        
                        elif key == ord('d'):
                            print(f"Dropping {image_name}. Bad Quality.")
                            self.move_image(self.image_path, self.bad_image_folder)
                            break

                        elif key == ord('r'):
                            print(f"Starting over for {image_name}. Redo the annotations.")
                            self.img = cv2.imread(self.image_path)  # Reload the image to clear drawn points
                            self.reset() # Sets click count to zero

                        elif key == ord('q'):
                            print("Quitting.")
                            cv2.destroyAllWindows()
                            return

                        else:
                            print("Ensure at least 6 pairs have been selected.")
                            print("Press: \n'k' to keep image annotations \n'r' to redo the annotations \n'd' to drop the image for quality reasons. \n'q' to quit the annotator.")

                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"An error occurred with image {image_name}: {e}")
                    self.move_image(self.image_path, self.image_error_folder)
        print("All images processed and data saved.")



##################################################################################################################################

# Initialize paths
image_folder = 'test_images'  
processed_folder = 'test_images/processed'
image_error_folder = 'test_images/image_error'
image_drop_folder = 'test_images/drop'
csv_file_path = 'test_images/test.csv'

image_annotator = AnnotationManager(image_folder = image_folder)

image_annotator.set_directories(processed_folder = processed_folder, 
                                image_error_folder = image_error_folder, 
                                bad_image_folder = image_drop_folder)

image_annotator.set_cvs_file(csv_file_path = csv_file_path)
image_annotator.annotate_images()