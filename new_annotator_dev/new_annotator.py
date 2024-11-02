""" 
Image Annotator
"""
import cv2
import os
import csv
import shutil
from pathlib import Path

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
                print(f'Click pair {pair_num} lit')
            else:
                print(f'Click pair {pair_num} shadow')
        else:
            print("All 6 pairs completed. Press 'c' to confirm or 'r' to redo.")

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

    # Mouse event callback function
    def click_event(self, event, x, y, flags, params):
        """
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            row, col = y, x  # Convert (x, y) to (row, col)
            self.add_click(row, col)

            # Alternate between two colors for the circles ~ Green for light, Red for shadow
            color = (0, 255, 0) if self.click_count % 2 == 0 else (0, 0, 255)
            cv2.circle(self.img, (x, y), 5, color, 1)

            # Draw a line between every pair of clicks
            if self.click_count % 2 == 0:
                cv2.line(self.img, (self.clicks[-1][1], self.clicks[-1][0]),
                         (self.clicks[-1][3], self.clicks[-1][2]), (255, 255, 255), 1)

            cv2.imshow('image', self.img)
            self.show_message()

    def process_image(self):
        """
        """
        cv2.imshow("Processed Image", self.img)

        # Main annotation loop
    def annotate_images(self):
        """ 
        """
        for image_name in os.listdir(self.image_folder):
            if image_name.endswith(('tif', 'tiff')):
                image_path = os.path.join(self.image_folder, image_name)

                try:
                    self.img = cv2.imread(image_path)

                    if self.img is None:
                        print(f"Cannot open image: {image_name}. Moving to wrong folder.")
                        self.move_image(image_path, self.image_error_folder)
                        continue

                    self.reset()
                    cv2.imshow('image', self.img)
                    self.show_message()  # Show initial message

                    while True:
                        cv2.imshow('image', self.img)
                        cv2.setMouseCallback('image', self.click_event)
                        self.process_image(self.img)

                        key = cv2.waitKey(0)

                        if key == ord('k') and self.is_complete():  # Changed key to 'c'
                            self.write_to_csv(image_name)
                            self.move_image(image_path, self.processed_folder)
                            break  # Move to the next image
                        
                        elif key == ord('d'):
                            print(f"Dropping {image_name}. Bad Quality.")
                            self.move_image(image_path, self.bad_image_folder)

                        elif key == ord('r'):
                            print(f"Starting over for {image_name}. Redo the annotations.")
                            img = cv2.imread(image_path)  # Reload the image to clear drawn points
                            self.reset()
                            cv2.imshow('image', self.img)

                        else:
                            print("Press 'k' keep image annotations, or 'r' to redo the annotations, or 'd' to drop the image for quality reasons.\
                                   Ensure all 12 points are clicked.")

                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"An error occurred with image {image_name}: {e}")
                    self.move_image(image_path, self.wrong_folder)

        print("All images processed and data saved.")

##################################################################################################################################

# Initialize paths
image_folder = 'new_annotator_dev/test_images'  
processed_folder = 'new_annotator_dev/test_images/processed'
image_error_folder = 'new_annotator_dev/test_images/image_error'
image_drop_folder = 'new_annotator_dev/test_images/drop'
csv_file_path = 'new_annotator_dev/test_images/test.csv'

image_annotator = AnnotationManager(image_folder = image_folder)

image_annotator.set_directories(processed_folder = processed_folder, 
                                image_error_folder = image_error_folder, 
                                bad_image_folder = image_drop_folder)

image_annotator.set_cvs_file(csv_file_path = csv_file_path)
image_annotator.annotate_images()