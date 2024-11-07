""" 
Image Annotator
"""
import cv2
import os
import csv
import shutil
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET


from new_process_image_class_combined_v2 import LogChromaticity

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
        self.image_folder = None
        self.high_quality_dir = None
        self.low_quality_dir = None
        self.duplicates_dir = None
        self.image_error_dir = None
        self.image_drop_dir = None
        self.isd_maps_dir = None
        

        # CSV file
        self.csv_file = None
        self.xml_file = None

        # Image of interest
        self.img = None
        self.image_path = None # Used to read a UNCHANAGED version for processing
        self.processed_img = None

        # Processing class object
        # self.img_processor = None
        self.img_processor = LogChromaticity()


    def set_directories(self, directories) -> None:
        """
        Create the target directories if they do not exist, set to attributes.  
        """
        for attribute_name, directory_path in directories.items():
            os.makedirs(directory_path, exist_ok=True)
            setattr(self, attribute_name, directory_path)
            print(f"Directory created or already exists: {directory_path}")

   
    def set_cvs_file(self, csv_file_path) -> None:
        """
        Create the cvs file if it does not exist; to csv_file attribute
        """
        if not os.path.exists(csv_file_path):
            with open(csv_file_path, 'w', newline='') as csvfile:
                self.csvwriter = csv.writer(csvfile)
        self.csv_file = csv_file_path

    def set_xml_file(self, xml_file_path) -> None:
        """
        Create the XML file with a root element if it does not exist; assigns to xml_file attribute.
        
        Parameters:
            xml_file_path (str): Path to the XML file.
        """
        # Check if the XML file already exists
        if not os.path.exists(xml_file_path):
            # Create the root element
            root = ET.Element("annotations")
            tree = ET.ElementTree(root)
            
            # Write the root element to the file to initialize it
            with open(xml_file_path, 'wb') as xmlfile:
                tree.write(xmlfile, encoding="utf-8", xml_declaration=True)
        
        # Assign the file path to the attribute
        self.xml_file = xml_file_path


    def is_complete(self) -> bool:
        """
        Check if a minimum of 1 pairs are annotated (12 clicks minimum).
        """
        if (self.click_count >= 2) and (self.click_count % 2 == 0):
            return True
        else:
            print("You must select at least one pair of lit/shadow pixels.")

    def show_message(self) -> None:
        """
        Show a message indicating which point is expected next.
        """
        if self.click_count > 0:
            pair_num = (self.click_count // 2) + 1
            if self.click_count % 2 == 0:
                print(f'Click lit patch for pair {pair_num}')
            else:
                print(f'Click shadow patch for pair {pair_num}')


    def write_to_csv(self, image_name) -> None:
        """ 
        Writes image name and annotations to csv file.

        Parameters:
            image_name (str): Current image filename.
        """
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [image_name.lower()] + [item for pair in self.clicks for item in pair]
            writer.writerow(row)

    def write_to_xml(self, image_name, target_directory) -> None:
        """ 
        Writes image name, target directory, and annotations to an XML file.
        If no annotations (clicks) are present, only the image and target directory are saved.

        Parameters:
            image_name (str): Current image filename.
            target_directory (str): Directory where the image is saved.
        """
        # Check if image name and target directory are provided
        if not image_name or not target_directory:
            print("Error: image_name or target_directory is empty.")
            return

        # Create the root element
        root = ET.Element("annotations")
        
        # Add image element with attributes for name and target directory
        image_element = ET.SubElement(root, "image")
        image_element.set("name", image_name.lower())
        image_element.set("target_directory", target_directory)
        image_element.set("patch_size", self.img_processor.get_patch_size())
        image_element.set("anchor_point", self.img_processor.get_anchor_point())

        # Only add click elements if self.clicks is not empty
        if self.clicks:
            for i, (lit_row, lit_col, shad_row, shad_col) in enumerate(self.clicks, start=1):
                click_element = ET.SubElement(image_element, "click", id=str(i))
                
                # Set lit and shadow points as sub-elements with row and col attributes
                lit_element = ET.SubElement(click_element, "lit")
                lit_element.set("row", str(lit_row))
                lit_element.set("col", str(lit_col))
                
                shad_element = ET.SubElement(click_element, "shadow")
                shad_element.set("row", str(shad_row) if shad_row is not None else "")
                shad_element.set("col", str(shad_col) if shad_col is not None else "")

        # Convert the XML tree to a string and write it to the XML file
        tree = ET.ElementTree(root)
        with open(self.xml_file, mode='a') as file:
            tree.write(file, encoding="unicode", xml_declaration=True)
        print(f"Data successfully written to {self.xml_file}")

    def move_image(self, image_path, destination_folder) -> None:
        """
        Moves image to specified directory.

        Parameters:
            image_path (str): Current image file path.
            destination_folder (str): Destination directory path.
        """
        try:
            shutil.move(image_path, destination_folder)
            print(f"Moved image {os.path.basename(image_path)} to {destination_folder}.")
        except Exception as e:
            print(f"Error moving image {image_path}: {e}")


    def process_image(self) -> None:
        """
        """
        # self.img_processor = LogChromaticity()
        self.processed_img = self.img_processor.process_img(self.image_path, self.clicks)

    def update_patch(self, size) -> None:
        """
        """
        if self.processed_img is not None:
            self.processed_img = self.img_processor.update_patch_size((size, size))
            self.display_images()

    def update_anchor(self, val) -> None:
        """
        """
        if self.processed_img is not None:
            self.processed_img = self.img_processor.update_anchor_point(val)
            self.display_images()
       
    
    def display_images(self, clickable=False) -> None:
        """ 
        """
        try:
            if self.processed_img is not None:
                combined_image = np.hstack((self.img, self.processed_img))
            else:
                combined_image = np.hstack((self.img, np.zeros_like(self.img)))

            window_name = "Original Image ---------------------------------------------------------------- Processed Image"
            #cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, combined_image)
            cv2.resizeWindow(window_name, 1000, 1000)
            
            if clickable:
                cv2.setMouseCallback('Original Image ---------------------------------------------------------------- Processed Image', self.click_event)

        except Exception as e:
                    print(f"Error with self.display_images(): {e}")
        
    def reset(self) -> None:
        """
        Reset the annotation state for the current image.
        """
        self.click_count = 0
        self.clicks = []


    def add_click(self, row, col) -> None:
        """
        Add a click (light or shadow) based on the click count.

        Parameters:
            row (int): Click location row (y) index value.
            col (int): Click location col (x) index value.
        """
        if self.click_count % 2 == 0:
            # Add a new lit point
            self.clicks.append((row, col, None, None))  # Placeholder for shadow
        else:
            # Add shadow point to the last lit point
            self.clicks[-1] = (self.clicks[-1][0], self.clicks[-1][1], row, col)
        self.click_count += 1

    def remove_click(self) -> None:
        """
        Removes the last recorded click.
        - If the last click was a shadow point, it removes just the shadow point.
        - If the last click was a lit point without a shadow point, it removes the entire lit point entry.
        """
        if self.click_count == 0:
            print("No clicks to remove.")
            return
        
        # remove current annotation pair
        self.clicks.pop()

        # update click count
        if self.click_count % 2 == 1:
            self.click_count -= 1
        if self.click_count % 2 == 0:
            self.click_count -= 2      


    # Mouse event callback function
    def click_event(self, event, x, y, flags, params):
        """
        Records left or right click event. Left click selects a pixel for annotation, right click removes the most recent pixel or pixel pair. 

        Parameters:
            x (int): CLick location x coordinate.
            y (int): click lcoation y coordinate.

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
            
            # Update display images
            if self.click_count >= 2 and self.click_count % 2 == 0:
                self.process_image()            
            self.display_images()

    def save_pixel_map(self, image_name) -> None:
        """
        """
        
        pixel_map = self.img_processor.get_isd_pixel_map()
        image_name_no_ext = os.path.splitext(image_name)[0]
        save_path = os.path.join(self.isd_maps_dir, f"{image_name_no_ext}_isd.png")
        cv2.imwrite(save_path, pixel_map)
        print(f"ISD Pixel Map saved at {save_path}")


    def export_completed_image(self, image_name, loc) -> None:
        """
        """
        print(f"Saving processed image to {loc}.")
        self.write_to_xml(image_name=image_name, target_directory=loc)
        self.move_image(self.image_path, loc)
        self.save_pixel_map(image_name)
        self.processed_img = None



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
                        self.move_image(self.image_path, self.image_error_dir)
                        continue

                    self.reset() # Sets click count to zero
                    self.display_images(clickable=True)
                    cv2.namedWindow("Log Space Widget")
                    cv2.createTrackbar("Anchor Point", "Log Space Widget", 104, 111, self.update_anchor)
                    cv2.createTrackbar("Patch Size", "Log Space Widget", 1, 31, self.update_patch)

                    print("""Press: 
                            \n    'h' to save image HIGH quality annotations 
                            \n    'l' to save image LOW quality annotations
                            \n    'd' to save image duplicate image annotations 
                            \n    'r' to redo the annotations 
                            \n    't' to drop the image for quality reasons. 
                            \n    'q' to quit the annotator.
                            """)                    
                    while True:

                        key = cv2.waitKey(0)

                        if key == ord('h') and self.is_complete():
                            loc = self.high_quality_dir
                            self.export_completed_image(image_name, loc)
                            break  # Move to the next image

                        elif key == ord('l') and self.is_complete():
                            loc = self.low_quality_dir
                            self.export_completed_image(image_name, loc)
                            break  # Move to the next image

                        elif key == ord('d') and self.is_complete():
                            loc = self.duplicates_dir
                            self.export_completed_image(image_name, loc)
                            break  # Move to the next image
                        
                        elif key == ord('t'):
                            print(f"Trashing {image_name}. Bad Quality.")
                            loc = self.image_drop_dir
                            self.export_completed_image(image_name, loc)
                            break

                        elif key == ord('r'):
                            print(f"Starting over for {image_name}. Redo the annotations.")
                            self.img = cv2.imread(self.image_path)  # Reload the image to clear drawn points
                            self.reset() # Sets click count to zero
                            self.display_images(clickable=True)

                        elif key == ord('q'):
                            print("Quitting.")
                            cv2.destroyAllWindows()
                            return

                        else:
                            print("""
                                  Press: 
                                  \n    'h' to save image HIGH quality annotations 
                                  \n    'l' to save image LOW quality annotations
                                  \n    'd' to save image duplicate image annotations 
                                  \n    'r' to redo the annotations 
                                  \n    't' to drop the image for quality reasons. 
                                  \n    'q' to quit the annotator.
                                  """)

                    cv2.destroyAllWindows()

                except Exception as e:
                    print(f"An error occurred with image {image_name}: {e}")
                    self.move_image(self.image_path, self.image_error_dir)
        print("All images processed and data saved.")



##################################################################################################################################

# Initialize paths
# Initialize paths
image_folder = 'new_annotator_dev/test_images/unprocessed'  
high_quality_dir = 'new_annotator_dev/test_images/processed/high_quality'
low_quality_dir = 'new_annotator_dev/test_images/processed/low_quality'
duplicates_dir = 'new_annotator_dev/test_images/processed/duplicates'
image_error_dir = 'new_annotator_dev/test_images/image_error'
image_drop_dir = 'new_annotator_dev/test_images/bad_images'
isd_maps_dir = 'new_annotator_dev/test_images/processed/isd_maps'

directories = {
    "image_folder": image_folder,
    "high_quality_dir": high_quality_dir,
    "low_quality_dir":low_quality_dir,
    "duplicates_dir": duplicates_dir,
    "image_error_dir": image_error_dir,
    "image_drop_dir": image_drop_dir,
    "isd_maps_dir": isd_maps_dir
}

csv_file_path = 'new_annotator_dev/test_images/test.csv'
xml_file_path = 'new_annotator_dev/test_images/annotations.xml'

image_annotator = AnnotationManager(image_folder = image_folder)

image_annotator.set_directories(directories)

image_annotator.set_cvs_file(csv_file_path = csv_file_path)
image_annotator.set_xml_file(xml_file_path)
image_annotator.annotate_images()