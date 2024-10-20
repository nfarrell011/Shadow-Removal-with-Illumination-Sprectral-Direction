import cv2
import os
import csv
import shutil

# Class to manage the annotation state for each image
class AnnotationManager:
    def __init__(self):
        self.click_count = 0
        self.clicks = []  # List to store (lit_row, lit_col, shad_row, shad_col) pairs

    def reset(self):
        """Reset the annotation state for the current image."""
        self.click_count = 0
        self.clicks = []

    def add_click(self, row, col):
        """Add a click (light or shadow) based on the click count."""
        if self.click_count % 2 == 0:
            # Add a new lit point
            self.clicks.append((row, col, None, None))  # Placeholder for shadow
        else:
            # Add shadow point to the last lit point
            self.clicks[-1] = (self.clicks[-1][0], self.clicks[-1][1], row, col)
        self.click_count += 1

    def is_complete(self):
        """Check if all 6 pairs are annotated (12 clicks in total)."""
        return self.click_count == 12

    def show_message(self):
        """Show a message indicating which point is expected next."""
        if self.click_count < 12:
            pair_num = (self.click_count // 2) + 1
            if self.click_count % 2 == 0:
                print(f'Click pair {pair_num} lit')
            else:
                print(f'Click pair {pair_num} shadow')
        else:
            print("All 6 pairs completed. Press 'c' to confirm or 'r' to redo.")

# File handling and utility functions
def write_to_csv(file_path, image_name, data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [image_name.lower()] + [item for pair in data for item in pair]
        writer.writerow(row)

def move_image(image_path, destination_folder):
    try:
        shutil.move(image_path, destination_folder)
        print(f"Moved image {os.path.basename(image_path)} to {destination_folder}.")
    except Exception as e:
        print(f"Error moving image {image_path}: {e}")

# Mouse event callback function
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        row, col = y, x  # Convert (x, y) to (row, col)
        annotation_manager.add_click(row, col)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('image', img)
        annotation_manager.show_message()

# Main annotation loop
def annotate_images(image_folder, done_folder, wrong_folder, csv_file_path):
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('tif', 'tiff')):
            image_path = os.path.join(image_folder, image_name)

            try:
                global img
                img = cv2.imread(image_path)

                if img is None:
                    print(f"Cannot open image: {image_name}. Moving to wrong folder.")
                    move_image(image_path, wrong_folder)
                    continue

                annotation_manager.reset()
                cv2.imshow('image', img)
                annotation_manager.show_message()  # Show initial message

                while True:
                    cv2.imshow('image', img)
                    cv2.setMouseCallback('image', click_event)

                    key = cv2.waitKey(0)

                    if key == ord('c') and annotation_manager.is_complete():  # Changed key to 'c'
                        write_to_csv(csv_file_path, image_name, annotation_manager.clicks)
                        move_image(image_path, done_folder)
                        break  # Move to the next image

                    elif key == ord('r'):
                        print(f"Starting over for {image_name}. Redo the annotations.")
                        img = cv2.imread(image_path)  # Reload the image to clear drawn points
                        annotation_manager.reset()
                        cv2.imshow('image', img)

                    else:
                        print("Press 'c' to confirm or 'r' to redo. Ensure all 12 points are clicked.")

                cv2.destroyAllWindows()

            except Exception as e:
                print(f"An error occurred with image {image_name}: {e}")
                move_image(image_path, wrong_folder)

    print("All images processed and data saved.")

# Initialize paths
image_folder = 'data/folder_8'  
done_folder = 'data/folder_8/done'
wrong_folder = 'data/folder_8/wrong'
csv_file_path = 'data/folder_8/annotation_folder_8.csv'

# Create folders if they don't exist
os.makedirs(done_folder, exist_ok=True)
os.makedirs(wrong_folder, exist_ok=True)

# Create the CSV file and header if it doesn't exist
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['filename']
        for i in range(6):
            header += [f'lit_row{i+1}', f'lit_col{i+1}', f'shad_row{i+1}', f'shad_col{i+1}']
        writer.writerow(header)

# Create the AnnotationManager instance
annotation_manager = AnnotationManager()

# Start the annotation process
annotate_images(image_folder, done_folder, wrong_folder, csv_file_path)
