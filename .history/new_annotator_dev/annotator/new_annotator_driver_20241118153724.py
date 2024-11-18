""" 
Image Annotator Driver

Nelson Farrell and Michael Massone
7180 Advanced Perception
November 10, 2024
Bruce Maxwell, PhD.

This is the driver file for the AnnotationManager 
"""
# Modules
from new_process_image_class_combined_v2 import LogChromaticity
from new_annotator_combined_v2 import AnnotationManager


# Initialize paths
# These variables need to be updated when you use the program

# This folder contains the images to be processed
image_folder = 'test'

# These are destination directories and files
# Directories
high_quality_dir = 'test/processed/high_quality'
low_quality_dir = 'test/processed/low_quality'
duplicates_dir = 'test/processed/duplicates'
image_error_dir = 'test/processed/image_error'
image_drop_dir = 'test/processed/bad_images'
isd_maps_dir = 'data/folder_3/processed/isd_maps'

# Files
csv_file_path = 'test/processed/annotation_csv/annotations_csv.csv'
xml_file_path = 'test/processed/annotation_xml/annotations.xml'

# NONE OF THIS NEEDS TO BE UPDATED!!!!
directories = {
    "image_folder": image_folder,
    "high_quality_dir": high_quality_dir,
    "low_quality_dir":low_quality_dir,
    "duplicates_dir": duplicates_dir,
    "image_error_dir": image_error_dir,
    "image_drop_dir": image_drop_dir,
    "isd_maps_dir": isd_maps_dir
}

image_annotator = AnnotationManager()
image_annotator.set_directories(directories)
image_annotator.set_cvs_file(csv_file_path = csv_file_path)
image_annotator.set_xml_file(xml_file_path)
image_annotator.annotate_images()