""" 
Nelson Farrell and Michael Massone  
CS7180 - Fall 2024  
Bruce Maxwell, PhD  
Final Project: ISD Convolutional Transformer  

Driver for data file reorganizer
"""
from data_file_reorganizer_class import DataFileReorganizer

def main():
    """
    Executes Reorganization
    """
    ################################ Params ################################
    # Control bools
    # Do you want to compute the mean and std?
    compute_mean_and_std = False

    # Do you want to get the min and max image dims?
    get_min_max_image_dims = False

    # Do you want to crop the images?
    crop_images = True

    # Params for step # 1
    parent_data_folder = "data"
    image_folder_list = ["folder_1", "folder_3"]
    image_quality_list = ["high_quality"]

    # Params for step # 2 and # 3
    # Folder name with training images
    # For computing channel wise mean and std AND getting min mx dims
    training_images_folder_name = "training_images"

    # Params Step # 3
    training_images_folder = "training_images"
    cropped_images_folder = "training_images_cropped"
    
    training_isd_folder = "training_isds"
    cropped_isd_folder = "training_isds_cropped"
    crop_size = 448
    sub_crop_size = 224

    ########################################################################
    # Instantiate DataFileReorganizer
    dfr = DataFileReorganizer()

    # Step # 1
    dfr.reorganize_data_files(parent_data_folder, image_folder_list, image_quality_list)

    # Step # 2
    if compute_mean_and_std:
        mean, std = dfr.get_channel_wise_mean_and_std(training_images_folder_name)

    # Step # 3
    if get_min_max_image_dims:
        max_W, max_H, min_W, min_H = dfr.get_min_max_dims(training_images_folder_name)

    # Step 4
    if crop_images:
        dfr.center_crop_and_split(training_images_folder, cropped_images_folder, crop_size, sub_crop_size)
        dfr.center_crop_and_split(training_isd_folder, cropped_isd_folder, crop_size, sub_crop_size)

if __name__ == "__main__":
    main()