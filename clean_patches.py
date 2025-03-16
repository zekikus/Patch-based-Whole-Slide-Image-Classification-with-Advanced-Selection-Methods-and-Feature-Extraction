from PIL import Image
import os
import glob
import numpy as np
import cv2

# Function to calculate mean and standard deviation of a patch
def calculate_patch_statistics(patch):
    mean_intensity = np.mean(patch)  # Calculate mean intensity
    std_intensity = np.std(patch)  # Calculate standard deviation
    return mean_intensity, std_intensity   
				
# Function to split an image into patches
def image_to_tiles(image_path, tile_size, patch_folder):
    
    # Extract the base name of the image
    patches_name = os.path.basename(image_path).split('.')[0] + "_patch_"
    
    # Load the image
    image = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = image.size
    
    # Calculate new dimensions to ensure full tile coverage
    new_width = width + (tile_size - (width % tile_size)) % tile_size
    new_height = height + (tile_size - (height % tile_size)) % tile_size
    
    # Resize the image and add padding
    padded_image = Image.new("RGB", (new_width, new_height))
    padded_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
    
    # Calculate the number of tiles in each direction
    num_tiles_x = new_width // tile_size
    num_tiles_y = new_height // tile_size
    
    # Split the image into patches
    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            left = x * tile_size
            upper = y * tile_size
            right = left + tile_size
            lower = upper + tile_size
            
            # Crop the tile
            tile = padded_image.crop((left, upper, right, lower))
            
            # Generate the tile filename
            tile_name = patches_name + str(x) + "_" + str(y) + ".jpg"
            
            # Define the patch path
            patch_path = os.path.join(patch_folder, tile_name)
            tile.save(patch_path)  # Save the tile

            # Read the patch as grayscale
            im = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
            mean_intensity, std_intensity = calculate_patch_statistics(im)

            # Compute variance using Laplacian
            patch_var = cv2.Laplacian(im, cv2.CV_64F).var()
            
            # Discard the patch if conditions are met
            if ((mean_intensity > 160 or patch_var > 650) & (std_intensity < 41)):
                discarded_folder = patch_folder.replace("patches", "discardedPatches")
                os.makedirs(discarded_folder, exist_ok=True)
                os.rename(patch_path, os.path.join(discarded_folder, os.path.basename(patch_path)))
            
# Define the dataset path
path = "new_data\\ColHis-IDS\\"
dir_list = glob.glob(path + "*")
data = []

# Iterate over each class directory
for class_path in dir_list:
    img_paths = glob.glob(class_path + "\\*\\*\\*.jpg", recursive=True)
    for img_path in img_paths:
        # Extract the image name
        img_name = os.path.basename(img_path)
        # Remove the file extension
        patch_path = os.path.splitext(img_path)[0]
        magnitude_dir = patch_path.split('\\')[-2]
        
        # Process only images with 200x magnification
        if magnitude_dir == '200':        
            # Generate patch folder name
            patch_folder = os.path.splitext(img_path)[0] + "_patches"
            
            # Create the patch folder if it does not exist
            os.makedirs(patch_folder, exist_ok=True)
            
            # Convert image to tiles
            image_to_tiles(img_path, 128, patch_folder)
