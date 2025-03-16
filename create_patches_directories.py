import glob
import os
import pickle
import shutil

# Global counter to track the number of processed WSI selections
count = 0

def find_path_from_indexes(base_path, patch_name, index):
    """
    Constructs the file path for a patch based on its base path, name, and index.
    """
    patch_name = patch_name.replace('es', '')  # Remove 'es' from patch name (if applicable)
    index_string = "_".join(map(str, index))  # Convert index tuple to a string
    extension = ".jpg"  # File extension
    full_path = f"{base_path}\\{patch_name}_{index_string}{extension}"  # Construct full path
    return full_path

def save_max_cosin_similar_patches(dict_path, max_num=20):
    """
    Selects and saves patches with the highest cosine similarity.
    """
    global count
    
    # Load the similarity dictionary from the pickle file
    with open(dict_path, "rb") as file:
        max_similars_dict = pickle.load(file)
    
    # Get the indexes of the top max_num most similar patches
    indexes_of_maximums = list(max_similars_dict)[:max_num]
    indexes_of_maximums = list(map(int, indexes_of_maximums))
    
    # Extract base path and patch name from dict_path
    parts_of_path = dict_path.split("\\") 
    base_path = parts_of_path[:5]  # First five parts of the path
    patch_name = parts_of_path[-2]  # Extract patch name from path
    base_path = "\\".join(base_path)  # Reconstruct base path
    
    # Define the destination folder name
    selected_folder_name = 'selected' + str(max_num) + 'patches'
    
    for index in indexes_of_maximums:
        row_index = index // 12  # Compute row index
        col_index = index % 12  # Compute column index
        
        # Get source patch path
        source_patch_path = find_path_from_indexes(base_path, patch_name, (row_index, col_index))
        
        # Define destination paths
        destination_patch_path = source_patch_path.replace("patches", selected_folder_name)
        source_embedding_path = source_patch_path.replace('.jpg', '_embedding.npy')
        dest_embedding_path = destination_patch_path.replace('.jpg', '_embedding.npy')
        
        # Create the destination directory if it doesn't exist
        destination_folder = os.path.dirname(destination_patch_path)
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        
        # Copy the patch image and its corresponding embedding file
        shutil.copyfile(source_patch_path, destination_patch_path)
        shutil.copyfile(source_embedding_path, dest_embedding_path)
    
    count += 1  # Increment the counter
    print(count, ". wsi selection")  # Print the selection count

# Define the root path for data processing
path = "ColHis-IDS\\"

# Get the list of all directories in the root path
dir_list = glob.glob(path + "*")

similarity_list = []  # List to store similarity dictionary paths

# Iterate over each class directory
for class_path in dir_list:
    # Find all .pickle files within the directory hierarchy
    patch_similarity_dicts = glob.glob(class_path + "\\*\\*\\*\\*.pickle", recursive=False)
    
    # If any .pickle files are found, add them to similarity_list
    if patch_similarity_dicts is not None:
        for patch_similarity in patch_similarity_dicts:
            similarity_list.append(patch_similarity)

# Process each similarity dictionary
for similarity_path in similarity_list:
    save_max_cosin_similar_patches(similarity_path, 50)
