import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import xgboost as xgb
import time
import tqdm
from torch.utils.data import DataLoader
from patch_dataset import My_dataloader
from compare import predict

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(42)
selected_patch_num = 50  # Number of selected patches

if __name__ == '__main__':
    start_time = time.time()  # Record start time
    print("Program started")

    # Load training dataset
    train_dataset = My_dataloader(train=True, selected_patch_num=selected_patch_num)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print("train_dataloader is created")

    train_embeds_flat = []
    train_labels_flat = []
    train_img_paths = []

    # Extract embeddings and labels from training dataset
    for imgs, img_paths, image_names, embeddings, class_ids in train_dataloader:
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        train_embeds_flat.append(embeddings_flat.numpy())
        train_img_paths.append(img_paths)
        labels_flat = class_ids.numpy().flatten()
        train_labels_flat.append(labels_flat)
    
    # Convert lists to numpy arrays
    train_embeds_flat = np.concatenate(train_embeds_flat, axis=0)
    train_labels_flat = np.concatenate(train_labels_flat, axis=0)

    end_time = time.time()  # Record end time
    print("Execution time of creating train embeds and labels:", end_time - start_time, "seconds")
    
    print("Train embeddings shape:", train_embeds_flat.shape)
    print("Train labels shape:", train_labels_flat.shape)
    
    # Flatten image paths
    train_paths = [list(i) for i in train_img_paths]
    flattened_img_path_list = sum(train_paths, [])
    flattened_img_path_list = ['\\'.join(path.split('\\')[:-1]) for path in flattened_img_path_list]

    start_time = time.time()  # Record start time

    # Load test dataset
    test_dataset = My_dataloader(train=False, selected_patch_num=selected_patch_num)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    test_embeds_flat = []
    test_labels_flat = []
    test_img_paths = []

    # Extract embeddings and labels from test dataset
    for imgs, img_paths, image_names, embeddings, class_ids in tqdm.tqdm(test_dataloader):
        embeddings_flat = embeddings.view(embeddings.size(0), -1)
        test_embeds_flat.append(embeddings_flat.numpy())
        test_img_paths.append(img_paths)
        labels_flat = class_ids.numpy().flatten()
        test_labels_flat.append(labels_flat)
    
    # Convert lists to numpy arrays
    test_embeds_flat = np.concatenate(test_embeds_flat, axis=0)
    test_labels_flat = np.concatenate(test_labels_flat, axis=0)

    end_time = time.time()  # Record end time
    print("Execution time of creating test embeds and labels:", end_time - start_time, "seconds")
    
    print("Test embeddings shape:", test_embeds_flat.shape)
    print("Test labels shape:", test_labels_flat.shape)
    
    # Flatten image paths
    test_paths = [list(i) for i in test_img_paths]
    flattened_img_path_list = sum(test_paths, [])
    flattened_img_path_list = ['\\'.join(path.split('\\')[:-1]) for path in flattened_img_path_list]

    start_time = time.time()  # Record start time

    # Create and train XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(train_embeds_flat, train_labels_flat)
    
    # Predict using the trained XGBoost model
    ypred = xgb_model.predict(test_embeds_flat)
    
    end_time = time.time()  # Record end time
    print("Execution time of prediction:", end_time - start_time, "seconds")
    
    
    # Compare predictions with ground truth
    conf_matris_path= f"ConfMatrix\\xgboost_{selected_patch_num}patches_not_discarding.png"
    predict(flattened_img_path_list=flattened_img_path_list, ypred=ypred, model="XGBoost", outputh_path = conf_matris_path, selected_patch_num=selected_patch_num)
