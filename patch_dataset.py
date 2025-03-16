from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image

# Custom dataset class
class My_dataloader(Dataset):

    
    def __init__(self, train=True, selected_patch_num=10):
        """
        Initialize dataset by reading file paths and processing patches.
        """
        if train:
            with open('ColHis-IDS\\train_paths.txt', 'r') as f: #if working with clean data change the path as open('new_data\\ColHis-IDS\\train_paths.txt', 'r')
                paths = [line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f: #if working with clean data change the path:open('new_data\\ColHis-IDS\\test_paths.txt', 'r')
                paths = [line.strip() for line in f]
        
        self.data = []  # List to store processed data
        for path in paths:
            if train:
                selected_directory_name = '_selected' + str(selected_patch_num) + 'patches'
                selected_patches_paths = glob(path.replace('.jpg', selected_directory_name) + '\\*.jpg')
            else:
                selected_patches_paths = glob(path.replace('.jpg', '_patches') + '\\*.jpg')
            
            for selected_patch in selected_patches_paths:
                img_class = selected_patch.split('\\')[1]  # Extract class name from path          
                embedding_path = selected_patch.replace('.jpg', '_embedding.npy')
                self.data.append([img_class, selected_patch, embedding_path])     
        
        # Class mapping dictionary
        self.class_map = {
            "Adenocarcinoma": 4,
            "High-grade IN": 3,
            "Low-grade IN": 2,
            "Polyp": 1,
            "Normal": 0
        }
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):  
        return len(self.data)
    
    def __getitem__(self, idx): 
        """
        Get item by index: returns image, path, embedding, and class ID.
        """
        class_name, img_path, embedding_path = self.data[idx]
        image_name = os.path.basename(img_path).split('_patch')[0]
        class_id = self.class_map[class_name]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        embedding = np.load(embedding_path)
        return img, img_path, image_name, embedding, class_id
