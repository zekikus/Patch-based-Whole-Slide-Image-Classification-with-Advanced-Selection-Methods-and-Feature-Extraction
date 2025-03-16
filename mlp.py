import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
from torchvision import transforms
import os
from PIL import Image
from glob import glob
import random
from compare import compare

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(42)

selected_patch_num = 50  # Number of selected patches

# Custom dataset class
class My_dataloader(Dataset):
    global selected_patch_num
    
    def __init__(self, train=True, mod=0):
        """
        Initialize dataset by reading file paths and processing patches.
        """
        if train:
            with open('ColHis-IDS\\train_paths.txt', 'r') as f: # if working with clean data change the path as open('new_data\\ColHis-IDS\\train_paths.txt', 'r')
                paths = [line.strip() for line in f]
        else:
            with open('ColHis-IDS\\test_paths.txt', 'r') as f: # if working with clean data change the path as open('new_data\\ColHis-IDS\\test_paths.txt', 'r')
                paths = [line.strip() for line in f]
        
        self.data = []  # List to store processed data
        for path in paths:
            if train:
                selected_directory_name = '_selected' + str(selected_patch_num) + 'patches'
                if mod == 0:              
                    selected_patches_paths = glob(path.replace('.jpg', selected_directory_name) + '\\*.jpg')[:selected_patch_num - 5] # For validation
                elif mod == 1:
                    selected_patches_paths = glob(path.replace('.jpg', selected_directory_name) + '\\*.jpg')[selected_patch_num - 5:]
            else:            
                selected_patches_paths = glob(path.replace('.jpg', '_patches') + '\\*.jpg')

            for selected_patch in selected_patches_paths:
                img_class = selected_patch.split('\\')[-5]  # Extract class name from path          
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


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(1024, 512)  # Input to hidden layer
        self.fc2 = nn.Linear(512, 1024)  # Hidden layer 1
        self.fc3 = nn.Linear(1024, 512)  # Hidden layer 2
        self.fc4 = nn.Linear(512, 5)     # Output layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x = self.softmax(x)
        return x
    
if __name__ == '__main__':
    print("Program started")
    
    # Load datasets and dataloaders
    train_dataset = My_dataloader(True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    test_dataset = My_dataloader(False)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    val_dataset = My_dataloader(True, mod=1)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = MLP().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    epochs = 200
    best_loss = float('inf')
    validation_interval = 20  # Validate every 20 epochs

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        # Iterate over the training dataset
        for x, y, z, inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move tensors to device

            optimizer.zero_grad()  # Reset gradients
            
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()

        # Perform validation every 'validation_interval' epochs
        if (epoch + 1) % validation_interval == 0:
            model.eval()  # Set the model to evaluation mode
            validation_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation
                for x, y, z, inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

            validation_loss /= len(val_dataloader)
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}, Validation Loss: {validation_loss:.4f}")

            # Save the best model based on validation loss
            if validation_loss < best_loss:
                best_loss = validation_loss
                model_file_name = f"best_model_trained_with_{selected_patch_num}patch.pth"
                torch.save(model.state_dict(), model_file_name)
                print("Best model saved with validation loss: {:.4f}".format(validation_loss))
        else:
            print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_dataloader):.4f}")

    # Load the best model for evaluation
    best_model = MLP().to(device)
    best_model.load_state_dict(torch.load(model_file_name))
    best_model.to(device)
    best_model.eval()  # Set model to evaluation mode

    # Initialize dictionaries to store predictions and labels
    all_predictions = {}
    all_labels = {}

    with torch.no_grad():
        for imgs, img_paths, image_names, embeddings, class_ids in test_dataloader:
            embeddings, class_ids = embeddings.to(device), class_ids.to(device)
            outputs = best_model(embeddings)  # Get predictions from model
            
            # Select the class with the highest probability
            _, predicted = torch.max(outputs.data, 1)

            for idx in range(len(predicted)):
                key = os.path.dirname(img_paths[idx])

                if key not in all_predictions.keys():
                    all_predictions[key] = []

                all_predictions[key].append(predicted[idx])

                if key not in all_labels.keys():
                    all_labels[key] = class_ids[idx] 
            
    # Compare predictions with ground truth
    compare(all_labels, all_predictions, selected_patch_num)