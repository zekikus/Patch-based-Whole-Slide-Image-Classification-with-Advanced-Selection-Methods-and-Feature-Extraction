import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# Function to compare predictions and ground truth (used with MLP)
def compare(img_labels, patch_predictions, selected_patch_num):
    
    img_prediction = {}

    # Get the most common prediction for each image
    for img_path, prediction_list in patch_predictions.items():
        img_prediction[img_path] = Counter(prediction_list).most_common(1)[0][0]
        
    true_values = 0
    false_values = 0
    conf_matrix = np.zeros((5, 5))  # Initialize confusion matrix
    
    # Compare predictions with ground truth
    for img, pred in img_prediction.items():
        class_id = img_labels[img]
        conf_matrix[class_id, pred] += 1
        if class_id == pred:
            true_values += 1
        else:
            false_values += 1

    accuracy = true_values / (true_values + false_values)
    print("Accuracy: ", accuracy)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                      xticklabels=['Normal', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma'],
                      yticklabels=['Normal', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'MLP\nTraining with {selected_patch_num} patches\nConfusion Matrix\nAccuracy: {accuracy:.2f}')
    plt.show()
    
    conf_matris_path = f"ConfMatrix\\mlp_{selected_patch_num}patches_not_discarding.png"
    fig = ax.get_figure()
    fig.savefig(conf_matris_path)  



# Function to predict class labels
def predict(flattened_img_path_list, ypred, model, outputh_path, selected_patch_num): # for random forest and xgboost
    
    # Find indexes of each image
    indexes = {img: [idx for idx, img_path in enumerate(flattened_img_path_list) if img_path == img] for img in set(flattened_img_path_list)}
    
    img_prediction = {}
    
    for img, idx_list in indexes.items():
        patch_predictions = [ypred[i] for i in idx_list]
        img_prediction[img] = Counter(patch_predictions).most_common(1)[0][0]
        
    # Class mapping dictionary
    class_map = {
        "Adenocarcinoma": 4,
        "High-grade IN": 3,
        "Low-grade IN": 2,
        "Polyp": 1,
        "Normal": 0
    }
    
    true_values = 0
    false_values = 0
    conf_matrix = np.zeros((5, 5))  # Initialize confusion matrix
    
    for img, pred in img_prediction.items():
        class_name = img.split('\\')[1]
        class_id = class_map[class_name]
        conf_matrix[class_id, pred] += 1
        if class_id == pred:
            true_values += 1
        else:
            false_values += 1
    
    accuracy = true_values / (true_values + false_values)
    print("Accuracy: ", accuracy)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
                      xticklabels=['Normal', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma'],
                      yticklabels=['Normal', 'Polyp', 'Low-grade IN', 'High-grade IN', 'Adenocarcinoma'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{model}\nTraining with {selected_patch_num} patches\nConfusion Matrix\nAccuracy: {accuracy:.2f}')
    plt.show()
    
    fig = ax.get_figure()
    fig.savefig(outputh_path)  # Save as PNG file