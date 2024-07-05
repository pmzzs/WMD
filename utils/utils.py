import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, device, criterion=nn.BCEWithLogitsLoss()):
    """
    Evaluate the model on the given dataset.

    Parameters:
    - model: The trained PyTorch model to evaluate.
    - dataloader: DataLoader providing the dataset for evaluation.
    - device: The device (CPU or GPU) to perform evaluation on.

    Returns:
    - average_loss: The average loss over the dataset.
    - accuracy: The accuracy of the model on the dataset.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize metrics
    total_loss = 0.0
    total_correct = 0
    total_images = 0

    # No gradient needed
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device) # Adjust labels shape if necessary
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_images += inputs.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    
    print(f"Average loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    return average_loss, accuracy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def evaluate_model_binary(model, dataloader, device, use_fp16=False):
    """
    Evaluate the model on the given dataset using binary classification and calculate metrics using scikit-learn.

    Parameters:
    - model: The trained PyTorch model to evaluate.
    - dataloader: DataLoader providing the dataset for evaluation.
    - device: The device (CPU or GPU) to perform evaluation on.
    - use_fp16: Boolean to indicate whether to use FP16 for inference.

    Returns:
    - metrics: Dictionary containing average loss, accuracy, TPR (recall), precision, F1 score, and AUC.
    """
    model.eval()
    if use_fp16:
        model.half()  # Convert model to half precision
    criterion = nn.BCEWithLogitsLoss()

    predictions = []
    true_labels = []
    outputs_list = []
    labels_list = []
    total_loss = 0.0
    TP = 0  # True Positives
    FP = 0  # False Positives
    TN = 0  # True Negatives
    FN = 0  # False Negatives

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_fp16:
                inputs, labels = inputs.half(), labels.half()  # Convert inputs and labels to half precision
                
            outputs = model(inputs)
            outputs_list.extend(outputs.float().cpu().numpy().flatten())  # Convert outputs back to float for numpy compatibility
            labels_list.extend(labels.float().cpu().numpy().flatten())
            
            loss = criterion(outputs, labels.float())  # Ensure labels are float for loss calculation
            total_loss += loss.item() * inputs.size(0)
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            preds = torch.sigmoid(outputs).float().cpu().numpy()  # Ensure predictions are float
            preds = (preds > 0.5).astype(int)
            predictions.extend(preds.flatten())
            true_labels.extend(labels.float().cpu().numpy().flatten())
            
            # Update TP, FP, TN, FN
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    # Calculate metrics
    tpr = TP / (TP + FN) if (TP + FN) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0

    # Plotting histograms
    outputs_array = np.array(outputs_list)
    labels_array = np.array(labels_list)
    plt.figure(figsize=(10, 6))
    if outputs_array[labels_array == 1].shape[0] > 0:
        plt.hist(outputs_array[labels_array == 1], bins=min(100, outputs_array[labels_array == 1].shape[0]), color='green', alpha=0.5, label='Positive', density=True)
    if outputs_array[labels_array == 0].shape[0] > 0:
        plt.hist(outputs_array[labels_array == 0], bins=min(100, outputs_array[labels_array == 0].shape[0]), color='red', alpha=0.5, label='Negative', density=True)
    plt.title('Distribution of Model Outputs by True Label')
    plt.xlabel('Model output (Sigmoid)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Compute metrics using scikit-learn
    accuracy = accuracy_score(labels_array, predictions)
    precision = precision_score(labels_array, predictions)
    f1 = f1_score(labels_array, predictions)
    auc = roc_auc_score(labels_array, outputs_array)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels_array, outputs_array)

    # Find the closest threshold to 90% TPR (0.9)
    closest_tpr_index = (np.abs(tpr - 0.9)).argmin()
    fpr_at_90_tpr = fpr[closest_tpr_index]

    # Find the closest threshold to 10% FPR (0.1)
    closest_fpr_index = (np.abs(fpr - 0.1)).argmin()
    tpr_at_10_fpr = tpr[closest_fpr_index]

    average_loss = total_loss / len(dataloader.dataset)

    metrics = {
        'Average Loss': average_loss,
        'Accuracy': accuracy,
        'Precision': precision,
        'TPR': tpr,
        'FPR': fpr,
        'F1 Score': f1,
        'AUC': auc,
        'TPR10FPR': tpr_at_10_fpr,
        'FPR90TPR': fpr_at_90_tpr
    }
    
    print(f"Average loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, TPR10FPR: {tpr_at_10_fpr:.4f}, FPR90TPR: {fpr_at_90_tpr:.4f}, AUC: {auc:.4f}")

    return metrics

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
        
        
def get_ranking(percent, clean_idx, model, dataloader, device, use_fp16=False):
    model.eval()
    if use_fp16:
        model.half()

    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_fp16:
                inputs, labels = inputs.half(), labels.half()  # Convert inputs and labels to half precision
                
            outputs = model(inputs)
            outputs_list.extend(outputs.float().cpu().numpy().flatten())  # Convert outputs back to float for numpy compatibility
            labels_list.extend(labels.float().cpu().numpy().flatten())
    
    full_idx = np.array(list(range(len(dataloader.dataset))))
    remain_list = [item for item in full_idx if item not in clean_idx]
    outputs_list = list(np.array(outputs_list)[remain_list])
    temp_idx = np.argsort(outputs_list)[: int(len(outputs_list) * percent)]
    least_idx = full_idx[remain_list][temp_idx]
    least_labels = np.array(labels_list)[least_idx]
    num_wm = np.sum(least_labels)
    
    model.train()
    return outputs_list, least_idx, num_wm

def get_ranking_reeval(percent, clean_idx, model, dataloader, device, use_fp16=False):
    model.eval()
    if use_fp16:
        model.half()

    outputs_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_fp16:
                inputs, labels = inputs.half(), labels.half()  # Convert inputs and labels to half precision
                
            outputs = model(inputs)
            outputs_list.extend(outputs.float().cpu().numpy().flatten())  # Convert outputs back to float for numpy compatibility
            labels_list.extend(labels.float().cpu().numpy().flatten())
    
    least_idx = np.argsort(outputs_list)[: int(len(outputs_list) * percent)]
    least_labels = np.array(labels_list)[least_idx]
    num_wm = np.sum(least_labels)
    
    model.train()
    return outputs_list, least_idx, num_wm