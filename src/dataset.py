import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO, PneumoniaMNIST

def get_dataset_info():
    """
    Get the dataset information for PneumoniaMNIST.

    """
    return INFO['pneumoniamnist']

def get_transforms():
    # Define the preprocessing pipeline for training data
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Define the preprocessing pipeline for validation/testing data
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def get_dataset_info():
    return {
        'description': 'PneumoniaMNIST dataset for binary classification of chest X-rays.',
        'label': {
            '0': 'normal',
            '1': 'pneumonia'
        }
    }
