import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST

# The Folder Bridge
sys.path.append(os.path.abspath('src'))
from dataset import get_transforms

# 1. Fetch our custom preprocessing pipelines
train_transform, _ = get_transforms()

# 2. Trigger the automated download and stream the training files
print("Checking for dataset records on your SSD...")
train_dataset = PneumoniaMNIST(split='train', transform=train_transform, download=True)
print(f"SUCCESS: Loaded {len(train_dataset)} chest X-ray samples.")

# Extract all raw images and collapse diagnoses into a simple 1D array
images = train_dataset.imgs
labels = train_dataset.labels.flatten()

# Find the exact index locations of normal vs pneumonia cases
normal_indices = np.where(labels == 0)[0]
pneumonia_indices = np.where(labels == 1)[0]

# Pick the first 4 patient index cards from each group
selected_normal = normal_indices[:4]
selected_pneumonia = pneumonia_indices[:4]

# Build an empty canvas layout with 2 horizontal rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# Row 1: Draw the 4 Healthy Lungs
for i, idx in enumerate(selected_normal):
    axes[0, i].imshow(images[idx], cmap='gray')
    axes[0, i].set_title(f"Normal #{i+1}")
    axes[0, i].axis('off') # Hides graph grid lines

# Row 2: Draw the 4 Infected Lungs
for i, idx in enumerate(selected_pneumonia):
    axes[1, i].imshow(images[idx], cmap='gray')
    axes[1, i].set_title(f"Pneumonia #{i+1}")
    axes[1, i].axis('off')

# Save the final graphic to your workspace folder tree
plt.tight_layout()
plt.savefig("pneumonia_samples.png", dpi=300)
print("SUCCESS: Cached your visual matrix to 'pneumonia_samples.png'!")
plt.show()