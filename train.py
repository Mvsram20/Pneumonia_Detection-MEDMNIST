import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST

# The Folder Bridge: Allows train.py to look inside the 'src' folder
sys.path.append(os.path.abspath('src'))
from dataset import get_transforms
from models import get_model

def train_engine():
    
    #  CORE HYPERPARAMETERS & CONFIGURATIONS
   
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 5
    MODEL_NAME = "vanilla"  # Options: "vanilla", "resnet", "mobilenet", "inception"

    # Target your M4 MacBook's GPU Accelerator
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Active Compute Hardware: {device.type.upper()}")

   
    #  DATA STREAM INGESTION (LOADERS)
    
    train_transform, val_transform = get_transforms()
    
    # Downloads (if missing) and wraps the clinical data splits
    train_dataset = PneumoniaMNIST(split='train', transform=train_transform, download=True)
    val_dataset = PneumoniaMNIST(split='val', transform=val_transform, download=True)
    
    # Chops the massive datasets into streamable batches of 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" Pipelines Active | Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    
    # INITIALIZE BRAIN & OPTIMIZATION MATH
    
    model = get_model(MODEL_NAME).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Initiating model training cycle for: {MODEL_NAME.upper()}\n")
    print("--- Starting Training Loop ---")
    
    best_accuracy = 0.0
    
    
    #  THE LIVE TRAINING & VALIDATION ENGINE
    
    for epoch in range(EPOCHS):
        # --- PHASE A: TRAINING ON PATIENT RECORDS ---
        model.train()  # Activates gradient tracking
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze().long() # Standardizes shape for calculation
            
            # The 5-Step Core Optimization Sequence
            optimizer.zero_grad()               # 1. Reset memory slopes
            outputs = model(inputs)             # 2. Make a diagnostic guess
            loss = criterion(outputs, labels)   # 3. Grade the mistakes
            loss.backward()                     # 4. Trace the calculus backwards
            optimizer.step()                    # 5. Nudge weights down toward accuracy
            
            # Metrics Tracking
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = (correct_train / total_train) * 100
        
        # --- PHASE B: VALIDATION (EXAMINING UNSEEN PATIENTS) ---
        model.eval()  # Disables dropout/batchnorm updates for stability
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():  # Freezes gradient calculus memory to run lightning fast
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze().long()
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_acc = (correct_val / total_val) * 100
        
        # Output progress dashboard directly to terminal
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
       
        #  CHECKPOINT WEIGHT CODES INFRASTRUCTURE
       
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Serializes and exports neural network parameters to an SSD file
            torch.save(model.state_dict(), "best_model.pth")
            print("[SAVED] New peak accuracy checkpoint stored locally.")

    print(f"\n PROCESS COMPLETE! Maximum Validation Accuracy Secure: {best_accuracy:.2f}%")

if __name__ == "__main__":
    train_engine()