import torch
import torch.nn as nn
import torchvision.models as models

# 1. Your Custom Baseline Neural Network
class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super(PneumoniaClassifier, self).__init__()
        # Defining standard convolutional layer blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # Binary outcome: Normal vs Pneumonia

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 2. Advanced Pre-built Blueprint Architectures (FLUSHED TO THE LEFT MARGIN)
def get_resnet18():
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model

def get_mobilenet_v2():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model

def get_inception():
    model = models.inception_v3(weights=None, aux_logits=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model


# 3. Master Factory Switchboard (FLUSHED COMPLETELY TO THE LEFT MARGIN)
def get_model(model_name):
    model_name = model_name.lower()
    
    # Matches the exact string tags requested by test_pipeline.py
    if model_name == "vanilla":
        return PneumoniaClassifier()
    elif model_name == "resnet":
        return get_resnet18()
    elif model_name == "mobilenet":
        return get_mobilenet_v2()
    elif model_name == "inception":
        return get_inception()
    else:
        raise ValueError(f"Model {model_name} is not supported.")