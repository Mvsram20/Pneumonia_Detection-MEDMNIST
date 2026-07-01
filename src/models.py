import torch 
import torch.nn as nn
import torchvision.models as models

class PneumoniaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PneumoniaClassifier, self).__init__()
        # Convolutional Block 1: Extracts low-level edges
        # Input shape: [Batch, 3, 128, 128]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        # Output shape after pool1: [Batch, 16, 64, 64]

        # Convolutional Block 2: Extracts mid-level shapes
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output shape after pool2: [Batch, 32, 32, 32]

        # Flatten Layer: Converts 3D feature maps into a 1D vector
        self.flatten = nn.Flatten()
        # 32 channels * 32 width * 32 height = 32,768 input features

        # Fully Connected Decision Block
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 2) # 2 Outputs: [Normal Score, Pneumonia Score]


    def forward(self, x):
        # Defines how data flows sequentially through the layers
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
        

    def get_resnet18():
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)  # Adjusting the final layer
        return model
    
    def get_mobilenet_v2():
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)  # Adjusting the final layer
        return model
    
    def get_inception():
        model = models.inception_v3(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)  # Adjusting the final layer
        return model
    
    def get_model(model_name):
        if model_name == 'resnet18':
            return PneumoniaClassifier.get_resnet18()
        elif model_name == 'mobilenet_v2':
            return PneumoniaClassifier.get_mobilenet_v2()
        elif model_name == 'inception':
            return PneumoniaClassifier.get_inception()
        else:
            raise ValueError(f"Model {model_name} is not supported.")