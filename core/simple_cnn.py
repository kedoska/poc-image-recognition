import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):  # Assuming 10 different models of coffee machines
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64 * 37 * 37, 512)  # After pooling, the image size becomes (37, 37)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output size: (16, 149, 149)
        x = self.pool(F.relu(self.conv2(x)))  # Output size: (32, 74, 74)
        x = self.pool(F.relu(self.conv3(x)))  # Output size: (64, 37, 37)
        
        x = x.view(-1, 64 * 37 * 37)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
