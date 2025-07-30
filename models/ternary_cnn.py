import torch
import torch.nn as nn
import torch.nn.functional as F
from .ternary_layers import TernaryConv2d, TernaryLinear


class TernaryCNN(nn.Module):
    """
    TernaryCNN is a convolutional neural network that uses ternary quantized 
    weights in its convolutional and fully connected layers. The model is designed 
    for efficient inference and reduced memory usage while maintaining competitive 
    performance on classification tasks like MNIST.

    Architecture:
    - 4 convolutional layers with TernaryConv2d (ternary weights)
    - BatchNorm after each convolutional layer
    - MaxPooling after every 2 convolutions to reduce spatial dimensions
    - 3 fully connected layers with TernaryLinear
    - BatchNorm applied to the first two fully connected layers
    - Final output layer produces logits for `num_classes` classes

    Args:
        num_classes (int): Number of output classes (default: 10 for MNIST).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = TernaryConv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        
        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        
        self.conv3 = TernaryConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = TernaryConv2d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        
        self.pool  = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1   = TernaryLinear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)

        self.fc2   = TernaryLinear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.fc3   = TernaryLinear(128, num_classes)

    def forward(self, x):
        """
        Defines the forward pass through the TernaryCNN network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.shape[0], -1)

        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = F.relu(self.bn_fc2(self.fc2(x)))
        return self.fc3(x)
