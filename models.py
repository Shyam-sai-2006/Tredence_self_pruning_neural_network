import torch
import torch.nn as nn
from layers import PrunableLinear

class CIFAR10PrunableNet(nn.Module):
    def __init__(self):
        super(CIFAR10PrunableNet, self).__init__()
        
        # Standard CNN backbone for high accuracy on CIFAR-10
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # The Prunable Head (where we apply the case study logic)
        # Input to linear is 128 * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            PrunableLinear(2048, 512),
            nn.ReLU(),
            PrunableLinear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x
