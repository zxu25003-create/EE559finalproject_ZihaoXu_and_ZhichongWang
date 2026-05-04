import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # [B, in_channels, 32, 32] -> [B, 6, 28, 28]
        x = self.conv1(x)
        x = self.relu(x)

        # [B, 6, 28, 28] -> [B, 6, 14, 14]
        x = self.pool1(x)

        # [B, 6, 14, 14] -> [B, 16, 10, 10]
        x = self.conv2(x)
        x = self.relu(x)

        # [B, 16, 10, 10] -> [B, 16, 5, 5]
        x = self.pool2(x)

        # [B, 16, 5, 5] -> [B, 400]
        x = x.view(x.size(0), -1)

        # [B, 400] -> [B, 120]
        x = self.fc1(x)
        x = self.relu(x)

        # [B, 120] -> [B, 84]
        x = self.fc2(x)
        x = self.relu(x)

        # [B, 84] -> [B, num_classes]
        x = self.fc3(x)

        return x
    