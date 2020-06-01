import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms    # For computer vision
from torchvision.utils import make_grid

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

# Get train/test data as tensor
transform = transforms.ToTensor()
train_data = datasets.MNIST("../../../PYTORCH_NOTEBOOKS/Data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="../../../PYTORCH_NOTEBOOKS/Data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)   # Doesn't need to be shuffled since it's not training

# Convolutional layers (example, implemented as a class later on)
# 1 input channel since it's grayscale
# 6 filters that the CNN layer will figure out (feature maps)
# Kernel size is the n x n sized window for the image kernel (filter)
conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)

# 6 input channels since it takes in conv1
# 16 output channels (filters)
conv2 = nn.Conv2d(6, 16, 3, 1)
# Input and output numbers are somewhat arbitrary

for i, (X_train, y_train) in enumerate(train_data):
    break


# A single image
print(X_train.shape)

# Transform to a 4D batch of 1 image
x = X_train.view(1, 1, 28, 28)
print(x.shape)  # 1x1x28x28

x = F.relu(conv1(x))
print(x.shape)  # 1x6x26x26

# 28x28 => 26x26
# Lost some border information since padding = 0 in our layer
# Not a big deal this time, might be important for other data

# Max pooling layer
# Kernel size = 2x2, stride = 2
x = F.max_pool2d(x, 2, 2)

print(x.shape)  # 1x16x13x13, pooled down

x = F.relu(conv2(x))

print(x.shape)  # 1x16x11x11

x = F.max_pool2d(x, 2, 2)

print(x.shape)  # 1x16x5x5

x = x.view(-1, 16*5*5)
print(x.shape)  # 1x400

# CNN Classa
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        # Fully connected layers
        # Input is 5*5*16 since that's the flattened size of the output of the last convolutional layer
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)   # Flatten for the fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

# Let's test it out
model = ConvolutionalNetwork()
print(model)

for param in model.parameters():
    print(param.numel())
# Total is 60,074, about half of the parameters against a ANN

# The usual
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Time this bad boy
import time
start_time = time.time()

epochs = 5

# Trackers
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_correct_count = 0
    test_correct_count = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1      # Batches start at 1 to look nice
        
        y_pred = model(X_train)     # Not flattened since the CNN receives 2D data
        
        # See how its doing
        loss = criterion(y_pred, y_train)
        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted == y_train).sum()

        train_correct_count += batch_correct

        # BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f"{i}\t{b}\t{loss.item()}")

    train_correct.append(loss)
    train_correct.append(train_correct_count)

    # Test
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            test_correct_count += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_correct_count)

current_time = time.time()
total_time = current_time - start_time
print(f"Finished in {total_time / 60} minutes")