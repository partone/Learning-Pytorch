import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms    # For computer vision

import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt

# Get the data and creates a tensor from it
transform = transforms.ToTensor()
train_data = datasets.MNIST("../../../PYTORCH_NOTEBOOKS/Data", train=True, download=True, transform=transform)

# train=False means we don't want the training set
test_data = datasets.MNIST(root="../../../PYTORCH_NOTEBOOKS/Data", train=False, download=True, transform=transform)

print(train_data)   # 60,000 examples
print(test_data)    # 10,000 examples

print(type(train_data[0]))
#print(train_data[0])    # 28x28 tensor representing the image, and the label

image, label = train_data[0]
print(image.shape)
print(image)    # The 1 means it's a single colour channel (0-1)
print(label)

# gist_yarg makes the plot grayscale
plt.imshow(image.reshape(28, 28), cmap="gist_yarg")       # Get rid of the 1 dimension
# plt.show()

# Load in batches to stop my laptop blowing up
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Batch size is larger since testing is so hard
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

from torchvision.utils import make_grid
# Some formatting
np.set_printoptions(formatter=dict(int=lambda x: f'(x:4)'))


# First batch
for images, labels in train_loader:
    break
print(images.shape) # 100x1x28x28

class MultiLayerPerceptron(nn.Module):
    # Input size is 28x28, output is numbers 0-9
    #Input, 120 hidden layer, 84 hidden layer, output
    def __init__(self, in_sz=784, out_sz=10, layers=[120,84]):
        super().__init__()
        # Define layers and their dimensions
        self.fc1 = nn.Linear(in_sz, layers[0])      # Linear mapping from layer 0 -> 1
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    #FP
    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)     # Not passed through RELU since we want to evaluate the multiclass output
    
        return F.log_softmax(X, dim=1)

model = MultiLayerPerceptron()
print(model)

# Loads of parameters
for param in model.parameters():
    print(param.numel())    # Number of elements

# 105,214 total parameters

# A convolutional neural network has fewer parameters and is better for this case

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

images.view(100, -1)    # Flattens view to create a 100x784 tensor instead of 100x1x28x28

# Time training
import time
start_time = time.time()

# Train
epochs = 10

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_correct_counter = 0
    test_correct_counter = 0

    # Enumerate just adds a number to each train_loader element
    # Ex. {1 image_batch_1}, {2, image_batch_2}, etc.
    # so b is the batch number, X_train are the x parameters of the image, y_train is the y classification of the image
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1  # Call batch 0, batch 1 instead
        y_pred = model(X_train.view(100, -1))   # Flatten to a 100x784 like before
        
        loss = criterion(y_pred, y_train)

        # The predicted is the class with the highest probability
        predicted = torch.max(y_pred.data, 1)[1]

        # Count good predictions
        # predicted == y_train will return a vector of 0s and 1s
        batch_correct = (predicted == y_train).sum()
        train_correct_counter += batch_correct

        #BP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 200 == 0:
            # .item() returns the first element of loss as a scalar
            print(f"{i}\t{b}\t{loss.item()}\t{train_correct_counter.item() / b}%")

    train_losses.append(loss)
    train_correct.append(train_correct)

    # Check how the test data checks are improving
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500, -1))     # 500 since test batches are that size

            predicted = torch.max(y_val.data, 1)[1]
            test_correct_counter += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_correct_counter)

total_time = time.time() - start_time
print(f"Duration: {total_time / 60}")

plt.clf()

plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")

plt.legend()
plt.show()
"""
train_accuracy = [t / 600 for t in train_correct]
print(train_accuracy)

test_accuracy = [t / 100 for t in test_correct]
print(test_accuracy)

plt.clf()

plt.plot(train_accuracy, label="Train accuracy")
plt.plot(test_accuracy, label="Test accuracy")

plt.legend()
plt.show()
"""
# Test out new data

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    
    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

print(f"{correct.item() * 100 / len(test_data)}%")