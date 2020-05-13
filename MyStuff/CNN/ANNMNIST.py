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
plt.show()

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
print(images.shape)