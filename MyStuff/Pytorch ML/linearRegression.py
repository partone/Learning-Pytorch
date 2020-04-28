
import torch
import numpy as np
import matplotlib.pyplot as plt     # Plotting library

import torch.nn as nn       # For faster syntax

X = torch.linspace(1, 50, 50).reshape(-1, 1)           # 50 linearly spaced elements column vector

e = torch.randint(-8, 9, (50, 1), dtype=torch.float)   # 50 random numbers from -8 to 8, column vector

y = 2 * X + 1 + e   # e is just there to create some noise

# To plot with matplotlib, the tensor has to be a numpy array
plt.scatter(X.numpy(), y.numpy())

#plt.show()

model = nn.Linear(in_features=1, out_features=1)     # Linear layer

# These start off random
print(model.weight)
print(model.bias)

# Model class inherited from nn.Module
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model(1, 1)

# Print model parameters 
def printModelParameters(m):
    for name, param in m.named_parameters():
        print(name, '\t', param.item())

printModelParameters(model)

x = torch.tensor([2.0])

print(model.forward(x))     # Just plugging in x to the model, y = xw + b

#Fitting to data

x1 = np.linspace(0., 50., 50)
w1 = 0.1059
b1 = 0.9637

y1 = w1 * x1 * b1
print(y1)       # Predicted y based on random x values

# Very poor fit since weight and bias are at their initial values
plt.plot(x1, y1, 'r')
#plt.show()

# Defining the loss function - Mean squared error
# Criterion is the standard practice variable name
criterion = nn.MSELoss()

# Set optimisation
# Stochastic gradient descent w/ user defined learning rate
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

# Decide number of epochs (iterations of data set)
epochs = 50
losses = []     # To keep track of error

plt.clf()       # Clear plot

for i in range(epochs):
    i = i + 1
    y_pred = model.forward(X)       # Predict
    loss = criterion(y_pred, y)     # Get error
    losses.append(loss)             # Append error to list

    print(f"{i} \tLoss: {loss.item():.5f} \tWeight: {model.linear.weight.item():.5f}\tBias: {model.linear.bias.item():.5f}")     # print(f) means formatted printing

    optimizer.zero_grad()       # Reset gradients
    loss.backward()             # Backprop
    optimizer.step()            # Updates parameters

plt.plot(range(epochs), losses)
plt.ylabel("Mean squared error")
plt.xlabel("Iteration")

#plt.show()

# Use the model
x = np.linspace(0., 50., 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
predicted_y = current_weight * x + current_bias

plt.clf()
plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, 'r')
plt.show()