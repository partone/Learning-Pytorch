import torch
import torch.nn as nn
import torch.nn.functional as F     # Convention from pytorch doc

class Model(nn.Module):
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
        # Pick number and type of layers
        # Input -> Hidden 1 -> Hidden 2 -> Output
        # Input has 4 features, output has 3 classes
        # Picking 8 neurons for hidden layer 1, 9 for hidden layer 2

        # Override nn.Module's initializer
        super().__init__()
        
        # fc = fully connected
        # Layer 1
        self.fc1 = nn.Linear(in_features, h1)
        # Layer 2
        self.fc2 = nn.Linear(h1, h2)
        # Output
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        # relu = rectified linear unit (function for getting a result >= 0)
        a1 = F.relu(self.fc1(x))     # Pass x through the first layer
        a2 = F.relu(self.fc2(a1))     # Pass x through the second layer
        a3 = self.out(a2)             # Pass x through the output
        return a3 


model = Model()

# Working with the data
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/iris.csv')

def printGraphs(dfe):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
    fig.tight_layout()

    plots = [(0,1),(2,3),(0,2),(1,3)]
    colors = ['b', 'r', 'g']
    labels = ['Iris setosa','Iris virginica','Iris versicolor']

    for i, ax in enumerate(axes.flat):
        for j in range(3):
            x = dfe.columns[plots[i][0]]
            y = dfe.columns[plots[i][1]]
            ax.scatter(df[df['target']==j][x], df[df['target']==j][y], color=colors[j])
            ax.set(xlabel=x, ylabel=y)

    fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0,0.85))
    plt.show()

# Target is the y
X = df.drop('target', axis = 1)
y = df['target']

# Conversion to numpy arrays
X = X.values
y = y.values


# Modelling 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

criterion = nn.CrossEntropyLoss()

# model.parameters() is a method that generates an object from the neural network model
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

epochs = 200
losses = []     # Empty list

for i in range(epochs):
    # FP
    y_pred = model.forward(X_train)

    # Error
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    # Print loss every 10 epochs
    if i % 10 == 0:
        print(f"{i}\t\t{loss}")

    # BP
    optimizer.zero_grad()       # Find min of something (?)
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")


plt.show()
