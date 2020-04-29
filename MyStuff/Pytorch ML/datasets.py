import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/iris.csv')

print(df.head())
#print(df.shape)       #It's a 150 x 5

# Print the data

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

#printGraphs(df)

# Splitting data in to test and training sets
from sklearn.model_selection import train_test_split
features = df.drop('target', axis = 1).values
label = df['target']

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)   # 20% of the data goes to the test set

# Create the tensors

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# y can be a Long since y is supposed to be 0, 1, or 2
y_train = torch.LongTensor(y_train).reshape(-1, 1)      #Make it a column
y_test = torch.LongTensor(y_test.values).reshape(-1, 1)

''' Alternatively:
from torch.utils.data import TensortDataset, DataLoader
data = df.drop("target", axis = 1).values
labels = df["target"].values
iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
print(type(iris))   # TensorDataset
# It's an list of tensors basically

iris_loader = DataLoader(iris, batch_size = 50, shuffle = True)


'''