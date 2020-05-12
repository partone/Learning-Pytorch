import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import TabularModel as tb

df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/income.csv')

print(len(df))      # 30k examples
print(df.head())

print(df['label'].value_counts())   # 21,700 <= 50K examples, the rest are > 50k

# Distinguish categorical/continuous data

cat_cols = ["sex", "education-num", "marital-status", "workclass", "occupation"]
cont_cols = ["age", "hours-per-week"]
y_col = ["label"]

print(f'cat_cols  has {len(cat_cols)} columns')
print(f'cont_cols has {len(cont_cols)} columns')
print(f'y_col     has {len(y_col)} column')

# Change caterogorical columns to categories rather than numbers
for cat in cat_cols:
    df[cat] = df[cat].astype("category")

# Shuffle the data set
df = shuffle(df, random_state=101)
df.reset_index(drop=True, inplace=True)
df.head()

# Get the category sizes
cat_szs = []
for col in cat_cols:
    cat_szs.append(len(df[col].cat.categories))

emb_szs = [(size, min(50,(size + 1) // 2)) for size in cat_szs]

print(emb_szs)

# Create an array of categorical values
cats = np.transpose(np.array([df[col].cat.codes.values for col in cat_cols]))
print(cats[:5])

cats = torch.tensor(cats, dtype=torch.int64)

# Do the same for continuous
conts = np.transpose(np.array([df[col] for col in cont_cols]))
print(conts[:5])
conts = torch.tensor(conts, dtype=torch.float32)

# And for the result
#y = torch.tensor(np.array(df[y_col]).reshape(1, -1))       # This produces a 1xn tensor rather than a truly flat one
y = torch.tensor(np.array(df[y_col])).flatten()
print(y)

b = 30000 # suggested batch size
t = 5000  # suggested test size

# Create batches
cat_train = cats[:b - t]
cat_test = cats[b - t:b]

cont_train = conts[:b - t]
cont_test = conts[b - t:b]

y_train = y[:b - t]
y_test = y[b - t:b]

# Create NN model
#model = tb.TabularModel(emb_szs, len(conts), 2, [50], 0.4)     # Wrong axis D:
model = tb.TabularModel(emb_szs, conts.shape[1], 2, [50], 0.4)

# Loss calculation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.001)

# Train model
import time
start_time = time.time()

epochs = 300
losses = []

print(cat_train.shape)
print(cont_train.shape)
print(y_train.shape)
print(y_test.shape)

for i in range(epochs):
    i += 1

    y_pred = model(cat_train, cont_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

# Plot the loss
plt.plot(range(epochs), losses)
plt.show()


# Evaluate over the test set
with torch.no_grad():
    y_val = model(cat_test, cont_test)
    loss = criterion(y_val, y_test)

print(f'CE Loss: {loss:.8f}')

correct = 0 

for i in range(len(y_test)):
    if np.argmax(y_val[i]) == y_test[i]:
        correct += 1

print(f"{100 * correct / len(y_test)}% correct")

