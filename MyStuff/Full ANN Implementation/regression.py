import torch
import torch.nn as nn

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import haversineDistance as hav      # The distance between two coordinates

# 12k examples of taxi ride data
df = pd.read_csv('../../PYTORCH_NOTEBOOKS/Data/NYCTaxiFares.csv')

print(df.head())

print(df["fare_amount"].describe())

# Feature engineering

# Create a new feature for distance, *feature engineering*
df['dist_km'] = hav.haversine_distance(df, "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude")
print(df["dist_km"].describe())
# Convert the datetime column to a pandas datetime object (more versatile)
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

# Take in to account time zone
df["EDT_datetime"] = df["pickup_datetime"] - pd.Timedelta(hours = 4)

df["hour"] = df["EDT_datetime"].dt.hour

# where(condition, true, false)
df["am_pm"] = np.where(df["hour"] > 12, "pm", "am")

# Returns the abrreviated weekday name
df["weekday"] = df["EDT_datetime"].dt.strftime("%a")

# Categorical vs continuous columns
cat_cols = ["hour", "am_pm", "weekday"]
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']

y_col = ["fare_amount"]

# Change caterogorical columns to categories rather than numbers
for cat in cat_cols:
    df[cat] = df[cat].astype("category")

# pandas has now changed hour in to a 24 category data type
print(df["hour"].head())

# Each category is assigned a code
print(df["am_pm"].cat.codes)
print(df["am_pm"].cat.codes.values)     # This creates a numpy array of the codes

hr = df["hour"].cat.codes.values
ampm = df["am_pm"].cat.codes.values
wkdy = df["weekday"].cat.codes.values

cats = np.stack([hr, ampm, wkdy], axis = 1)  # This creates a 12,000 x 3 numpy array of the code values

# Could've been done in 1 line
# cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)

# Creating the a tensor
cats = torch.tensor(cats, dtype=torch.int64)

# Now for the continuous values
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float64)

# Converting y in to a tensor
y = torch.tensor(df[y_col].values, dtype = torch.float)

print(cats.shape)       # 12k x 3
print(conts.shape)      # 12k x 6
print(y.shape)          # 12k x 1

# Embedding
cat_szs = [len(df[col].cat.categories) for col in cat_cols] # Returns the sizes of categories of the contiuous data

# Get embedding sizes.  The size of the embedding and whether 50 or the size / 2 is smaller
emb_szs = [(size, min(50,(size + 1) // 2)) for size in cat_szs]
print(emb_szs)
# [(24, 12), (2, 1), (7, 4)]

# Get the first two cat examples
catz = cats[:4]

# [4, 0, 1],
# [11, 0, 2],
# [7, 0, 2],
# [17, 1, 3]

# Embedding takes the number of embeddings and embedding dimensions
# Creates 3 embedding layers
self_embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
print(self_embeds)
print(self_embeds.type)

embeddingz = []

# Forward method
# Creates a group of tensors based on the embedding dictionary
for i, e in enumerate(self_embeds):
    embeddingz.append(e(catz[:, i]))



# Joins all tensors in to one
z = torch.cat(embeddingz, 1)
print(z)

# The dropout layer randomly zeros some of the input to avoid overfitting (40% in this case)
selfembdrop = nn.Dropout(0.4)

z = selfembdrop(z)
print(z)

'''
emb_szs: list of tuples: each categorical variable size is paired with an embedding size
n_cont: int: number of continuous variables
out_sz: int: output size
layers: list of ints: layer sizes
p: float: dropout probability for each layer (for simplicity we'll use the same value throughout)
'''

class TabularModel(nn.Module):
    def __init__(self, emd_szs, n_cont, out_sz, layers, p = 0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])       # Create the embeddings list
        self.emb_drop = nn.Dropout(p)           # Create a dropout function
        self.bn_cont = nn.BatchNorm1d(n_cont)   # Normalise data

        layerlist = []  # Will contain a list of functions representing the layers of the neural network
        n_emb = sum((nf for ni, nf in emb_szs))

        # The number of nodes in the input data
        n_in = n_emb + n_cont 

        # For each layer
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))        # Add in a linear function
            layerlist.append(nn.ReLU(inplace=True))     # Add in a ReLU activation function
            layerlist.append(nn.BatchNorm1d(i))         # Add a normalisation setp
            layerlist.append(nn.Dropout(p))             # Add a dropout step
            n_in = i        # n_in becomes i because the number of inputs of each layer is the number of nodes of the previous one
        
        # Add an output layer
        layerlist.append(nn.Linear(layers[-1], out_sz))

        print(layerlist)

        # Combine the list of layers to create the model
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        x_cat = torch.tensor(x_cat).to(torch.int64)
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))       # Pass x_cat[:, i] (representing a column of data) through its corresponding embedding function

        x = torch.cat(embeddings, 1)    # Add embeddings to x
        x = self.emb_drop(x)            # Apply the dropout function

        # Add in the continuous data
        x_cont = self.bn_cont(x_cont)   # Normalise the continuous data
        x = torch.cat([x, x_cont], 1) 

        # Pass x through the layers
        x = self.layers(x)
        return x

# Embedding sizes, number of continuous data columns, output size, layer sizes, dropout probability
# Predicting a single value so output = 1 (since this is regression)
# Output would be equal to the number of output classes in a classification problem
model = TabularModel(emb_szs, conts.shape[1], 1, [200,100], 0.4)

print(model)

# Loss function
criterion = nn.MSELoss()        # We would use cross entropy loss for classification problems

optimizer = torch.optim.Adam(model.parameters(), 0.001)

batch_size = 60000
test_size = int(batch_size * 0.2)

# Data's already shuffled
cat_train = cats[:batch_size - test_size]
cat_test = cats[batch_size - test_size:batch_size]

cont_train = conts[:batch_size - test_size]
cont_test = conts[batch_size - test_size:batch_size]

y_train = y[:batch_size - test_size]
y_test = y[batch_size - test_size:batch_size]

print(len(cat_train))
print(len(cont_train))
print(len(cat_test))
print(len(cont_test))

# Time the training
import time
start_time = time.time()

epochs = 300
losses = []

cat_train = cat_train.type(torch.FloatTensor)
cont_train = cont_train.type(torch.FloatTensor)

for i in range(epochs):
    i += 1

    y_pred = model(cat_train, cont_train)
    loss = torch.sqrt(criterion(y_pred,y_train))
    losses.append(loss)

    if i % 10 == 1:
        print(f"{i}: {loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Wow, took a while.  Roughly {(time.time() - start_time) / 60}!")

plt.plot(range(epochs), losses)
plt.show()

cat_test = cat_test.type(torch.FloatTensor)
cont_test = cont_test.type(torch.FloatTensor)

# Predict on test set
with torch.no_grad():
    y_val = model(cat_test, cont_test)
    loss = criterion(y_val, y_test)

print(loss)     # Test set loss

for i in range(10):
    diff = y_val[i].item() - y_test[i].item()
    print(f"{i}: {y_val[i].item():8.2f}\t{y_test[i].item():8.2f}\t{diff:8.2f}")
    
# Save the model
torch.save(model.state_dict(), "taxiModel.pt")