import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

arr = np.random.randint(0, 5, 6)
print(arr)

x = torch.tensor(arr)
print(x)

x = x.type(torch.int64)
print(x.type())

x = x.reshape(3, 2)
print(x)

print(x[:, 1:])

print(pow(x, 2))

y = torch.randint(0, 5, (2, 3))
print(y)

print(x.type())
print(y.type())

print(x.mm(y))