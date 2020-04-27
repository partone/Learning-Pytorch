# Tensors
# A tensor is a matrix of N dimensions of a single data type

import torch
import numpy as np

print(torch.__version__)

arr = np.array([1, 2, 3, 4, 5])

# Creates a tensor from a numpy array
tensorArr = torch.from_numpy(arr)

# or a more flexible option
torch.as_tensor(arr)

# Array from 0-11, 4x3
arr2d = np.arange(0.0, 12.0)
arr2d = arr2d.reshape(4, 3)

tensorArr2d = torch.from_numpy(arr2d)

print(tensorArr2d)

# Ahhhhh
arr[0] = 99
print(tensorArr)

# This makes a copy which is converted to a tensor
tensorArr = torch.tensor(arr)
arr[0] = 3
print(tensorArr)    # Much better


# Class constructors
# Creating tensors from scratch

# Capital t in Tensor makes it a float type
# Same as torch.FloatTensor(arr)
tensorArrFloat = torch.Tensor(arr)

emptyTensor = torch.empty(2, 2)

# Data type is optional
zerosTensor = torch.zeros(4, 4, dtype=torch.int64)
torch.ones(4, 4)

filledTensor = torch.arange(0, 18, 2).reshape(3, 3)
torch.linspace(0, 18, 12).reshape(3, 4)

# Changing data type
myTensor = torch.tensor([1, 2, 3])
myTensor.type(torch.float32)

torch.rand(4, 3)
torch.randn(4, 3)   # Std = 1, mu = 0
torch.randint(3, 5, (5, 5))  # Integers, low 3, high 5 (exclusive), 5x5

#torch.randn_like(zerosTensor)   # Takes the shape and creates a rand tensor
# Applicable with rand, randn, randint

torch.manual_seed(3)


# Operations

x = torch.arange(6).reshape(3, 2)
print(x)
print(type(x[1,1]))     # Indices are still torch.Tensor objects

print(x[:, 1])      # Converts to a vector
print(x[:, 1:])     # Retains dimension

x = torch.arange(10)

print(x.view(2, 5))     # Just projected as this shape, but didn't reshape
print(x)
x = x.reshape(2, 5)     # Actually reshaped it

x = torch.arange(10)
z = x.view(2, 5)
print(z)                # Not a copy, stil linked

x[0] = 123
print(z)

print(x.view(2, -1))   # Calculates the second dimension

a = torch.Tensor([1, 2, 3])
b = torch.Tensor([4, 5, 6])

print(a + b)
# Same as torch.add(a, b)

a.mul(b)    # Element-wise multiplication
a.mul_(b)   # Reassignment
print(a)

print(a.dot(b))     # Dot product

a = torch.Tensor([[1, 2, 3], [4, 5, 6]])
b = torch.Tensor([[7, 8, 9], [10, 11, 12]])

print(a.mm(torch.t(b)))   # Cross product (matrix multiplication)
# Or a @ b

print(a.norm())     # Euclidean norm
print(a.numel())    # Number of elements
print(len(a))       # Number of rows

