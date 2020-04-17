import numpy as np

print(np.zeros(10))

print(np.ones(10))

print(np.ones(10) * 5)

print(np.arange(10, 51))

print(np.arange(10, 51, 2))

print(np.arange(0, 9).reshape(3, 3))

print(np.eye(3))

print(np.random.rand())

print(np.random.randn(5, 5))

print(np.linspace(0.01, 1, 100).reshape(10, 10))

print(np.linspace(0, 1, 20))

mat = np.arange(1,26).reshape(5,5)

print(mat[2:, 1:])

print(mat[3][4])

print(mat[0:3, 1:2])

print(mat[4])

print(mat[3:5])

print(mat.sum())

print(mat.std())

print(mat.sum(0))