import torch


x = torch.tensor(2.0, requires_grad=True)       #Generates a gradient function when used
y = 2 * x**4 + x**3 + 3 * x**2 + 5*x + 1        #Standard polynomial function
print(y)    # Shows gradient function

y.backward()    # Perform back propagation

print(x.grad)   # Represents y'(x)
# The slope at x = 2 of y is x.grad


x = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
y = 3 * x + 2   # First layer
z = 2 * y**2    # Second layer

print(z)

out = z.mean()  #Output layer
print(out)

out.backward()
print(x.grad)   # Bam





