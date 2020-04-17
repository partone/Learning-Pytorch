import numpy as np
myList = [1, 2, 3]

np.array(myList)        #Convert to numPy array

type(myList)        #Still the same type

arr = np.array(myList)      #This is an array

myList = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] #Python's sad excuse for an array

myMatrix = np.array(myList)        #This is a real array

print(myMatrix)
print(myMatrix.shape)   #The dimensions of the array

print(np.arange(0, 10))     #Creates an array from 0 to 9
print(np.arange(0, 10, 2))  #Same as above but step two

print(np.zeros(5))      #Creates a vector of zeros
#0. indicates it's a float of 0

print(np.ones((4, 5)))   #Creates an array of 4x5 ones

print(np.ones((5, 5)) + 4)  #Prints a 5x5 of 5s, can't be done with a normal Python list
#Works with other operators, called "broadcast"

print(np.linspace(0, 10, 20))  #Twenty numbers evenly spaced between 0 and 10 (start and end inclusive)

print(np.eye(3))    #Identity matrix

print(np.random.rand(4))     #4 random numbers between 0, 1 / Uniform distribution
#Can be done to make an array too

print(np.random.randn(3, 4))    #Random numbers with a normal distribution, mean = 0, standard deviation = 1

print(np.random.randint(1, 100, 10))   #10 integers between 1 and 100

#np.random.seed(42)         #Set the seed for random numbers

arr = np.arange(25)    #0-24
ranarr = np.random.randint(0, 50, 10)   #10 random numbers

arr.reshape(5, 5)   #arr becomes a 5x5, dimensions must make sense

print(ranarr.max())     #Max value in ranarr
print(ranarr.argmax())  #Max value index in ranarr

print(ranarr.dtype)     #Data type held by the array

arr.reshape(1, 25)

#Slicing
print(arr[8])
print(arr[3:5])
print(arr[:7])

print(arr ** 2)

arrCopy = arr.copy()            #Create a copy
sliceOfCopy = arrCopy[:5]       #Assign the first 5 numbers
sliceOfCopy[:] = 99             #Make them 99
print(sliceOfCopy)              #Now all 99s
print(arrCopy)                  #sliceOfCopy was just a pointer evidently
print(arr)                      #Only arrCopy was a true copy

arr2d = arr.copy().reshape(5, 5)
print(arr2d)

print(arr2d[1, :])  #Equivalent to arr2d[1]
print(arr2d[:, 3])
print(arr2d[4, 4])  #Equivalent to arr2d[4][4]

print(arr2d[:2, :2])
print(arr2d[1:3, 1:3])

print(arr2d > 10)           #Applies the comparison to all values in the array
print(arr2d[arr2d > 10])    #Return the values that give true

print(np.sqrt(arr2d))

print(arr2d.mean())
print(arr2d.sum())

print(arr2d.sum(0)) #Gives the sum of each column

