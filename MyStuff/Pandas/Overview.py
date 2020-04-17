# Pandas = Panel Data
# Based on NumPy

# Series

# A series holds an array of info organised by an index 
# Built on a NumPy Array
# Can have a named index as well as the standard numerical one

import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
myList = [10, 20, 30]

arr = np.array(myList)

d = {'a':10, 'b':20, 'c':30}    #This is a dictionary

mySeries = pd.Series(data=myList)
mySeriesSame = pd.Series(arr)
print(mySeries)

print(len(mySeries))    # Number of rows

# The labels are now set as indices
mySeries = pd.Series(data=arr, index=labels)
print(mySeries)

# You can mix data types
print(pd.Series(data=[1.3, 'a', 3]))

countries = pd.Series([1, 2, 3, 4], index=["USA", "Germany", "USSR", "Japan"])
print(countries)
print(countries["USSR"])

countries2 = pd.Series([1, 2, 5, 6], index=["USA", "Germany", "Italy", "Japan"])

# Sum the values of matching indices, NaN if there was no match
print(countries + countries2)

# DataFrames
# Multiple series that share the same index (tables)

from numpy.random import randn

randMat = randn(5, 4)
myDataFrame = pd.DataFrame(data=randMat)

print(myDataFrame)

myDataFrame = pd.DataFrame(data=randMat, index='A B C D E'.split())
print(myDataFrame)

myDataFrame = pd.DataFrame(data=randMat, index='A B C D E'.split(), columns="W X Y Z".split())
print(myDataFrame)

# Selecting columns
# Each individual column is a series
print(myDataFrame["W"])

print(myDataFrame[["W", "Y"]])

print(myDataFrame[["Y", "W"]])


# Adding a new column
myDataFrame["Eric"] = myDataFrame["W"] + myDataFrame["Y"]
print(myDataFrame)

# Removing a row, not in place by default
myDataFrame.drop("B", 0, inplace=True)

# Removing a column
myDataFrame.drop("W", 1, inplace=True)

print(myDataFrame)

# Selecting rows

print(myDataFrame.loc["A"])
print(myDataFrame.iloc[1])

print(myDataFrame.loc[["A", "E"]])
print(myDataFrame.iloc[[0, 3]])

# Advanced selection
print(myDataFrame.loc[["C", "D"],["Y", "Z"]])
print(myDataFrame.loc[["C", "D"]][["Y", "Z"]])
print(myDataFrame.loc["A"]["X"])

# Comparison selection and filters
print(myDataFrame > 0)      # Boolean values

print(myDataFrame[myDataFrame > 0])         # Applying the boolean values to the original data frame

print(myDataFrame[myDataFrame["X"] > 0])    # Prints only rows where X > 0

print(myDataFrame[(myDataFrame["X"] > 0) & (myDataFrame["Y"] > 0)])     # Rows where X and Y are > 0
# Normally Python uses and/or but pandas uses &/|

checks = myDataFrame["X"] > 0
print(checks.value_counts())
#or alternatively
sum(checks)
 
# Remove the row names/indices
print(myDataFrame.reset_index())

newIndex = "CA NY WY OR".split()

myDataFrame["States"] = newIndex
print(myDataFrame)

newIndex = "CA NY WY OR".split()
myDataFrame.set_index("States", inplace=True)
print(myDataFrame)

# Data frame info

print(myDataFrame.info())

print(myDataFrame.dtypes)

print(myDataFrame.describe())   # Some statistics by column

# Group by

# Create dataframe
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
       'Sales':[200,120,340,124,243,350]}

df = pd.DataFrame(data)

print(df)

print(df.groupby("Company")) # Returns a strange object since Pandas doesn't know how you want to group

print(df.groupby("Company").mean()) # Groups and calculates the means
print(df.groupby("Company").max())
print(df.groupby("Company").describe()) # Offers a bunch of stats
print(df.groupby("Company").describe().transpose())




