#Operations

import pandas as pd

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']})

print(df)

print(df["col2"].unique())      #See unique values

print(df["col2"].nunique())     #See number of unique vaules

print(df["col2"].value_counts())    #See count of each value


newdf = df[(df["col1"] > 2) & (df["col2"] == 444)]
print(newdf)

#Creating functions

def timesTwo(n):
    return n*2

print(timesTwo(2))

#Apply to the data frame

df["new"] = df["col1"].apply(timesTwo)
print(df)

#Remove a column

del df["new"]       #This is in place
print(df)

#Attributes

print(df.columns)

print(df.index)

#Sorting

print(df.sort_values("col2"))

print(df.sort_values("col2",ascending=False))

#Note sorting maintains index values