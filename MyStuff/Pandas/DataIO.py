# Data Input/Output

import os
import pandas as pd

#Loading

df = pd.read_csv("../../PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/example.csv")

print(df)

newdf = df[["a", "b"]]
print(newdf)

#Saving

newdf.to_csv("new_csv.csv", index=False)   #Index=true keeps row indices

df2 = pd.read_excel("../../PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/excel_sample.xlsx", sheet_name="Sheet1")
df2 = df2.drop("Unnamed: 0", axis=1)  #Remove a weird thing that Excel adds in.  The unnamed column is what happens if the indices are saved.  Dropping that column/series.
print(df2)


