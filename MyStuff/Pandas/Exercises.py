import pandas as pd

df = pd.read_csv("../../PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/bank.csv")

print(df.iloc[0:5])

print(df["age"].mean())

print(df["marital"][df["age"] == df["age"].min()])

print(df["job"].nunique())

print(df["job"].value_counts())

print(len(df[df["marital"] == "married"]) / len(df) * 100)

df["default code"] = df["default"].map({"no" : 0, "yes" : 1})

print(df[["default", "default code"]])

def shortMaritalCode(full):
    return {
        "married" : "m",
        "divorced" : "d",
        "single" : "s",
    } [full]

df["marital code"] = df["marital"].apply(shortMaritalCode)

print(df[["marital", "marital code"]])

print(df["duration"].max())

print(df["education"][df["job"] == "unemployed"].value_counts())

print(df["age"][df["job"] == "unemployed"].mean())