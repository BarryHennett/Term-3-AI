import pandas as pd

#importing data
dataset_path = "C:/Users/harra/Documents/Term-3-AI/Car_Purchasing_Data.xlsx"
df = pd.read_excel(dataset_path)

print("Head")
print(df.head())

print("Tail")
print(df.tail())