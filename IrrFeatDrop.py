import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import the data
dataset_path = "C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx"
data = pd.read_excel(dataset_path)
print()

#excluding not needed columns
NotNeed = ['Customer Name', 'Customer e-mail', 'Country']

input_data = data.drop(columns=NotNeed)

print(input_data.head())


output_data = data[['Car Purchase Amount']]
print(output_data.head())