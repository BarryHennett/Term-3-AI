import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#importing data
dataset_path = "C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx"
df = pd.read_excel(dataset_path)

print()

#list of irrelevant features
irrelevant = ['Customer Name', 'Customer e-mail', 'Country']
#new data without ^
input_df = df.drop(columns=irrelevant)

#prints the new data
print(input_df.head())

scaler = MinMaxScaler()

sinput = scaler.fit_transform(input_df)

scaled_input_df = pd.DataFrame(sinput, columns=input_df.columns)
print("Scaled Input Data:")
print(scaled_input_df.head())

# output data scaled
output_df = df[['Car Purchase Amount']]
print("Output Data:")
print(output_df.head())