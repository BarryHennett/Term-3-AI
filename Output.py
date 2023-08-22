import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#importing data
dataset_path = "C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx"
df = pd.read_excel(dataset_path)

print()

#output data
output_df = df[['Car Purchase Amount']]
#this outputs this only ^
print(output_df.head())


# reshape the output dataset
output_values = output_df.values 
output_values_reshaped = output_values.reshape(-1, 1) 

# normalize the values to a range between 0 and 1
# Create a MinMaxScaler object
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(output_values_reshaped)

# normal values back to a DataFrame
normalized_output_df = pd.DataFrame(normalized_values, columns=['Normalized Purchase Amount'])

print(normalized_output_df.head())