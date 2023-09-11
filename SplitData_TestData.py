import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# iImporting data
dataset_path = "C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx"
df = pd.read_excel(dataset_path)

# list of irrelevant features
irrelevant = ['Customer Name', 'Customer e-mail', 'Country']

# new data without irrelevant features
input_df = df.drop(columns=irrelevant)

# output data
output_df = df[['Car Purchase Amount']]

# splitting the dataset
train_input, test_input, train_output, test_output = train_test_split(input_df, output_df, test_size=0.3, random_state=42)

# initializing MinMaxScaler
scaler = MinMaxScaler()

# Scaling the input data
train_input_scaled = scaler.fit_transform(train_input)
test_input_scaled = scaler.transform(test_input)

# convert scaled arrays back to DataFrames
train_input_scaled_df = pd.DataFrame(train_input_scaled, columns=train_input.columns)
test_input_scaled_df = pd.DataFrame(test_input_scaled, columns=test_input.columns)

# using Seaborn to visualize
sns.pairplot(train_input_scaled_df)
plt.show()

# first rows of train input data and test input
print("\nFirst few rows of train input dataset:")
print(train_input.head())

print("\nFirst few rows of test input dataset:")
print(test_input.head())

# frist rows of train output data and test output
print("\nFirst few rows of train output dataset:")
print(train_output.head())

print("\nFirst few rows of test output dataset:")
print(test_output.head())