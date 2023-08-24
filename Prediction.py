import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Import the dataset
data=pd.read_excel('C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx')

#Display the first 5 rows of the dataset (head)
#print("first 5 rows of dataset\n",data.head())

#Display the last 5 rows of the dataset (tail)
#print("last 5 rows of dataset\n",data.tail())

#Determine the shape of the dataset (shape - Total number of rows and columns)
#print("Number of rows and columns\n",data.shape)
#print("Number of rows\n",data.shape[0])
#print("Number of columns\n",data.shape[1])

#Display the concise summary of the dataset (info)
#print(data.info())

#Check the null values in dataset (isnull)
#print(data.isnull())
# OR
#print(data.isnull().sum())

#Identify the library to plot the graph to understand the relations among the various columns
#to select the independent variables, target variables and irrelevant features.
#sns.pairplot(data)
#plt.show()


#print(data.columns)
#Create the input dataset from the original dataset by dropping the irrelevant features
# store input variables in X
X= data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'],axis=1)
#print(X)

#Create the output dataset from the original dataset.
# store output variable in Y
Y= data['Car Purchase Amount']
#print(Y)

#Transform the input dataset into a percentage based weighted value between 0 and 1.
sc= MinMaxScaler()
X_scaled=sc.fit_transform(X)
#print(X_scaled)

#Transform the output dataset into a percentage based weighted value between 0 and 1
sc1= MinMaxScaler()
y_reshape= Y.values.reshape(-1,1)
y_scaled=sc1.fit_transform(y_reshape)
#print(Y_scaled)

# Print a few rows of the scaled input dataset (X)
#print("Scaled Input (X):")
#print(X_scaled[:5])

# Print a few rows of the scaled output dataset (y)
#print("Scaled Output (y):")
#print(y_scaled[:5])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

#print the shape of the test and train data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#print the first few rows
#head method can also be used but it only works with pandas dataframe but this can be work with numpy arrays 
print("First 5 rows of X_train:\n", X_train[:5])
print("First 5 rows of X_test:\n", X_test[:5])
print("First 5 rows of y_train:\n", y_train[:5])
print("First 5 rows of y_test:\n", y_test[:5])