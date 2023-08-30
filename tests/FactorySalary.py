import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

file_path = "C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx"


class ModelFactory:
    def create_model(self, model_type):
        if model_type == 'LinearRegression':
            return LinearRegression()
        elif model_type == 'SVR':
            return SVR()
        elif model_type == 'RandomForest':
            return RandomForestRegressor()
        elif model_type == 'GradientBoosting':
            return GradientBoostingRegressor()
        elif model_type == 'XGB':
            return XGBRegressor()
        else:
            raise ValueError(f"Invalid model type: {model_type}")

# Method to train models
def train_models(X_train, y_train):
    lr = LinearRegression()
    svm = SVR()
    rf = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    xg = XGBRegressor()
    
    lr.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    gbr.fit(X_train, y_train)
    xg.fit(X_train, y_train)
    
    return lr, svm, rf, gbr, xg


# Method to load data
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Method to preprocess data
def preprocess_data(data):
    X = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
    Y = data['Car Purchase Amount']
    
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

# Method to split data into training and testing sets
def split_data(X_scaled, y_scaled, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Method to evaluate model performance
def evaluate_models(models, X_test, y_test):
    model_names = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor']
    rmse_values = []
    
    for model in models:
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        rmse_values.append(rmse)
    
    return model_names, rmse_values

# Method to plot model performance
def plot_model_performance(model_names, rmse_values):
    plt.figure(figsize=(10, 7))
    bars = plt.bar(model_names, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Method to save the best model
def save_best_model(models, rmse_values):
    best_model_index = rmse_values.index(min(rmse_values))
    best_model_object = models[best_model_index]
    
    dump(best_model_object, "car_model.joblib")

# Method to predict new data
def predict_new_data(model, scaler_x, scaler_y, input_data):
    input_data_scaled = scaler_x.transform(input_data)
    pred_value = model.predict(input_data_scaled)
    pred_value_rescaled = scaler_y.inverse_transform(pred_value)
    return pred_value_rescaled

# Load the data
data = load_data('Car_Purchasing_Data.xlsx')

# Preprocess the data
X_scaled, y_scaled, sc, sc1 = preprocess_data(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)

# Train models
model_factory = ModelFactory()
models = train_models(X_train, y_train)

# Evaluate model performance
model_names, rmse_values = evaluate_models(models, X_test, y_test)

# Plot model performance
plot_model_performance(model_names, rmse_values)

# Save the best model
save_best_model(models, rmse_values)

# Load the best model
loaded_model = load("car_model.joblib")

# Gather user inputs
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = int(input("Enter age: "))
annual_salary = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
net_worth = float(input("Enter net worth: "))

# Predict on new data
input_data = [[gender, age, annual_salary, credit_card_debt, net_worth]]
pred_value = predict_new_data(loaded_model, sc, sc1, input_data)
print("Predicted Car Purchase Amount based on input:", pred_value)
