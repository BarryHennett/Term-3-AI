import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import numpy as np

class ModelFactory:
    @staticmethod
    def get_model(NameOModel):
        if NameOModel == 'Linear Regression':
            return LinearRegression()
        elif NameOModel == 'Support Vector Machine':
            return SVR()
        elif NameOModel == 'Random Forest':
            return RandomForestRegressor()
        elif NameOModel == 'Gradient Boosting Regressor':
            return GradientBoostingRegressor()
        elif NameOModel == 'XGBRegressor':
            return XGBRegressor()
        else:
            raise ValueError(f"Model '{NameOModel}' not recognized!")

def load_data(NameOFile):
    return pd.read_excel(NameOFile)

def preprocess_data(data):
# Checking for any missing values
    if data.isnull().any().any():
        raise ValueError("The data contains missing values! Please check the data is cleaned before processing.")

    X = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
    Y = data['Car Purchase Amount']
    
    sc = MinMaxScaler()
    ScaledOX = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    ReshapeOY = Y.values.reshape(-1, 1)
    ScaledOY = sc1.fit_transform(ReshapeOY)
    
    return ScaledOX, ScaledOY, sc, sc1

def split_data(ScaledOX, ScaledOY):
    return train_test_split(ScaledOX, ScaledOY, test_size=0.2, random_state=42)

def train_models(TrainOX, TrainOY):
    NameOModels = [
        'Linear Regression',
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor'
    ]
    
    models = {}
    for name in NameOModels:
        #Display model name
        print(f"Training model: {name}")
        model = ModelFactory.get_model(name)
        model.fit(TrainOX, TrainOY.ravel())
        models[name] = model
        #display when model trained successfully
        print(f"{name} trained successfully.")
        
    return models


def evaluate_models(models, TestOX, TestOY):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(TestOX)
        rmse_values[name] = mean_squared_error(TestOY, preds, squared=False)
        
    return rmse_values

def plot_model_performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def save_best_model(models, rmse_values):
    best_NameOModel = min(rmse_values, key=rmse_values.get)
    best_model = models[best_NameOModel]
    dump(best_model, "train_model.joblib")

def predict_new_data(loaded_model, sc, sc1):
    TestOX1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))
    pred_value = loaded_model.predict(TestOX1)
    print(pred_value)
    
    # Ensure pred_value is a 2D array before inverse transform
    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)

    print("Predicted output: ", sc1.inverse_transform(pred_value))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        data = load_data('train.xlsx')
        ScaledOX, ScaledOY, sc, sc1 = preprocess_data(data)
        TrainOX, TestOX, TrainOY, TestOY = split_data(ScaledOX, ScaledOY)
        models = train_models(TrainOX, TrainOY)
        rmse_values = evaluate_models(models, TestOX, TestOY)
        plot_model_performance(rmse_values)
        save_best_model(models, rmse_values)
        loaded_model = load("train_model.joblib")
        predict_new_data(loaded_model, sc, sc1)
    except ValueError as ValError:
        print(f"Error: {ValError}")
