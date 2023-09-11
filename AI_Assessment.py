import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge 
from sklearn.linear_model import PoissonRegressor
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
        elif NameOModel == 'Ridge Regression':
            return Ridge()
        elif NameOModel == 'Poisson Regressor':
            return PoissonRegressor()
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

def DataLoad(FileName):
    return pd.read_excel(FileName)
FileName = 'C:/Users/harra/Desktop/Term-3-AI/Net_Worth_Data.xlsx'

def preprocess_data(Data):
    if Data.isnull().any().any():
        raise ValueError("The Data has missing values. Please make sure that the Data is clean.")

    X = Data.drop(['Client Name', 'Client e-mail', 'Country','Profession','Education','Healthcare Cost','Gender','Net Worth'], axis=1)
    Y = Data['Net Worth']

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
        'Ridge Regression', 
        'Poisson Regressor', 
        'Support Vector Machine',
        'Random Forest',
        'Gradient Boosting Regressor',
        'XGBRegressor'
    ]
    
    models = {}
    for name in NameOModels:
        #Show model name
        print(f"Training model: {name}")
        model = ModelFactory.get_model(name)
        model.fit(TrainOX, TrainOY.ravel())
        models[name] = model
        #Show when model trained successfully
        print(f"{name} trained successfully.")
        
    return models

def Model_Evaluation(models, TestOX, TestOY):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(TestOX)
        rmse_values[name] = mean_squared_error(TestOY, preds, squared=False)
        
    return rmse_values

def PLotting_Model_Performance(rmse_values):
    plt.figure(figsize=(10,7))
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    bars = plt.bar(models, rmse, color=['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'brown'])

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def Best_Model_Save(models, rmse_values):
    best_NameOModel = min(rmse_values, key=rmse_values.get)
    best_model = models[best_NameOModel]
    dump(best_model, "NetWorth.joblib")
    
def New_Data_Prediction(loaded_model, sc, sc1):
    TestOX1 = sc.transform(np.array([[0,42,62812.09301,11609.38091,238961.2505]]))
    pred_value = loaded_model.predict(TestOX1)
    print(pred_value)
    
    # Make Sure pred_value is a 2D array before inverse transform
    if len(pred_value.shape) == 1:
        pred_value = pred_value.reshape(-1, 1)

    print("Predicted output: ", sc1.inverse_transform(pred_value))

if __name__ == "__main__":
    try: #add try except to handle missing value error
        data = DataLoad('Net_Worth_Data.xlsx')
        ScaledOX, ScaledOY, sc, sc1 = preprocess_data(data)
        TrainOX, TestOX, TrainOY, TestOY = split_data(ScaledOX, ScaledOY)
        models = train_models(TrainOX, TrainOY)
        rmse_values = Model_Evaluation(models, TestOX, TestOY)
        PLotting_Model_Performance(rmse_values)
        Best_Model_Save(models, rmse_values)
        loaded_model = load("NetWorth.joblib")
        New_Data_Prediction(loaded_model, sc, sc1)
    except ValueError as ValError:
        print(f"Error: {ValError}")
