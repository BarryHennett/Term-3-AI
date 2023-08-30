import unittest
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Import the functions from your code
from FactorySalary import load_data, preprocess_data, split_data, train_models, evaluate_models, plot_model_performance, save_best_model, predict_new_data

class TestCarPurchasePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the dataset
        cls.data = load_data('C:/Users/harra/Desktop/Term-3-AI/Car_Purchasing_Data.xlsx')

    def test_load_data(self):
        self.assertIsNotNone(self.data, "Dataset is not loaded.")

    def test_split_data(self):
        X_scaled, y_scaled, sc, sc1 = preprocess_data(self.data)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        
        self.assertEqual(X_train.shape[0] + X_test.shape[0], self.data.shape[0], "Data split is incorrect.")
		
    def test_preprocess_data(self):
        X_scaled, y_scaled, sc, sc1 = preprocess_data(self.data)
    
        self.assertIsNotNone(X_scaled)
        self.assertIsNotNone(y_scaled)
        self.assertIsNotNone(sc)
        self.assertIsNotNone(sc1)

if __name__ == '__main__':
    unittest.main()