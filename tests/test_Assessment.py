from AI_Assessment import DataLoad, preprocess_data, split_data
import pandas as pd
import numpy as np

def test_DataLoad():
    data = DataLoad('Net_Worth_Data.xlsx')
    assert isinstance(data, pd.DataFrame), "Loaded data Must be DataFrame."

    expected_columns = ['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Gender', 'Age',
                        'Income', 'Credit Card Debt', 'Healthcare Cost', 'Inherited Amount', 'Stocks', 'Bonds',
                        'Mutual Funds', 'ETFs', 'REITs', 'Net Worth']
    
    for col in expected_columns:
        assert col in data.columns, f"Expected column {col} not found in loaded data."

def test_data_range():
    data = DataLoad('Net_Worth_Data.xlsx')
    X_scaled, y_scaled, _, _ = preprocess_data(data)
    
    assert 0 <= np.min(X_scaled) <= 1, "X must be scale between 0 and 1."
    assert 0 <= np.min(y_scaled) <= 1, "Y must be scale between 0 and 1."
    assert 0 <= np.max(X_scaled) <= 1, "X must be scale between 0 and 1."
    assert 0 <= np.max(y_scaled) <= 1, "Y must be scale between 0 and 1."

def test_data_split():
    data = DataLoad('Net_Worth_Data.xlsx')
    X, Y, _, _ = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, Y)
    assert X_train.shape[0] / X.shape[0] == 0.8
    assert X_test.shape[0] / X.shape[0] == 0.2
    assert y_train.shape[0] / Y.shape[0] == 0.8
    assert y_test.shape[0] / Y.shape[0] == 0.2