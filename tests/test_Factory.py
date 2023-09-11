from FinalCode import load_data, preprocess_data, split_data
import pandas as pd
import numpy as np

#Test case to ensure the correct loading of data
def test_load_data():
    data = load_data('Car_Purchasing_Data.xlsx')
    assert isinstance(data, pd.DataFrame), "Loaded data should be a DataFrame."

    Col_Exp = ['Customer Name', 'Customer e-mail', 'Country', 'Gender', 
                        'Annual Salary', 'Credit Card Debt', 'Net Worth', 'Car Purchase Amount']

    for col in Col_Exp:
        assert col in data.columns, f"Expected column {col} not found in loaded data."

#Test case to ensure the correct shape of data
def test_shape_of_data():
    data = load_data('Car_Purchasing_Data.xlsx')
    sca_X, sca_Y, _, _ = preprocess_data(data)
    
    assert sca_X.shape[1] == 5, "Expected 5 features in the X data after preprocessing."
    assert sca_Y.shape[1] == 1, "Expected Y data to have a single column."

#Test case to ensure the correct columns for Input
def test_columns_X():
    data = load_data('Car_Purchasing_Data.xlsx')
    X, _, _, _ = preprocess_data(data)
    input_columns = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
    # Convert the NumPy array to a DataFrame
    X_DataFrame = pd.DataFrame(X, columns=input_columns)
    # Check if X_DataFrame is a DataFrame
    assert isinstance(X_DataFrame, pd.DataFrame)
    # Check that the columns have been dropped for X
    assert "Customer Name" not in X_DataFrame.columns
    assert "Customer e-mail" not in X_DataFrame.columns
    assert "Country" not in X_DataFrame.columns
    assert "Car Purchase Amount" not in X_DataFrame.columns

#Test case to ensure the correct column for output
def test_columns_Y():
    data = load_data('Car_Purchasing_Data.xlsx')
    _, Y, _, _ = preprocess_data(data)
    # Convert the NumPy array to a DataFrame
    Y_DataFrame = pd.DataFrame(Y, columns=['Car Purchase Amount'])
    
    # Check if Y_DataFrame is a DataFrame and has the correct column name
    assert isinstance(Y_DataFrame, pd.DataFrame)
    assert Y_DataFrame.columns == 'Car Purchase Amount'

#Test case to ensure the correct range of data
def test_data_range():
    data = load_data('Car_Purchasing_Data.xlsx')
    sca_X, sca_Y, _, _ = preprocess_data(data)
    
    assert 0 <= np.min(sca_X) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.min(sca_Y) <= 1, "Y data should be scaled between 0 and 1."
    assert 0 <= np.max(sca_X) <= 1, "X data should be scaled between 0 and 1."
    assert 0 <= np.max(sca_Y) <= 1, "Y data should be scaled between 0 and 1."

#Test case to ensure the correct splitting of data
def test_data_split():
    data = load_data('Car_Purchasing_Data.xlsx')
    X, Y, _, _ = preprocess_data(data)
    Train_X, Test_X, Train_Y, Test_Y = split_data(X, Y)
    # Check proportions for train-test split
    assert Train_X.shape[0] / X.shape[0] == 0.8
    assert Test_X.shape[0] / X.shape[0] == 0.2
    assert Train_Y.shape[0] / Y.shape[0] == 0.8
    assert Test_Y.shape[0] / Y.shape[0] == 0.2