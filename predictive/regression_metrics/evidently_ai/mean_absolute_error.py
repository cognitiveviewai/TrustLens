import numpy as np
from evidently.tests import TestValueMAE
from evidently.test_suite import TestSuite
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Step 1: Fetching the dataset
# Here, we use the California Housing dataset from sklearn for demonstration purposes.
# In practice, you should replace this with your own dataset.
dataset = datasets.fetch_california_housing(as_frame=True)
dataset = dataset.frame

# Step 2: Data Preparation
# Renaming the 'MedHouseVal' column to 'target' and creating a 'prediction' column with some noise.
dataset.rename(columns={'MedHouseVal': 'target'}, inplace=True)
dataset['prediction'] = dataset['target'].values + np.random.normal(0, 3, dataset.shape[0])

# Step 3: Splitting the dataset into reference and current datasets
'''
The reference dataset serves as a benchmark or baseline for comparison. It can be:
-> Historical data from a previous production period
-> A stable dataset that reflects realistic data patterns
-> Contains a large enough sample to derive meaningful statistics
-> A baseline against which new data can be compared

The reference dataset reflects realistic data patterns and seasonality and represents typical scenarios in your data domain.

The current dataset represents the new production data that you want to evaluate and compare against the reference dataset. 
This is the data you are monitoring in real-time or during a new production cycle.
'''
reference = dataset.sample(n=5000, replace=False)
current = dataset.sample(n=5000, replace=False)

# Step 4: Separating features and target labels
# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Step 5: Fitting a regression model
# We fit a Linear Regression model using the reference dataset.
model = LinearRegression()
model.fit(X_reference, y_reference)

# Step 6: Making predictions
# Getting the predictions for both reference and current data and adding them into respective dataframes.
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Step 7: Computing the Mean Absolute Error (MAE)
# We use the Evidently AI library to compute the MAE between the reference and current datasets.
regression_report = TestSuite(tests=[TestValueMAE()])
regression_report.run(reference_data=reference, current_data=current)

# Step 8: Extracting the MAE value
# Extracting the computed MAE value from the regression report.
mean_abs_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


"""
    Comprehensive Guide and Walkthrough:
    This script demonstrates how to compute the Mean Absolute Error (MAE) between a reference dataset and a current dataset using a Linear Regression model.

    Steps:
    1. Fetch the dataset: We use the California Housing dataset for demonstration. Replace this with your own dataset.
    2. Data Preparation: Rename the target column and add a prediction column with some noise.
    3. Split the dataset: Divide the dataset into reference and current datasets. The reference dataset serves as a baseline, while the current dataset represents new production data.
    4. Separate features and target labels: Extract features and target labels from both reference and current datasets.
    5. Fit a regression model: Train a Linear Regression model using the reference dataset.
    6. Make predictions: Generate predictions for both reference and current datasets using the trained model.
    7. Compute MAE: Use the Evidently AI library to compute the MAE between the reference and current datasets.
    8. Extract MAE value: Retrieve the computed MAE value from the regression report.

    This guide helps you understand how to evaluate the performance of your regression model by comparing predictions on new data against a reference dataset.
"""
