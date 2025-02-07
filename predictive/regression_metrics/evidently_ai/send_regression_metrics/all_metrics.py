import pandas as pd
import numpy as np
from evidently.tests import *
from evidently.metrics import RegressionQualityMetric
from evidently.test_suite import TestSuite
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from evidently.report import Report
# Step 1: Fetching the dataset
# Here we use the California Housing dataset from sklearn for demonstration purposes.
dataset = datasets.fetch_california_housing(as_frame=True)
dataset = dataset.frame

# Step 2: Data Preparation
# Renaming the 'MedHouseVal' column to 'target' and creating a 'prediction' column with some noise added.
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
# Using Linear Regression to fit the model on the reference data.
model = LinearRegression()
model.fit(X_reference, y_reference)

# Step 6: Making predictions
# Getting the predictions for both reference and current data and adding them into respective dataframes.
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Step 7: Getting the metrics
# Computing the Absolute Maximum Error
# Using Evidently AI's TestSuite to compute the Absolute Maximum Error between the reference and current datasets.
regression_report = TestSuite(tests=[TestValueAbsMaxError()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the Absolute Maximum Error
# Extracting the Absolute Maximum Error from the regression report 
abs_max_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


# We use the Evidently AI library to compute the MAE between the reference and current datasets.
regression_report = TestSuite(tests=[TestValueMAE()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the MAE value
# Extracting the computed MAE value from the regression report.
mean_abs_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


# Computing the mean_absolute_percentage_error using Evidently AI
regression_report = TestSuite(tests=[TestValueMAPE()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the mean_absolute_percentage_error value from the report
mean_abs_perc_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]



# Computing the mean error using Evidently AI's TestSuite
regression_report = TestSuite(tests=[TestValueMeanError()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the mean error value from the report
mean_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


# Computing the r_squared_score using Evidently AI's TestSuite
regression_report = TestSuite(tests=[TestValueR2Score()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the r_squared_score from the regression report
r_squared_score = regression_report.as_dict()["tests"][0]["parameters"]["value"]


# computing the Root Mean Squared Error (RMSE)
regression_report=TestSuite(
    tests=[TestValueRMSE()])
regression_report.run(reference_data=reference, current_data=current)

# getting the Root Mean Squared Error (RMSE)
rmse=regression_report.as_dict()["tests"][0]["parameters"]["value"]


# classification report containing the metrics R2 score, MAE, RMSE
regression_report = Report(metrics=[RegressionQualityMetric()])
regression_report.run(reference_data=reference, current_data=current)

# getting the standard deviation of error
current_error_std=regression_report.as_dict()["metrics"][0]["result"]["current"]["error_std"]


def get_regression_results():
    metrics = {
        "r_square_error": r_squared_score,
        "mean_absolute_error": mean_abs_error,
        "mean_error": mean_error,
        "absolute_maximum_error": abs_max_error,
        "root_mean_squared_error": rmse,
        "std_dev_error": current_error_std,
        "mean_absolute_percentage_error": mean_abs_perc_error
    }
    return metrics

"""
    This script performs regression analysis on the California Housing dataset using Evidently AI for metric computation.

    Steps:
    1. **Fetching the dataset**:
        - The California Housing dataset is fetched from sklearn and loaded into a pandas DataFrame.

    2. **Data Preparation**:
        - The 'MedHouseVal' column is renamed to 'target'.
        - A 'prediction' column is created by adding some noise to the 'target' values.

    3. **Splitting the dataset**:
        - The dataset is split into two parts: reference and current datasets, each containing 5000 samples.

    4. **Separating features and target labels**:
        - Features (X_reference and X_current) and target labels (y_reference) are separated from the reference and current datasets.

    5. **Fitting a regression model**:
        - A Linear Regression model is fitted on the reference data.

    6. **Making predictions**:
        - Predictions are made for both reference and current datasets using the fitted model.

    7. **Getting the metrics**:
        - Various regression metrics are computed using Evidently AI's TestSuite:
            - Absolute Maximum Error
            - Mean Absolute Error (MAE)
            - Mean Absolute Percentage Error (MAPE)
            - Mean Error
            - R-Squared Score
            - Root Mean Squared Error (RMSE)
        - The computed metrics are extracted.

    8. **Generating a regression report**:
        - A regression report containing metrics like R2 score, MAE, and RMSE is generated using Evidently AI's Report.
        - The standard deviation of error is extracted.

    9. **Function to get regression results**:
        - `get_regression_results()`: Returns a dictionary containing all the computed regression metrics.
"""