import numpy as np
from evidently.report import Report
from evidently.metrics import RegressionQualityMetric

from sklearn.linear_model import LinearRegression

from sklearn import datasets

# fetching the dataset
dataset = datasets.fetch_california_housing(as_frame=True)
dataset = dataset.frame

# renaming the MedHouseVal column to target and creating a column called prediction
dataset.rename(columns={'MedHouseVal': 'target'}, inplace=True)
dataset['prediction'] = dataset['target'].values + np.random.normal(0, 3, dataset.shape[0])

#divide the dataset into reference and current
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

# separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']

X_current = current[X_reference.columns]

# fitting a regression model
model = LinearRegression()

model.fit(X_reference, y_reference)

# getting the predictions for both reference and current data and adding them into respective dataframes
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# classification report containing the metrics R2 score, MAE, RMSE
regression_report = Report(metrics=[RegressionQualityMetric()])
regression_report.run(reference_data=reference, current_data=current)

# getting the standard deviation of error
current_error_std=regression_report.as_dict()["metrics"][0]["result"]["current"]["error_std"]

"""
    This script performs the following steps:

    1. Imports necessary libraries and modules:
        - pandas as pd
        - numpy as np
        - Report and RegressionQualityMetric from evidently
        - LinearRegression from sklearn.linear_model
        - datasets from sklearn

    2. Fetches the California housing dataset using sklearn.datasets.fetch_california_housing and converts it to a DataFrame.

    3. Renames the 'MedHouseVal' column to 'target' and creates a new column 'prediction' by adding random noise to the 'target' values.

    4. Divides the dataset into reference and current datasets:
        - The reference dataset serves as a benchmark or baseline for comparison.
        - The current dataset represents the new production data to be evaluated and compared against the reference dataset.

    5. Separates the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.

    6. Fits a Linear Regression model using the reference dataset.

    7. Gets the predictions for both reference and current data and adds them into respective DataFrames.

    8. Generates a regression report containing metrics such as R2 score, MAE, and RMSE using evidently's Report and RegressionQualityMetric.

    9. Calculates the standard deviation of error for the current dataset and multiplies it by 100 to get the final value.
"""
