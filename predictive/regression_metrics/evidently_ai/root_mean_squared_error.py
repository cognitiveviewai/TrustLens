import numpy as np
from evidently.tests import TestValueRMSE
from evidently.test_suite import TestSuite

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

# computing the Root Mean Squared Error (RMSE)
regression_report=TestSuite(
    tests=[TestValueRMSE()])
regression_report.run(reference_data=reference, current_data=current)

# getting the Root Mean Squared Error (RMSE)
rmse=regression_report.as_dict()["tests"][0]["parameters"]["value"]

"""
    This script performs the following steps:

    1. Imports necessary libraries:
        - pandas as pd
        - numpy as np
        - TestValueRMSE and TestSuite from evidently.tests and evidently.test_suite
        - LinearRegression from sklearn.linear_model
        - datasets from sklearn

    2. Fetches the California housing dataset using sklearn's datasets.fetch_california_housing function and stores it in a DataFrame.

    3. Renames the 'MedHouseVal' column to 'target' and creates a new column 'prediction' by adding random noise to the 'target' values.

    4. Divides the dataset into two subsets: reference and current. The reference dataset serves as a benchmark or baseline for comparison, while the current dataset represents new production data to be evaluated.

    5. Separates the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.

    6. Fits a Linear Regression model using the reference dataset.

    7. Predicts the target values for both reference and current datasets using the fitted model and adds the predictions to the respective DataFrames.

    8. Computes the Root Mean Squared Error (RMSE) using evidently's TestSuite and TestValueRMSE.

    9. Extracts the RMSE value from the regression report and stores it in the variable 'rmse'.`
"""