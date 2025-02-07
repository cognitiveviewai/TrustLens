import numpy as np
from evidently.tests import TestValueMeanError
from evidently.test_suite import TestSuite
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Fetching the California housing dataset
dataset = datasets.fetch_california_housing(as_frame=True)
dataset = dataset.frame

# Renaming the MedHouseVal column to target and creating a column called prediction
dataset.rename(columns={'MedHouseVal': 'target'}, inplace=True)
dataset['prediction'] = dataset['target'].values + np.random.normal(0, 3, dataset.shape[0])

# Divide the dataset into reference and current
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

# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Fitting a regression model
model = LinearRegression()
model.fit(X_reference, y_reference)

# Getting the predictions for both reference and current data and adding them into respective dataframes
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Computing the mean error using Evidently AI's TestSuite
regression_report = TestSuite(tests=[TestValueMeanError()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the mean error value from the report
mean_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]

# Comprehensive guide and walkthrough of the file:
'''
    1. Import necessary libraries:
        - pandas and numpy for data manipulation.
        - evidently for computing regression metrics.
        - sklearn for regression model and dataset.

    2. Fetch the California housing dataset and prepare it:
        - Rename the target column.
        - Add a prediction column with some noise.

    3. Split the dataset into reference and current datasets:
        - Reference dataset: Used as a benchmark for comparison.
        - Current dataset: Represents new production data.

    4. Separate features and target labels:
        - X_reference and y_reference for the reference dataset.
        - X_current for the current dataset.

    5. Fit a Linear Regression model using the reference dataset.

    6. Predict and update the prediction column for both reference and current datasets.

    7. Compute the mean error using Evidently AI's TestSuite:
        - Create a TestSuite with TestValueMeanError.
        - Run the TestSuite with reference and current data.
        - Extract the mean error value from the report.

    Scenarios:
    - Historical data comparison: Use historical data as the reference dataset to monitor changes in new data.
    - Real-time monitoring: Continuously update the current dataset with new production data and compare it against the reference dataset.
    - Model performance evaluation: Assess the performance of a regression model by comparing predictions on reference and current datasets.
'''

