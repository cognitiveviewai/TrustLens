import numpy as np
from evidently.tests import TestValueMAPE
from evidently.test_suite import TestSuite
from sklearn.linear_model import LinearRegression
from sklearn import datasets

# Fetching the California housing dataset
# This dataset is used for demonstration purposes
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

# Computing the mean_absolute_percentage_error using Evidently AI
regression_report = TestSuite(tests=[TestValueMAPE()])
regression_report.run(reference_data=reference, current_data=current)

# Extracting the mean_absolute_percentage_error value from the report
mean_abs_perc_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


"""
    Comprehensive guide and walkthrough of the file:
    1. Import necessary libraries: pandas, numpy, evidently, sklearn.
    2. Fetch the California housing dataset using sklearn's datasets module.
    3. Rename the 'MedHouseVal' column to 'target' and create a 'prediction' column with some random noise.
    4. Split the dataset into reference and current datasets. The reference dataset serves as a baseline for comparison, while the current dataset represents new production data.
    5. Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
    6. Fit a Linear Regression model using the reference dataset.
    7. Predict the target values for both reference and current datasets using the trained model and add these predictions to the respective dataframes.
    8. Use Evidently AI's TestSuite to compute the Mean Absolute Percentage Error (MAPE) between the reference and current datasets.
    9. Extract the MAPE value from the regression report.

    Scenarios where this guide is useful:
    - Comparing historical data with new production data to monitor model performance.
    - Evaluating the stability of a regression model over time.
    - Detecting significant deviations in predictions that may indicate model drift or data quality issues.
"""

