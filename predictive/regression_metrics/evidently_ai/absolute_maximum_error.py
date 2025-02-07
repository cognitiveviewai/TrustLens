import pandas as pd
import numpy as np
from evidently.tests import TestValueAbsMaxError
from evidently.test_suite import TestSuite
from sklearn.linear_model import LinearRegression
from sklearn import datasets

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

# Step 7: Computing the Absolute Maximum Error
# Using Evidently AI's TestSuite to compute the Absolute Maximum Error between the reference and current datasets.
regression_report = TestSuite(tests=[TestValueAbsMaxError()])
regression_report.run(reference_data=reference, current_data=current)

# Step 8: Extracting the Absolute Maximum Error
# Extracting the Absolute Maximum Error from the regression report.
abs_max_error = regression_report.as_dict()["tests"][0]["parameters"]["value"]


"""
    Comprehensive Guide and Walkthrough:

    1. **Fetching the Dataset**:
        - We use the California Housing dataset from sklearn for demonstration purposes. This dataset contains information about various features of houses in California and their median house values.
        - Users can replace this dataset with their own data for evaluation.

    2. **Data Preparation**:
        - We rename the 'MedHouseVal' column to 'target' to represent the target variable we want to predict.
        - We create a 'prediction' column by adding some random noise to the target values. This is just for demonstration and simulates a scenario where we have some initial predictions.

    3. **Splitting the Dataset**:
        - We split the dataset into two parts: reference and current datasets.
        - The reference dataset serves as a benchmark or baseline for comparison. It reflects realistic data patterns and represents typical scenarios in your data domain.
        - The current dataset represents the new production data that you want to evaluate and compare against the reference dataset.

    4. **Separating Features and Target Labels**:
        - We separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.

    5. **Fitting a Regression Model**:
        - We use Linear Regression to fit a model on the reference data. This model will be used to make predictions on both reference and current datasets.

    6. **Making Predictions**:
        - We get the predictions for both reference and current data using the fitted model and add them into the respective dataframes.

    7. **Computing the Absolute Maximum Error**:
        - We use Evidently AI's TestSuite to compute the Absolute Maximum Error between the reference and current datasets. This helps in understanding the maximum deviation in predictions.

    8. **Extracting the Absolute Maximum Error**:
        - We extract the Absolute Maximum Error from the regression report. This value indicates the maximum error observed in the predictions.

    This guide provides a step-by-step walkthrough of how to use a regression model to make predictions and evaluate the model's performance using the Absolute Maximum Error metric. Customers can replace the dataset with their own data and follow the same steps to evaluate their models.
"""
