from evidently.report import Report
from evidently.metric_preset import *
from evidently.test_suite import TestSuite
from evidently.tests import *
from evidently.metrics import *
import re
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# Step 1: Fetch and prepare the dataset
# Load the Adult dataset from OpenML
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# Split the data into reference and production datasets
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
reference = dataset.sample(n=500, replace=False)
current = dataset.sample(n=200, replace=False)

# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]
model = LogisticRegression(random_state=42)
model.fit(X_reference, y_reference)

# Adding predictions to the reference and current datasets
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Step 2: Define scenarios for data drift analysis
# Scenario 1: Using default parameters
data_report = Report(
    metrics=[
        DataDriftPreset(),
    ],
)
data_report.run(reference_data=reference, current_data=current)

columns_drift_score={
    column: float(data_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][column]["drift_score"])
    for column in reference.columns
}

data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(stattest="psi", lt=0.5),
])

# Step 5: Run the data drift test
# We run the test on the first 100 rows of the production data against the reference data.
data_drift_dataset_tests.run(reference_data=reference, current_data=current)

# Step 6: Extract the share of drifted columns
# We use a regular expression to extract the percentage of drifted columns from the test description.
match = re.search(r'\b\d+(?:\.\d+)?\b', data_drift_dataset_tests.as_dict()["tests"][0]["description"])
share_of_drifted_cols = float(match.group())


# Scenario 2: Using a specific statistical test and threshold for target drift detection
# Create a report to detect target drift using Population Stability Index (PSI)
num_target_drift_report = Report(metrics=[
    TargetDriftPreset(stattest="psi", stattest_threshold=0.5),
])

# Run the report on the reference and production datasets
num_target_drift_report.run(reference_data=reference, current_data=current.iloc[0:1000, :])

# Step 5: Extract and print the drift detection results
# Extract relevant information from the report
drift_score = float(num_target_drift_report.as_dict()["metrics"][0]["result"]["drift_score"])
stattest_name = num_target_drift_report.as_dict()["metrics"][0]["result"]["stattest_name"]
stattest_threshold = num_target_drift_report.as_dict()["metrics"][0]["result"]["stattest_threshold"]
drift_detected = num_target_drift_report.as_dict()["metrics"][0]["result"]["drift_detected"]

# Create a dictionary to store the drift detection results
target_drift = {
    "drift_detected": drift_detected,
    "drift_score": drift_score,
    "stattest_name": stattest_name,
    "stattest_threshold": stattest_threshold
}

missing_values_test = TestSuite(tests=[
    TestShareOfMissingValues(),
    TestNumberOfMissingValues(),
    TestNumberOfEmptyRows(),
    TestNumberOfEmptyColumns(),
])

# Step 5: Run the test suite on the reference and production data
# Comparing the quality of production data against the reference data to identify potential issues.
# Here, we are using the first 100 rows of the production data for the test.
missing_values_test.run(reference_data=reference, current_data=current.iloc[:100, :])

# Step 6: Extract test results
# Converting the test results into a dictionary format for easier parsing and analysis.
results_dict = missing_values_test.as_dict()

# Step 7: Parse results into a structured dictionary
# Summarizing the findings from the test suite into a report format.
missing_values = {
    "share_of_missing_values": float(results_dict["tests"][0]["parameters"]["value"]),
    "number_of_missing_values": float(results_dict["tests"][1]["parameters"]["value"]),
    "number_of_empty_rows": float(results_dict["tests"][2]["parameters"]["value"]),
    "number_of_empty_columns": float(results_dict["tests"][3]["parameters"]["value"]),
}

# Step 4: Initialize a test suite for detecting Out of Range Values.
# Pass the column name and the range of values to check for out-of-range values. If the range is not passed, values will be calculated from the reference datset
share_of_out_of_range_values = TestSuite(tests=[
    TestShareOfOutRangeValues(column_name="mean area", left=0, right=100) # optional: left and right  Compulsory: column_name
])

# Step 5: Run the test suite on the reference and production data
# Here, we are using the first 100 rows of the production data for the test.
share_of_out_of_range_values.run(reference_data=reference, current_data=current.iloc[:100, :])

# Extracting the results from the test suite
out_of_range_values = float(share_of_out_of_range_values.as_dict()["tests"][0]["parameters"]["value"])

# Scenario 1: Create a report to detect correlations in the dataset
# Create a correlation report using Evidently AI
correlation = Report(metrics=[
    DatasetCorrelationsMetric()
])
# Run the correlation report on the reference and a subset of the current data
correlation.run(reference_data=reference, current_data=current)

# Get the maximum absolute Pearson correlation value from the report
max_pearson_corr = float(correlation.as_dict()["metrics"][0]["result"]["current"]["stats"]["pearson"]["abs_max_correlation"])

# Scenario 2: Create a test suite to check the correlation between target and prediction
# Create a test suite to check the correlation between the target variable and predictions
correlation_test = TestSuite(tests=[
    TestTargetPredictionCorrelation(),
    TestTargetFeaturesCorrelations(),
    TestPredictionFeaturesCorrelations(),
    TestCorrelationChanges(corr_diff=0.25), 
])
correlation_test.run(reference_data=reference, current_data=current)

# Step 6: Extract correlation values from the report and test suite

# Extract correlation values from the test suites
t_p_cor = float(correlation_test.as_dict()["tests"][0]["parameters"]["value"])
t_f_cor = float(correlation_test.as_dict()["tests"][1]["parameters"]["value"])
p_f_cor = float(correlation_test.as_dict()["tests"][2]["parameters"]["value"])
cor_change = float(correlation_test.as_dict()["tests"][3]["parameters"]["value"])

# Step 7: Store the correlation values in a dictionary
correlation = {
    "maximum_pearson_correlation": max_pearson_corr,
    "target_prediction_corr": t_p_cor,
    "target_feature_corr": t_f_cor,
    "prediction_feature_corr": p_f_cor,
    "correlation_change": cor_change
}

# Step 4: Define the tests for duplicated rows and columns
# We are using Evidently AI's TestSuite to check for duplicated rows and columns.
duplicate_rows_cols = TestSuite(
    tests=[
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns()
    ])

# Step 5: Run the tests
# Run the tests on the reference data and a subset (first 100 rows) of the production data.
duplicate_rows_cols.run(reference_data=reference, current_data=current)

# Step 6: Extract the results
# Get the number of duplicated rows and columns from the test results.
duplicated_rows = duplicate_rows_cols.as_dict()["tests"][0]["parameters"]["value"]
duplicated_cols = duplicate_rows_cols.as_dict()["tests"][1]["parameters"]["value"]

# Step 7: Create a dictionary to store the results
# Store the number of duplicated rows and columns in a dictionary.
dup_rows_cols = {
    "duplicated_rows": duplicated_rows,
    "duplicated_cols": duplicated_cols
}

# Generate a report containing the feature importance for each column.
# It uses a random forest to get the feature importances.
report = Report(metrics=[
    DataDriftTable(feature_importance=True)
])
report.run(reference_data=reference, current_data=current)

# Extract the feature importances from the report.
original_fi = report.as_dict()["metrics"][0]['result']['current_fi']

# Arrange the feature importances in descending order.
feature_importance = dict(sorted(
    {k: float(v) for k, v in original_fi.items()}.items(),
    key=lambda x: x[1],
    reverse=True
))

print(feature_importance)

def get_data_metrics():

    return {
        "columns_drift_score": columns_drift_score,
        "share_of_drifted_cols": share_of_drifted_cols,
        "target_drift": target_drift,
        "missing_values": missing_values,
        "out_of_range_values": out_of_range_values,
        "dup_rows_cols": dup_rows_cols,
        "correlation": correlation,
        "feature_importance": feature_importance
    }

"""
    This script performs various data quality and drift analysis tasks on a dataset using Evidently AI and scikit-learn libraries. The steps are as follows:

    1. **Fetch and Prepare the Dataset**:
        - Load the Breast Cancer dataset from scikit-learn.
        - Split the dataset into reference and current datasets for comparison.

    2. **Train a Logistic Regression Model**:
        - Separate features and target labels from the reference dataset.
        - Train a Logistic Regression model on the reference dataset.
        - Add predictions to both reference and current datasets.

    3. **Data Drift Analysis**:
        - **Scenario 1**: Using default parameters to detect data drift.
            - Create a data drift report using Evidently AI.
            - Calculate and print the drift score for each column.
            - Define and run a test suite to detect the share of drifted columns.
            - Extract and print the share of drifted columns.

        - **Scenario 2**: Using a specific statistical test (PSI) and threshold for target drift detection.
            - Create a target drift report using Evidently AI.
            - Run the report on the reference and current datasets.
            - Extract and print the target drift detection results.

    4. **Missing Values Analysis**:
        - Define a test suite to detect missing values.
        - Run the test suite on the reference and current datasets.
        - Extract and print the missing values statistics.

    5. **Out of Range Values Analysis**:
        - Define a test suite to detect out-of-range values for a specific column.
        - Run the test suite on the reference and current datasets.
        - Extract and print the share of out-of-range values.

    6. **Correlation Analysis**:
        - **Scenario 1**: Create a report to detect correlations in the dataset.
            - Create a correlation report using Evidently AI.
            - Run the correlation report on the reference and current datasets.
            - Extract and print the maximum absolute Pearson correlation value.

        - **Scenario 2**: Create a test suite to check the correlation between target and prediction.
            - Define and run a test suite to check various correlations.
            - Extract and print the correlation values.

    7. **Duplicated Rows and Columns Analysis**:
        - Define a test suite to detect duplicated rows and columns.
        - Run the test suite on the reference and current datasets.
        - Extract and print the number of duplicated rows and columns.

    8. **Feature Importance Analysis**:
        - Generate a report containing the feature importance for each column using a random forest.
        - Extract and print the feature importances.

    9. **Get Data Metrics**:
        - Define a function to return a dictionary containing all the calculated metrics.
"""