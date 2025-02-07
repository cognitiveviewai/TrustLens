from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import *
from sklearn import datasets

# Step 1: Fetch the dataset
# Here we are using the 'adult' dataset from OpenML for demonstration purposes
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# Step 2: Prepare the data
# Create a duplicate column for demonstration purposes
adult["dup_col"] = adult["education"]
# Rename the 'class' column to 'target' to simulate a target variable
adult.rename(columns={'class': 'target'}, inplace=True)

# Step 3: Split the data into reference and production datasets

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

# Reference data: Exclude certain education levels
adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
# Production data: Include only certain education levels
adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# Step 4: Detect drift in the target column

# Scenario 1: Using default parameters for target drift detection
# Create a report to detect target drift using default parameters
num_target_drift_report = Report(metrics=[
    TargetDriftPreset(),
])

# Scenario 2: Using a specific statistical test and threshold for target drift detection
# Create a report to detect target drift using Population Stability Index (PSI)
num_target_drift_report = Report(metrics=[
    TargetDriftPreset(stattest="psi", stattest_threshold=0.5),
])

# Run the report on the reference and production datasets
num_target_drift_report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:1000, :])

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

"""
   Comprehensive Guide and Walkthrough:

   This script demonstrates how to detect target drift in a dataset using the Evidently AI library. Target drift occurs when the distribution of the target variable changes over time, which can affect the performance of machine learning models.

   Steps:
   1. Fetch the Dataset:
      - We use the 'adult' dataset from OpenML for this example.
      - The dataset is loaded into a DataFrame.

   2. Prepare the Data:
      - A duplicate column 'dup_col' is created for demonstration purposes.
      - The 'class' column is renamed to 'target' to simulate a target variable.

   3. Split the Data:
      - The data is split into reference and production datasets based on the 'education' column.
      - The reference dataset excludes certain education levels, while the production dataset includes them.

   4. Detect Target Drift:
      - A report is created using the Population Stability Index (PSI) to detect target drift.
      - The report is run on the reference and production datasets.

   5. Extract and Print Results:
      - Relevant information such as drift score, stattest name, stattest threshold, and drift detection status is extracted from the report.
      - The results are stored in a dictionary and printed.

   Scenarios:
   - This script can be adapted to use any dataset by replacing the dataset fetching and preparation steps.
   - Users can modify the reference and production data splitting criteria based on their specific use case.
   - The stattest and stattest_threshold parameters can be adjusted to use different statistical tests and thresholds for drift detection.
"""