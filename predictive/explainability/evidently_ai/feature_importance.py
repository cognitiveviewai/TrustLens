from evidently.report import Report
from evidently.metrics import DataDriftTable
import numpy as np
from sklearn import datasets

# Fetching the data from OpenML. Here, we use the "adult" dataset.
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# Create a duplicate column, rename 'class' to 'target', and add a prediction column simulating model's predictions.
adult["dup_col"] = adult["education"]
adult.rename(columns={'class': 'target'}, inplace=True)
adult['prediction'] = np.random.choice(['<=50K', '>50K'], size=len(adult))

# Create reference and production data based on the 'education' column.
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
adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# Generate a report containing the feature importance for each column.
# It uses a random forest to get the feature importances.
report = Report(metrics=[
    DataDriftTable(feature_importance=True)
])
report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])

# Extract the feature importances from the report.
original_fi = report.as_dict()["metrics"][0]['result']['current_fi']

# Arrange the feature importances in descending order.
arranged_fi = dict(sorted(
    {k: float(v) for k, v in original_fi.items()}.items(),
    key=lambda x: x[1],
    reverse=True
))


# Comprehensive guide and walkthrough of the file:

"""
   This script demonstrates how to use the Evidently AI library to generate a feature importance report for a dataset.
   The script is structured as follows:

   1. **Data Fetching**:
      - The script fetches the "adult" dataset from OpenML using sklearn's datasets module.
      - The dataset is loaded into a pandas DataFrame.

   2. **Data Preparation**:
      - A duplicate column of 'education' is created.
      - The 'class' column is renamed to 'target'.
      - A 'prediction' column is added to simulate model predictions.

   3. **Data Splitting**:
      - The dataset is split into reference and production datasets based on the 'education' column.
      - The reference dataset excludes certain education levels.
      - The production dataset includes certain education levels.

   4. **Report Generation**:
      - A report is generated using the Evidently AI library.
      - The report includes a Data Drift Table with feature importance enabled.
      - The report is run on the reference and production datasets.

   5. **Feature Importance Extraction**:
      - The feature importances are extracted from the report.
      - The feature importances are arranged in descending order.

   ### Scenarios:
   - **Using Custom Data**:
   Replace the data fetching section with your own dataset loading code. Ensure your dataset has a similar structure.
   - **Customizing Data Splitting**:
   Modify the conditions used to split the data into reference and production datasets based on your requirements.
   - **Analyzing Different Metrics**:
   Add or replace metrics in the `Report` initialization to analyze different aspects of your data.

   ### Steps to Use:
   1. Load your dataset.
   2. Prepare your data by creating necessary columns and renaming as required.
   3. Split your data into reference and production datasets.
   4. Initialize and run the Evidently AI report.
   5. Extract and analyze the feature importances or other metrics from the report.

   This script provides a basic template for generating feature importance reports using Evidently AI. Customize it as needed for your specific use case.
"""
