from evidently.test_suite import TestSuite
from evidently.tests import *
import re
from sklearn import datasets

# Step 1: Fetch the data
# Here we are using the 'adult' dataset from OpenML. This dataset is used for predicting whether income exceeds $50K/yr based on census data.
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# Step 2: Data Preparation
# Create a duplicate column 'dup_col' from the 'education' column.
# Rename the 'class' column to 'target' to simulate a target variable.
adult["dup_col"] = adult["education"]
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

adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# Step 4: Define the tests for duplicated rows and columns
# We are using Evidently AI's TestSuite to check for duplicated rows and columns.
duplicate_rows_cols = TestSuite(
    tests=[
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns()
    ])

# Step 5: Run the tests
# Run the tests on the reference data and a subset (first 100 rows) of the production data.
duplicate_rows_cols.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])

# Step 6: Extract the results
# Get the number of duplicated rows and columns from the test results.
duplicated_rows = duplicate_rows_cols.as_dict()["tests"][0]["parameters"]["value"]
duplicated_cols = duplicate_rows_cols.as_dict()["tests"][1]["parameters"]["value"]

# Step 7: Create a dictionary to store the results
# Store the number of duplicated rows and columns in a dictionary.
duplicated_dict = {
    "duplicated_rows": duplicated_rows,
    "duplicated_cols": duplicated_cols
}


"""
   Comprehensive Guide and Walkthrough:

   This script is designed to help users identify duplicated rows and columns in their datasets using Evidently AI's testing suite. Below is a step-by-step guide to understand and use this script with your own data.

   1. **Fetch the Data**:
      - The script starts by fetching the 'adult' dataset from OpenML. This dataset is commonly used for classification tasks.

   2. **Data Preparation**:
      - A duplicate column 'dup_col' is created from the 'education' column to simulate a scenario where duplicated columns might exist.
      - The 'class' column is renamed to 'target' to simulate a target variable in a typical machine learning dataset.

   3. **Split the Data**:
      - The dataset is split into reference and production datasets based on the 'education' column. This simulates a scenario where the reference dataset might be different from the production dataset.

   4. **Define the Tests**:
      - Two tests are defined using Evidently AI's TestSuite: one for checking duplicated rows and another for checking duplicated columns.

   5. **Run the Tests**:
      - The tests are run on the reference data and a subset (first 100 rows) of the production data. This step checks for duplicated rows and columns in the datasets.

   6. **Extract the Results**:
      - The results of the tests are extracted to get the number of duplicated rows and columns.

   7. **Store the Results**:
      - The number of duplicated rows and columns is stored in a dictionary for easy access and further use.



   **Usage with Your Own Data**:
   - Replace the data fetching step with your own dataset.
   - Adjust the data preparation steps as needed based on your dataset's structure.
   - Modify the conditions for splitting the data into reference and production datasets according to your requirements.
   - Run the script to identify duplicated rows and columns in your dataset.

   This script provides a comprehensive approach to identifying data quality issues related to duplicated rows and columns, which is crucial for maintaining the integrity of your machine learning models.
"""
