from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfMissingValues, TestNumberOfMissingValues, TestNumberOfEmptyRows, TestNumberOfEmptyColumns
from sklearn import datasets

# Step 1: Load the dataset
# Using the "adult" dataset from OpenML to analyze missing values and dataset integrity.
# This dataset is commonly used for machine learning and contains demographic information.
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame=True)
adult = adult_data.frame

# Step 2: Prepare the dataset
# Scenario: Simulating a real-world scenario where the dataset contains duplicate columns and needs renaming for clarity.
# Adding a duplicate column for demonstration purposes.
adult["dup_col"] = adult["education"]
# Renaming the 'class' column to 'target' for better clarity.
adult.rename(columns={'class': 'target'}, inplace=True)

# Step 3: Split the dataset into reference and production subsets

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

# Step 4: Initialize a test suite for detecting missing values and empty structures
# Creating a test suite to evaluate the dataset for missing values, empty rows, and columns to ensure data quality.
missing_values_test = TestSuite(tests=[
    TestShareOfMissingValues(),
    TestNumberOfMissingValues(),
    TestNumberOfEmptyRows(),
    TestNumberOfEmptyColumns(),
])

# Step 5: Run the test suite on the reference and production data
# Comparing the quality of production data against the reference data to identify potential issues.
# Here, we are using the first 100 rows of the production data for the test.
missing_values_test.run(reference_data=adult_ref, current_data=adult_prod.iloc[:100, :])

# Step 6: Extract test results
# Converting the test results into a dictionary format for easier parsing and analysis.
results_dict = missing_values_test.as_dict()

# Step 7: Parse results into a structured dictionary
# Summarizing the findings from the test suite into a report format.
missing_values_summary = {
    "share_of_missing_values": float(results_dict["tests"][0]["parameters"]["value"]),
    "number_of_missing_values": float(results_dict["tests"][1]["parameters"]["value"]),
    "number_of_empty_rows": float(results_dict["tests"][2]["parameters"]["value"]),
    "number_of_empty_columns": float(results_dict["tests"][3]["parameters"]["value"]),
}



"""
    Comprehensive Guide and Walkthrough:
    This script is designed to help users analyze the integrity of their datasets by detecting missing values and empty structures.
    It uses the "adult" dataset from OpenML as an example, but users can replace it with their own datasets.
    The script follows these steps:
    1. Load the dataset: Fetches the dataset and loads it into a DataFrame.
    2. Prepare the dataset: Simulates a real-world scenario by adding a duplicate column and renaming an existing column.
    3. Split the dataset: Divides the dataset into reference and production subsets for comparison.
    4. Initialize a test suite: Sets up tests to detect missing values, empty rows, and columns.
    5. Run the test suite: Executes the tests on the reference and production data.
    6. Extract test results: Converts the test results into a dictionary format.
    7. Parse results: Summarizes the test results into a structured dictionary.
    Users can follow these steps to ensure the quality of their datasets and identify potential issues with missing values and empty structures.
"""