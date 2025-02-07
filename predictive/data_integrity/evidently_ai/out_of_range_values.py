from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfOutRangeValues
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

# Step 4: Initialize a test suite for detecting Out of Range Values.
# Pass the column name and the range of values to check for out-of-range values. If the range is not passed, values will be calculated from the reference datset
share_of_out_of_range_values = TestSuite(tests=[
    TestShareOfOutRangeValues(column_name="age", left=0, right=100) # optional: left and right  Compulsory: column_name
])

# Step 5: Run the test suite on the reference and production data
# Here, we are using the first 100 rows of the production data for the test.
share_of_out_of_range_values.run(reference_data=adult_ref, current_data=adult_prod.iloc[:100, :])

# Extracting the results from the test suite
out_of_range_share = float(share_of_out_of_range_values.as_dict()["tests"][0]["parameters"]["value"])


"""
    Guide and Walkthrough:

    This script shows how to use the Evidently AI library to detect out-of-range values in a dataset. The script is organized into several steps for clarity.

    Step 1: Load the dataset
    - We use the "adult" dataset from OpenML, which contains demographic information and is commonly used for machine learning tasks.

    Step 2: Prepare the dataset
    - We simulate a real-world scenario by adding a duplicate column and renaming an existing column for better clarity.

    Step 3: Split the dataset into reference and production subsets
    - The reference data is a clean subset of the dataset, excluding specific education levels.
    - The production data includes specific education levels, simulating incoming data that needs validation.


    Step 4: Detect share of Out of Range Values
    - We initialize a test suite to detect the share of out-of-range values in the "age" column, with the range 0-100.

    Step 5: Run the test suite on the reference and production data
    - We run the test suites on the first 100 rows of the production data.
    - The results are extracted and printed, showing the number and share of out-of-range values.

    This script can be adapted to use other datasets by modifying the dataset loading and preparation steps accordingly.
"""
