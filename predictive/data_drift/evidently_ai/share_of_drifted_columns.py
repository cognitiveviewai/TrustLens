from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns
import re
from sklearn import datasets

# Step 1: Fetch the dataset
# Here we use the 'adult' dataset from OpenML. This dataset is commonly used for classification tasks.
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# Step 2: Prepare the dataset
# Create a duplicate column for demonstration purposes and rename the 'class' column to 'target'.
adult["dup_col"] = adult["education"]
adult.rename(columns={'class': 'target'}, inplace=True)

# Step 3: Split the dataset into reference and production data
# Reference data will exclude certain education levels, while production data will include them.
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

# Step 4: Define the data drift test suite
# We use the Population Stability Index (PSI) as the statistical test with a threshold of 0.5.
data_drift_dataset_tests = TestSuite(tests=[
    TestShareOfDriftedColumns(stattest="psi", lt=0.5),
])

# Step 5: Run the data drift test
# We run the test on the first 100 rows of the production data against the reference data.
data_drift_dataset_tests.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])

# Step 6: Extract the share of drifted columns
# We use a regular expression to extract the percentage of drifted columns from the test description.
match = re.search(r'\b\d+(?:\.\d+)?\b', data_drift_dataset_tests.as_dict()["tests"][0]["description"])
data_drift_percentage = float(match.group())



"""
    Comprehensive Guide and Walkthrough:
    This script demonstrates how to use the Evidently AI library to detect data drift in a dataset.

    Steps:
    1. Fetch the dataset: We use the 'adult' dataset from OpenML, which is a common dataset for classification tasks.
    2. Prepare the dataset: We create a duplicate column and rename the 'class' column to 'target' to simulate a real-world scenario.
    3. Split the dataset: We divide the dataset into reference and production data based on certain education levels.
    4. Define the data drift test suite: We use the Population Stability Index (PSI) as the statistical test with a threshold of 0.5.
    5. Run the data drift test: We run the test on the first 100 rows of the production data against the reference data.
    6. Extract the share of drifted columns: We extract the percentage of drifted columns from the test description using a regular expression.

    This script can be adapted to use any dataset by replacing the dataset fetching and preparation steps.
    The data drift test can be customized by changing the statistical test and threshold in the TestSuite definition.
"""
