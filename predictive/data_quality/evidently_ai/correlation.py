from evidently.report import Report
from evidently.metrics import DatasetCorrelationsMetric
from sklearn import datasets
from evidently.test_suite import TestSuite
from evidently.tests import *
from sklearn.linear_model import LogisticRegression

# Step 1: Load the dataset
# Loading the breast cancer dataset from sklearn
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# Step 2: Split the dataset into reference and current
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

# Step 3: Separate the features and the target labels
# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Step 4: Fit a Logistic Regression model
# Fitting a Logistic Regression model on the reference dataset
model = LogisticRegression(random_state=42)
model.fit(X_reference, y_reference)

# Adding predictions to the reference and current datasets
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Step 5: Detect correlation in the dataset

# Scenario 1: Create a report to detect correlations in the dataset
# Create a correlation report using Evidently AI
correlation = Report(metrics=[
    DatasetCorrelationsMetric()
])
# Run the correlation report on the reference and a subset of the current data
correlation.run(reference_data=reference, current_data=current.iloc[0:100, :])

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
cor_dict = {
    "maximum_pearson_correlation": max_pearson_corr,
    "target_prediction_corr": t_p_cor,
    "target_feature_corr": t_f_cor,
    "prediction_feature_corr": p_f_cor,
    "correlation_change": cor_change
}


"""
    Comprehensive Guide and Walkthrough:
    This script demonstrates how to use the Evidently AI library to detect correlations in a dataset.
    The steps are as follows:
    1. Load the dataset: We use the breast cancer dataset from sklearn for this example.
    2. Split the dataset: We create reference and current datasets to simulate a production environment.
    3. Separate the features and target labels: We prepare the data for model training and evaluation.
    4. Fit a Logistic Regression model: We train a model on the reference dataset and generate predictions.
    5. Detect correlation: We use the Evidently AI library to create a correlation report and test suite.
    6. Extract correlation values: We extract the maximum absolute Pearson correlation value from the report and other correlation values from the test suite.
    7. Store the correlation values: We store the extracted values in a dictionary for further analysis.


    Scenarios:
    - This script can be used as a template for detecting correlations in any dataset.
    - Users can replace the breast cancer dataset with their own dataset.
    - Users can modify the reference and current data splitting criteria based on their own requirements.
"""
