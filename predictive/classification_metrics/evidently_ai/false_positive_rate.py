import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from utils.call_metrics import get_metrics

# Load the breast cancer dataset from sklearn
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# Split the dataset into reference and current datasets
# The reference dataset serves as a benchmark or baseline for comparison.
# The current dataset represents the new production data that you want to evaluate and compare against the reference dataset.
reference = dataset.sample(n=500, replace=False)
current = dataset.sample(n=200, replace=False)

# Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Fit a Logistic Regression model on the reference data
model = LogisticRegression(random_state=42)
model.fit(X_reference, y_reference)

# Get the prediction probabilities for both reference and current data and add them into respective dataframes
reference['prediction'] = model.predict_proba(X_reference)[:, 1]
current['prediction'] = model.predict_proba(X_current)[:, 1]

# Generate a classification report containing metrics such as accuracy, precision, recall, F1 score, etc.
classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Extract the False Positive Rate (FPR) from the classification report for the current data
current_fpr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["fpr"])

# Calling a function to add to a json file
current_fpr_dict=get_metrics("fpr_value",current_fpr)


"""
Comprehensive Guide and Walkthrough:

1. Import necessary libraries:
    - pandas for data manipulation
    - evidently for generating classification reports
    - sklearn for machine learning models and datasets

2. Load the breast cancer dataset from sklearn:
    - The dataset is loaded as a pandas DataFrame.

3. Split the dataset into reference and current datasets:
    - The reference dataset serves as a benchmark or baseline for comparison.
    - The current dataset represents the new production data that you want to evaluate and compare against the reference dataset.

4. Separate the features and target labels:
    - X_reference and y_reference are the features and target labels for the reference dataset.
    - X_current contains the features for the current dataset.

5. Fit a Logistic Regression model on the reference data:
    - The model is trained using the reference dataset.

6. Get the prediction probabilities for both reference and current data:
    - The prediction probabilities are added to the respective dataframes.

7. Generate a classification report:
    - The report contains metrics such as accuracy, precision, recall, F1 score, etc.
    - The report is generated using the evidently library.

8. Extract the False Positive Rate (FPR) for the current data:
    - The FPR is extracted from the classification report.

9. Print the False Positive Rate (FPR):
    - The FPR for the current data is printed to the console.

Scenarios:
- This guide can be used to evaluate the performance of a machine learning model on new production data.
- The reference dataset can be historical data, and the current dataset can be new data that needs to be monitored.
- The False Positive Rate (FPR) is an important metric to monitor, especially in scenarios where false positives can have significant consequences (e.g., medical diagnosis, fraud detection).
"""