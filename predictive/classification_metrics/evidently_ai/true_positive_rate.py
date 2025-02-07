import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from utils.call_metrics import get_metrics

# Step 1: Load the breast cancer dataset
# This dataset is used for demonstration purposes. Replace this with your own dataset.
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# Step 2: Split the dataset into reference and current datasets
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

# Step 3: Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Step 4: Fit a Logistic Regression model
# This model is used for demonstration purposes. Replace this with your own model.

model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_reference, y_reference)

# Step 5: Get the predictions for both reference and current data and add them into respective dataframes
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Step 6: Generate a classification report containing the metrics accuracy, precision, recall, F1 score, etc.
classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Step 7: Extract the True Positive Rate (TPR) from the classification report
current_tpr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["tpr"])
# Calling a function to add to a json file
current_tpr_dict=get_metrics("tpr_value", current_tpr)


"""
This script demonstrates how to evaluate the performance of a classification model using the Evidently AI library.

Steps:
1. Load the breast cancer dataset from sklearn. This dataset is used for demonstration purposes. Replace this with your own dataset.
2. Split the dataset into reference and current datasets. The reference dataset serves as a benchmark or baseline for comparison, while the current dataset represents the new production data that you want to evaluate.
3. Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
4. Fit a Random Forest model using the reference dataset. This model is used for demonstration purposes. Replace this with your own model.
5. Get the predictions for both reference and current data and add them into respective dataframes.
6. Generate a classification report containing various metrics such as accuracy, precision, recall, F1 score, etc., using the Evidently AI library.
7. Extract the True Positive Rate (TPR) from the classification report

Note: This script uses the breast cancer dataset and a Random Forest model for demonstration purposes. You should replace these with your own dataset and model to evaluate your specific use case.
"""