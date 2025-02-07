# Importing necessary libraries
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from datetime import datetime
from utils.call_metrics import get_metrics

# Load the iris dataset
iris = load_iris()

# Convert the iris dataset into a DataFrame and add the target label as a new column named 'target'
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target

reference = dataset[:120]   
current = dataset[120:] 

# Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Fit a Logistic Regression model on the reference data
model = LogisticRegression(random_state=42)
model.fit(X_reference, y_reference)

# Get predictions for both reference and current data and add them into respective dataframes
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Generate a classification report containing various metrics
classification_report = Report(metrics=[ClassificationPreset()])
classification_report.run(reference_data=reference, current_data=current)

# Extract the accuracy of the current dataset from the classification report
current_accuracy = classification_report.as_dict()["metrics"][0]["result"]["current"]["accuracy"] * 100

# Calling a function to add to a json file
accuracy_info=get_metrics("accuracy_score",current_accuracy)
"""
    This script performs the following steps to evaluate the accuracy of a logistic regression model on the Iris dataset:

    1. Import necessary libraries and modules.
    2. Load the Iris dataset from sklearn.
    3. Convert the Iris dataset into a pandas DataFrame and add the target labels as a new column named 'target'.
    4. Split the dataset into reference (first 120 samples) and current (remaining samples) datasets.
    5. Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
    6. Fit a Logistic Regression model on the reference data.
    7. Get predictions for both reference and current data and add them into respective dataframes.
    8. Generate a classification report containing various metrics using Evidently AI.
    9. Extract the accuracy of the current dataset from the classification report.
    10. Call a function to add the accuracy information to a JSON file.
"""

