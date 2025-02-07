# Importing necessary libraries
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from utils.call_metrics import get_metrics
# Loading the iris dataset, converting it into a DataFrame, and adding the target label as a new column named 'target'
iris = load_iris()
dataset = pd.DataFrame(data=iris.data, columns=iris.feature_names)
dataset['target'] = iris.target

# Splitting the dataset into reference and current datasets
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
reference = dataset[:120]   
current = dataset[120:] 

# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Fitting a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_reference, y_reference)

# Getting the predictions for both reference and current data and adding them into respective dataframes
reference['prediction'] = model.predict(X_reference)
current['prediction'] = model.predict(X_current)

# Generating a classification report containing various metrics such as accuracy, precision, recall, F1 score, class representation, confusion matrix, and quality metrics by class
classification_report = Report(metrics=[ClassificationPreset()])
classification_report.run(reference_data=reference, current_data=current)

# Extracting the recall metric from the classification report for the current dataset
current_recall = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["recall"])
# Calling a function to add to a json file
current_recall_dict=get_metrics("recall_score", current_recall)

# Comprehensive guide and walkthrough of the file:
'''
Steps:
1. Import necessary libraries: pandas, numpy, evidently, sklearn.
2. Load the iris dataset using sklearn's load_iris function.
3. Convert the dataset into a pandas DataFrame and add the target labels as a new column named 'target'.
4. Split the dataset into reference and current datasets. The reference dataset serves as a benchmark, while the current dataset represents new production data.
5. Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
6. Fit a Random Forest model on the reference data.
7. Get predictions for both reference and current data and add them into respective dataframes.
8. Generate a classification report using evidently's Report class, which includes various metrics such as accuracy, precision, recall, F1 score, class representation, confusion matrix, and quality metrics by class.
9. Extract the recall metric from the classification report for the current dataset.

Scenarios:
- This guide can be used by customers to evaluate the performance of their classification models on new production data.
- The reference dataset can be historical data from a previous production period or a stable dataset that reflects realistic data patterns.
- The current dataset represents the new production data that needs to be monitored and compared against the reference dataset.
- The classification report provides a comprehensive set of metrics to evaluate the model's performance and identify any potential issues with the new production data.
'''
