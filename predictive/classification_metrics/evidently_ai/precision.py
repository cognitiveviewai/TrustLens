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

# Generating a classification report containing metrics such as accuracy, precision, recall, F1 score, class representation, confusion matrix, and quality metrics by class
classification_report = Report(metrics=[ClassificationPreset()])
classification_report.run(reference_data=reference, current_data=current)

# Extracting the precision metric for the current dataset
current_precision = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["precision"]) * 100
# Calling a function to add to a json file
current_precision_dict=get_metrics("precision_score",current_precision)

# Comprehensive guide and walkthrough of the file:
'''
Step-by-Step Guide:

1. Import necessary libraries:
    - pandas and numpy for data manipulation.
    - evidently for generating classification reports.
    - sklearn for machine learning models and datasets.

2. Load the iris dataset:
    - Convert it into a pandas DataFrame.
    - Add the target labels as a new column named 'target'.

3. Split the dataset into reference and current datasets:
    - The reference dataset serves as a benchmark or baseline for comparison.
    - The current dataset represents the new production data to be evaluated.

4. Separate features and target labels:
    - Extract features (X_reference and X_current) and target labels (y_reference) from the reference and current datasets.

5. Fit a Random Forest model:
    - Train the model using the reference dataset.

6. Generate predictions:
    - Get predictions for both reference and current datasets.
    - Add the predictions to the respective DataFrames.

7. Generate a classification report:
    - Use evidently to create a report containing various classification metrics.

8. Extract the precision metric:
    - Retrieve the precision value for the current dataset from the report and convert it to a percentage.

Scenarios:
- This script can be used to monitor the performance of a machine learning model in production by comparing new data (current dataset) against historical data (reference dataset).
- It helps in identifying any drifts or changes in the model's performance over time.
- Users can replace the iris dataset with their own data to evaluate their specific use case.
'''