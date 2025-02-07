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

# Extracting the f1 score from the classification report for the current dataset
current_f1 = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["f1"])

# Calling a function to add to a json file
f1_score_dict=get_metrics("f1_score",current_f1)


"""
    This script performs the following steps:

    1. Imports necessary libraries and modules:
        - `sys` and `Path` from `pathlib` to manipulate system paths.
        - `pandas` for data manipulation.
        - `Report` and `ClassificationPreset` from `evidently` for generating classification reports.
        - `RandomForestClassifier` from `sklearn.ensemble` for building the classification model.
        - `load_iris` from `sklearn.datasets` to load the iris dataset.
        - `get_metrics` from `utils.call_metrics` to handle metric extraction.

    2. Loads the iris dataset, converts it into a DataFrame, and adds the target label as a new column named 'target'.

    3. Splits the dataset into reference and current datasets:
        - `reference` dataset contains the first 120 samples.
        - `current` dataset contains the remaining samples.

    4. Separates the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.

    5. Fits a Random Forest model using the reference dataset.

    6. Gets the predictions for both reference and current data and adds them into respective dataframes.

    7. Generates a classification report containing various metrics such as accuracy, precision, recall, F1 score, class representation, confusion matrix, and quality metrics by class using the `evidently` library.

    8. Extracts the F1 score from the classification report for the current dataset.

    9. Calls a function to add the F1 score to a JSON file.
"""