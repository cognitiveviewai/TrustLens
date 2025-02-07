import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from utils.call_metrics import get_metrics
# Step 1: Load the breast cancer dataset
# The dataset is loaded from sklearn's datasets module. It contains features and target labels for breast cancer diagnosis.
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

# Step 3: Separate features and target labels
# X_reference and X_current contain the features, while y_reference contains the target labels for the reference dataset.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Step 4: Fit a Random Forest model
# A Random Forest model is trained using the reference dataset.
model = RandomForestClassifier(random_state=42)
model.fit(X_reference, y_reference)

# Step 5: Get prediction probabilities
# The model's prediction probabilities for both reference and current datasets are added to their respective dataframes.
reference['prediction'] = model.predict_proba(X_reference)[:, 1]
current['prediction'] = model.predict_proba(X_current)[:, 1]

# Step 6: Generate a classification report
# The classification report contains metrics such as accuracy, precision, recall, F1 score, class representation, confusion matrix, and quality metrics by class.
classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Step 7: Extract the ROC AUC score
# The ROC AUC score for the current dataset is extracted from the classification report and converted to a percentage.
current_roc_auc = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["roc_auc"])
# Calling a function to add to a json file
current_roc_auc_dict=get_metrics("roc_auc_score", current_roc_auc)


# Comprehensive Guide and Walkthrough:
''' This script demonstrates how to use a Random Forest model to evaluate the performance of a classification task using the breast cancer dataset.
 The steps are as follows:
 1. Load the dataset: The breast cancer dataset is loaded from sklearn's datasets module.
 2. Split the dataset: The dataset is split into reference and current datasets. The reference dataset serves as a benchmark, while the current dataset represents new production data.
 3. Separate features and target labels: The features and target labels are separated for both reference and current datasets.
 4. Fit the model: A Random Forest model is trained using the reference dataset.
 5. Get prediction probabilities: The model's prediction probabilities for both reference and current datasets are added to their respective dataframes.
 6. Generate a classification report: A classification report is generated using the evidently library, which contains various classification metrics.
 7. Extract the ROC AUC score: The ROC AUC score for the current dataset is extracted from the classification report and converted to a percentage.
 
 This script can be adapted to use your own data by replacing the dataset loading step with your own data loading logic.
 Ensure that your data is in a similar format, with features and target labels, and follow the same steps to evaluate your model's performance.'''