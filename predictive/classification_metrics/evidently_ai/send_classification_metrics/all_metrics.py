import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))


from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from evidently.metric_preset import ClassificationPreset
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
# loading the breast cancer dataset
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# splitting the dataset into reference and current
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
# Extract the precision of the current dataset from the classification report
current_precision = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["precision"]) * 100
# Extract the f1_score of the current dataset from the classification report
current_f1 = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["f1"])
# Extract the recall of the current dataset from the classification report
current_recall = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["recall"])

classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Extracting the False Negative Rate (FNR) for the current dataset
current_fnr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["fnr"]) * 100

# Extracting the False Positive Rate (FPR) for the current dataset
current_fpr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["fpr"])


# Extract the True Negative Rate (TNR) from the classification report
current_tnr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["tnr"])
 
# Extract the True Poitive Rate (TPR) from the classification report
current_tpr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["tpr"])

# Get the prediction proababilities
reference['prediction'] = model.predict_proba(X_reference)[:, 1]
current['prediction'] = model.predict_proba(X_current)[:, 1]

# Get the AUC ROC score
current_roc_auc = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["roc_auc"])

#Get the Log Loss value
current_log_loss = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["log_loss"])


# Create a metrics dictionary
metrics_dict = {
    "accuracy_score": current_accuracy,
    "precision_score": current_precision,
    "recall_score": current_recall,
    "f1_score": current_f1,
    "tpr_value": current_tpr,
    "fpr_value": current_fpr,
    "tnr_value": current_tnr,
    "fnr_value": current_fnr,
    "log_loss_value": current_log_loss,
    "roc_auc_score": current_roc_auc
}

def get_classification_metrics():
    return metrics_dict

"""
    Retrieve the dictionary containing classification metrics.

    Returns:
        dict: A dictionary where keys are metric names and values are their corresponding values.

    Summary:
        This function returns a dictionary that holds various classification metrics.
"""