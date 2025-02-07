import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from utils.call_metrics import get_metrics
# Load the breast cancer dataset from sklearn
dataset = datasets.load_breast_cancer(as_frame=True)
dataset = dataset.frame

# Split the dataset into reference and current datasets
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

# Separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']
X_current = current[X_reference.columns]

# Fit a Random Forestmodel on the reference data
model = RandomForestClassifier(random_state=42)
model.fit(X_reference, y_reference)

# Get the prediction probabilities for both reference and current data and add them into respective dataframes
reference['prediction'] = model.predict_proba(X_reference)[:, 1]
current['prediction'] = model.predict_proba(X_current)[:, 1]

# Generate a classification report containing metrics such as accuracy, precision, recall, F1 score, etc.
classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Extract the log_loss metric from the classification report for the current data
current_log_loss = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["log_loss"])
# Calling a function to add to a json file
log_loss_dict=get_metrics("log_loss_value",current_log_loss)

# Comprehensive guide and walkthrough of the file:
'''
This script demonstrates how to use the Evidently AI library to evaluate the performance of a classification model on new production data. 
The script uses the breast cancer dataset from sklearn as an example. 

1. **Loading the Dataset**:
    - The breast cancer dataset is loaded using `datasets.load_breast_cancer(as_frame=True)`.
    - The dataset is converted to a pandas DataFrame.

2. **Splitting the Dataset**:
    - The dataset is split into two parts: reference and current.
    - The reference dataset serves as a benchmark or baseline for comparison.
    - The current dataset represents the new production data that you want to evaluate.

3. **Separating Features and Target Labels**:
    - Features (X_reference and X_current) and target labels (y_reference) are separated from the reference and current datasets.

4. **Fitting the Model**:
    - A Random Forest model is fitted on the reference data.

5. **Generating Predictions**:
    - Prediction probabilities for both reference and current data are obtained and added to the respective dataframes.

6. **Generating Classification Report**:
    - A classification report is generated using the Evidently AI library, which includes metrics such as accuracy, precision, recall, F1 score, etc.

7. **Extracting Log Loss**:
    - The log_loss metric for the current data is extracted from the classification report.

**Scenarios**:
- This script can be adapted to use any dataset by replacing the dataset loading part.
- The reference and current datasets can be adjusted based on the specific use case and data availability.
- The model can be changed to any other classification model as required.
- Additional metrics can be added to the classification report by modifying the `metrics` parameter in the `Report` object.
'''