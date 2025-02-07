import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))


from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from utils.call_metrics import get_metrics
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

# Separating the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
X_reference = reference.drop(columns=['target'])
y_reference = reference['target']

X_current = current[X_reference.columns]

# Fitting a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_reference, y_reference)

# Getting the probabilities for both reference and current data and adding them into respective dataframes
reference['prediction'] = model.predict_proba(X_reference)[:, 1] 
current['prediction'] = model.predict_proba(X_current)[:, 1]

# Generating a classification report containing metrics such as accuracy, precision, recall, F1 score, etc.
classification_report = Report(metrics=[ClassificationQualityMetric()])
classification_report.run(reference_data=reference, current_data=current)

# Extracting the False Negative Rate (FNR) for the current dataset
current_fnr = float(classification_report.as_dict()["metrics"][0]["result"]["current"]["fnr"]) * 100
# Calling a function to add to a json file
fnr_value_dict=get_metrics("fnr_value",current_fnr)
'''
Comprehensive Guide and Walkthrough:

1. **Loading the Dataset**:
    - We load the breast cancer dataset from sklearn's datasets module. This dataset is used for demonstration purposes.
    - The dataset is converted to a pandas DataFrame for easier manipulation.

2. **Splitting the Dataset**:
    - The dataset is split into two parts: reference and current.
    - The reference dataset serves as a benchmark or baseline for comparison. It should reflect realistic data patterns and seasonality.
    - The current dataset represents the new production data that you want to evaluate and compare against the reference dataset.

3. **Separating Features and Target Labels**:
    - We separate the features (X_reference and X_current) and the target labels (y_reference) from the reference and current datasets.
    - This separation is necessary for training the model and making predictions.

4. **Fitting the Random Forest Model**:
    - A Random Forest model is fitted using the reference dataset.
    - This model will be used to predict probabilities for both the reference and current datasets.

5. **Adding Predictions to DataFrames**:
    - The predicted probabilities for both reference and current data are added to their respective DataFrames.
    - These probabilities will be used to generate the classification report.

6. **Generating the Classification Report**:
    - A classification report is generated using the evidently library.
    - This report contains various metrics such as accuracy, precision, recall, F1 score, etc.

7. **Calculating the False Negative Rate (FNR)**:
    - The False Negative Rate (FNR) is extracted from the classification report for the current dataset.
    - The FNR is multiplied by 100 to convert it to a percentage.

**Scenarios**:
- **Historical Data Comparison**: Use historical data as the reference dataset to compare with new production data.
- **Real-time Monitoring**: Continuously monitor new production data by comparing it with a stable reference dataset.
- **Model Performance Evaluation**: Evaluate the performance of a model by comparing its predictions on reference and current datasets.

This guide provides a comprehensive walkthrough of the code, explaining each step and its purpose. Customers can use their own data by following the same steps and replacing the dataset loading part with their own data.
'''