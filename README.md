# About
This repository is a comprehensive guide to call Evidence API.
It contains examples and scripts for generating metrics for predictive, and generative AI using observability frameworks like Evidently AI, Fiddler AI, and Whylabs AI. It includes tools for classification and regression metrics, data integrity, explainability, and security in machine learning models. 

# Predictive Metrics
This repository contains examples for generating different predictive metrics using Evidently AI. 
It is organized into several folders focusing on different aspects of predictive modeling, including classification metrics, regression metrics, data integrity, data drift, and explainability.

### 1. **classification_metrics**
This folder contains sub folders containing Python scripts and to compute metrics used for evaluating classification models using Evidently AI. This folder is divided into the following sub folders:

- **evidently_ai**: Scripts leveraging Evidently AI for generating multiple classification metrics. 

#### 1.1 **Evidently AI classification metrics:**
- `accuracy.py`: Calculates the accuracy of classification models.
- `false_negative_rate.py`: Computes the false negative rate metric.
- `false_positive_rate.py`: Computes the false positive rate metric.
- `log_loss.py`: Computes log loss metric.
- `precision.py`: Computes the precision metric.
- `recall.py`: Calculates the recall metric.
- `f1_score.py`: Calculates the f1_score metric.
- `roc_auc.py`: Calculates the ROC-AUC score.
- `true_negative_rate.py`: Calculates the true negative rate metric.
- `true_positive_rate.py`: Calculates the true positive rate metric.

#### 1.1.1 **Evidently AI send_classification_metrics:**
- `alerts.py`: Evaluates each metric against preset risk thresholds and prints color-coded alerts indicating its performance risk level.
- `all_metrics.py`: Loads a demo data, trains a logistic regression model on a reference set, computes classification metrics with Evidently for both reference and current data, and returns these metrics in a dictionary.
- `model_metrics.py`: Fetches classification metrics from `all_metrics.py` and packages them into a performance dictionary using a helper function, handling any errors that arise.
- `send_metrics.py`: Retrieves model performance metrics, sends them via a POST request to an API, and then evaluates the simplified metrics with alert thresholds.


### 2. **data_integrity**
This folder contains sub folders containing scripts designed for ensuring data quality and performing validation in machine learning workflows using Evidently AI. This folder contains the following sub folders:

- **evidently_ai**: Scripts specific to Evidently AI for data monitoring and data integrity validation.

#### 2.1 **Evidently AI data integrity:**
- `missing_values.py`: Loads and prepares a demo dataset, runs missing value and empty structure tests on reference versus production subsets using Evidently, and summarizes the test results in a dictionary.
- `out_of_range_values.py`: Loads and prepares a demo dataset, splits it into reference and production subsets, runs an Evidently test to check for out-of-range "age" values (0–100), and extracts the resulting share.

### 3. **explainability**
This folder contains sub folders containing scripts that provide insights into the workings of machine learning models using Evidently AI, Fiddler AI and Whylabs AI. This folder is divided into the following sub folders:

- **evidently_ai**: Model monitoring and interpretability tools specific to Evidently AI.

#### 3.1 **Evidently AI explainability metrics:**
- `feature_importances.py`: It computes the feature importances and provides a dictionary containing the feature importance per column with the column name as key and the feature importance value as the value.

### 4. **regression_metrics**
It contains python scripts and tools to compute metrics used for evaluating regression models provided by Evidently AI, Fiddler AI and Whylabs AI. This folder is divided into the following sub folders:

- **evidently_ai**: Scripts leveraging Evidently AI for generating multiple regression metrics. 

#### 4.1 **Evidently AI regression metrics:**
- `mean_absolute_error.py`: Calculates the Mean Absolute Error (MAE) for regression models.
- `mean_absolute_percentage_error.py`: Calculates the Mean Absolute Percentage Error (MAPE) for regression models.
- `mean_error.py`: Calculates the mean error for regression models.
- `r_squared_score.py`: Calculates the R2 score for regression models.
- `regression.py`:  An aggregation of all the metrics and tests provided by Evidently AI for regression.
- `root_mean_squared_error.py`:  Calculates the root mean squared error (RMSE) for regression models.
- `standard_deviation_error.py`:  Calculates the standard deviation of error for regression models.
- `absolute_maximum_error.py`:  Calculates the absolute maximum error and compares it to the reference or against a defined condition for regression models.

### Alert Thresholds

| Indicator | Acceptable Range | Not Acceptable | Low Risk | Medium Risk | High Risk |
|-----------|-----------------|----------------|-----------|-------------|------------|
| Data Drift | 0.0-0.7 | 0.7-1.0 for 7 days | 0.7-0.79 | 0.8-0.89 | 0.9-1.0 |
| Target drift | 0-20% | 21-100% | 21-34% | 35-49% | 50-100% |
| Missing values | 0-5% | 6-100% | 6-19% | 20-49% | 50-100% |
| Bias Detection | 0-20% | 21-100% | 21-34% | 35-49% | 50-100% |
| Out of range values | 0-2% | 3-100% | 3-19% | 20-49% | 50-100% |
| Outlier Detection | 0-5% | 6-100% | 6-19% | 20-49% | 50-100% |
| Duplicated Rows/Columns | 0-1% | 1-100% | 1-9% | 10-19% | 20-100% |
| Class balance | <4:1 | >4:1 | 4.1-5:1 | 6:1-10:1 | >10:1 |
| Accuracy | 85-100 | 0-84 | 71-84% | 51-70% | 0-50% |
| Precision | 80-100 | 0-79 | 66-79% | 51-65% | 0-50% |
| Recall | 75-100 | 0-74 | 66-74% | 51-65% | 0-50% |
| F1-Score | 75-100 | 0-74 | 66-74% | 51-65% | 0-50% |
| Log loss value | 0-0.5 | 0.6-1.0 | 0.6-0.69 | 0.7-0.79 | 0.8-1.0 |
| ROC AUC Score | 0.75-1.0 | 0.0-0.76 | 0.66-0.76 | 0.51-0.65 | 0.0-0.5 |
| Classification Quality by Feature | 0-10% | 11-100% | 11-24% | 25-49% | 50-100% |
| Classification Quality by Class | 0-10% | 11-100% |  11-24% | 25-49% | 50-100% |
| R squared error | 0.75-1.0 | 0.0-0.74 | 0.61-0.74 | 0.41-0.6 | 0.0-0.4 |
| MAE (Mean Absolute Error) | 0-10% | 11-100% | 11-24% | 25-49% | 50-100% |
| MAPE (Mean Absolute Percentage Error) | 0-15% | 16-100% | 16-29% | 30-49% | 50-100% |

#### Risk Level Descriptions

- **Acceptable Range**: Normal operating conditions, no action required
- **Not Acceptable**: Values that fall outside acceptable parameters
- **Low Risk**: Monitor closely, potential issues developing
- **Medium Risk**: Requires attention and investigation
- **High Risk**: Immediate action required


### Getting Started

### Prerequisites
- Python 3.12
- Libraries: Install dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
### Usage
Navigate to the folder of interest and use the respective scripts for your use case. All the datasets used in the files are used from `scikit-learn` for simplicity and hence running the code would fetech the data itself. You can take reference of those scripts, and use it by modifying for your own use-case. 

```bash
python copilot-connector/regression_metrics/evidently_ai/mean_absolute_error.py  
```

# Generative Metrics
This repository contains examples for generating different generative metrics using Evidently AI, Fiddler AI, and Whylabs AI. 
It is organized into several folders focusing on different aspects of generative modeling, including harmful content detection, privacy, reliability, and response relevance.

### 1. **harmful_content**
This folder contains subfolders with python scripts to compute metrics used for evaluating harmful content in generative models using Evidently AI. This folder is divided into the following subfolders:

- **evidently_ai**: Scripts leveraging Evidently AI for detecting harmful content.

#### 1.1 **Evidently AI harmful content metrics:**
- `biased_content.py`: Detects biased content in generative outputs.
- `negative_content.py`: Detects negative content in generative outputs.
- `toxic_content.py`: Detects toxic content in generative outputs.

### 2. **privacy**
This folder contains subfolders with scripts designed for ensuring privacy in generative models using Evidently AI. This folder contains the following subfolders:

- **evidently_ai**: Scripts specific to Evidently AI for privacy detection.

#### 2.1 **Evidently AI privacy metrics:**
- `detect_pii.py`: Detects personally identifiable information (PII) in generative outputs.

### 3. **reliability**
This folder contains subfolders with scripts that provide insights into the reliability of generative models using Evidently AI. This folder is divided into the following subfolders:

- **evidently_ai**: Reliability tools specific to Evidently AI.

#### 3.1 **Evidently AI reliability metrics:**
- `context_relevance.py`: Evaluates the context relevance of generative outputs.

### 4. **response_relevance**
This folder contains python scripts and tools to compute metrics used for evaluating the relevance of responses generated by generative models provided by Evidently AI. This folder is divided into the following subfolders:

- **evidently_ai**: Scripts leveraging Evidently AI for generating response relevance metrics.

#### 4.1 **Evidently AI response relevance metrics:**
- `alignment_score.py`: Calculates the alignment score of generative responses.
- `is_declined.py`: Checks if the response is declined.

## Getting Started

### Prerequisites
- Python 3.12
- Libraries: Install dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
### Usage
Navigate to the folder of interest and use the respective scripts for your use case. All the datasets used in the files are used from `scikit-learn` for simplicity and hence running the code would fetch the data itself. You can take reference of those scripts, and use it by modifying for your own use-case. 

```bash
python copilot-connector/generative_metrics/evidently_ai/biased_content.py  
```
<br>

# How to send evidence through evidence API ?
You can do a **POST** request on the **endpoint** with the different metrics you want to send in the supported schema. One example playload containing all the metrics we support can be found in `send_evidence.py` inside **utils** folder, and it also contains the snippet of how to do a **POST** request using **requests** python library.   

