from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn import datasets

# Step 1: Fetch and prepare the dataset
# Load the Adult dataset from OpenML
adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame

# Create a duplicate column, rename 'class' column to 'target', and simulate model predictions by adding a prediction column
adult["dup_col"] = adult["education"]
adult.rename(columns={'class': 'target'}, inplace=True)

# Split the data into reference and production datasets
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

adult_ref = adult[~adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]
adult_prod = adult[adult.education.isin(["Some-college", "HS-grad", "Bachelors"])]

# Step 2: Define scenarios for data drift analysis
# Scenario 1: Using default parameters
data_report = Report(
    metrics=[
        DataDriftPreset(),
    ],
)
data_report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])

report={
    column: float(data_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][column]["drift_score"])
    for column in adult_ref.columns
}

# Scenario 2: Using a single statistical test for the entire dataset

data_report = Report(
    metrics=[
        DataDriftPreset(stattest='psi', stattest_threshold=0.3, drift_share=0.3),
    ],
)
data_report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])

report={
    column: float(data_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][column]["drift_score"])
    for column in adult_ref.columns
}

# Scenario 3: Using custom statistical tests for specific columns

per_column_stattest = {x: 'wasserstein' for x in ['age', 'education-num']}

for column in ['sex', 'target']:
    per_column_stattest[column] = 'z'

for column in ['workclass', 'education']:
    per_column_stattest[column] = 'kl_div'

for column in ['relationship', 'race', 'native-country']:
    per_column_stattest[column] = 'jensenshannon'

for column in ['fnlwgt', 'hours-per-week']:
    per_column_stattest[column] = 'anderson'

for column in ['capital-gain', 'capital-loss']:
    per_column_stattest[column] = 'cramer_von_mises'

data_report = Report(
    metrics=[
        DataDriftPreset(per_column_stattest=per_column_stattest, drift_share=0.3),
    ],
)
data_report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])
report= {
    column: float(data_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][column]["drift_score"])
    for column in adult_ref.columns
}

# Scenario 4: Adjusting drift sensitivity for high-risk features

high_risk_features = ['capital-gain', 'capital-loss', 'hours-per-week']
per_column_stattest = {feature: 'psi' for feature in high_risk_features}

data_report = Report(
    metrics=[
        DataDriftPreset(per_column_stattest=per_column_stattest, drift_share=0.2),
    ],
)
data_report.run(reference_data=adult_ref, current_data=adult_prod.iloc[0:100, :])
report={
    column: float(data_report.as_dict()["metrics"][1]["result"]["drift_by_columns"][column]["drift_score"])
    for column in high_risk_features
}


"""
    This script demonstrates multiple approaches to analyzing data drift using the `evidently` library. The scenarios include:

    Scenario 1: Default parameters
    - Applies default statistical tests to all columns.
    - Suitable for a quick overview of data drift with minimal customization.

    Scenario 2: Single Statistical Test
    - Applies a single statistical test (PSI) to all columns.
    - Suitable for quick, generalized drift detection.
    - **Customer Mapping**: Use this scenario when you need a high-level overview of data drift across the entire dataset without focusing on specific features.

    Scenario 3: Custom Statistical Tests per Column
    - Assigns specific statistical tests to individual columns based on domain knowledge.
    - Ideal for datasets with diverse feature types or specific customer requirements.
    - **Customer Mapping**: Apply this scenario when you know which features require specific statistical tests due to their importance or unique characteristics.
        Example: For categorical features like `education`, use KL Divergence; for numerical features like `age`, use Wasserstein.

    Scenario 4: High-Risk Feature Sensitivity
    - Focuses on high-risk features with lower drift thresholds for heightened sensitivity.
    - Useful for detecting subtle changes in critical business metrics.
    - **Customer Mapping**: Utilize this scenario for features directly tied to business impact or regulatory compliance. For instance, monitor `capital-gain` and `capital-loss` if they influence key financial decisions.

    Key Parameters:
    - `stattest`: Statistical test to use for drift detection (e.g., PSI, Wasserstein, KL Divergence).
    - `stattest_threshold`: Threshold for detecting drift (default 0.5).
    - `drift_share`: Proportion of drifted features to flag overall drift (default 0.5).

    General Guidelines for Customization:
    - **Feature Prioritization**: Identify high-priority features based on business goals or domain knowledge.
    - **Sensitivity Adjustment**: Modify `stattest_threshold` and `drift_share` to control the sensitivity of drift detection.
    - **Scenario Selection**: Choose a scenario based on your dataset's characteristics and the specific use case requirements.

    Example Usage:
    - Run `scenario_single_test()` for a quick overview.
    - Use `scenario_custom_column_tests()` for detailed drift analysis tailored to your data.
    - Apply `scenario_high_risk_features()` to closely monitor critical features with lower tolerance for drift.
"""