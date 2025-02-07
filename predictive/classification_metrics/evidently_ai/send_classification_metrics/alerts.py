def evaluate_metrics(metrics_dict):
    # Define thresholds for each metric normalized to 0-1 scale
    thresholds = {
        'accuracy_score': {
            'suspicious': 1.0,  # Nearly perfect accuracy is suspicious
            'acceptable': (0.85, 0.99),
            'low_risk': (0.71, 0.84),
            'medium_risk': (0.51, 0.70),
            'high_risk': (0.0, 0.50)
        },
        'precision_score': {
            'suspicious': 1.0,
            'acceptable': (0.80, 0.99),
            'low_risk': (0.66, 0.79),
            'medium_risk': (0.51, 0.65),
            'high_risk': (0.0, 0.50)
        },
        'recall_score': {
            'suspicious': 1.0,
            'acceptable': (0.75, 0.99),
            'low_risk': (0.66, 0.74),
            'medium_risk': (0.51, 0.65),
            'high_risk': (0.0, 0.50)
        },
        'f1_score': {
            'suspicious': 1.0,
            'acceptable': (0.75, 0.99),
            'low_risk': (0.66, 0.74),
            'medium_risk': (0.51, 0.65),
            'high_risk': (0.0, 0.50)
        },
        'tpr_value': {  # True Positive Rate (same as recall)
            'suspicious': 0.0,
            'acceptable': (0.85, 0.99),
            'low_risk': (0.71, 0.84),
            'medium_risk': (0.51, 0.70),
            'high_risk': (0.0, 0.50)
        },
        'fpr_value': {  # False Positive Rate (lower is better)
            'suspicious': 0.0,  # Suspiciously low FPR
            'acceptable': (0.001, 0.15),
            'low_risk': (0.16, 0.29),
            'medium_risk': (0.30, 0.49),
            'high_risk': (0.50, 1.0)
        },
        'tnr_value': {  # True Negative Rate (Specificity)
            'suspicious': 1.0,
            'acceptable': (0.85, 0.99),
            'low_risk': (0.71, 0.84),
            'medium_risk': (0.51, 0.70),
            'high_risk': (0.0, 0.50)
        },
        'fnr_value': {  # False Negative Rate (lower is better)
            'suspicious': 0.0,  # Suspiciously low FNR
            'acceptable': (0.001, 0.15),
            'low_risk': (0.16, 0.29),
            'medium_risk': (0.30, 0.49),
            'high_risk': (0.50, 1.0)
        },
        'log_loss_value': {  # Lower is better
            'suspicious': 0.0,  # Suspiciously low loss
            'acceptable': (0.01, 0.5),
            'low_risk': (0.6, 0.69),
            'medium_risk': (0.7, 0.79),
            'high_risk': (0.8, 1.0)
        },
        'roc_auc_score': {
            'suspicious': 1.0,
            'acceptable': (0.75, 0.99),
            'low_risk': (0.66, 0.76),
            'medium_risk': (0.51, 0.65),
            'high_risk': (0.0, 0.5)
        }
    }
    
    def get_risk_level(metric_name, value):
        ranges = thresholds[metric_name]
        
        if ranges['acceptable'][0] <= value <= ranges['acceptable'][1]:
            return 'ACCEPTABLE'
        elif ranges['low_risk'][0] <= value <= ranges['low_risk'][1]:
            return 'LOW RISK'
        elif ranges['medium_risk'][0] <= value <= ranges['medium_risk'][1]:
            return 'MEDIUM RISK'
        else:
            return 'HIGH RISK'
    
    def print_alert(metric_name, value, risk_level, prefix=''):
        alert = f"ALERT - {prefix}{metric_name}: {value:.2f} - Risk Level: {risk_level}"
        msg = f"MESSAGE - {prefix}{metric_name}: {value:.2f} - works fine and is {risk_level}"
        if risk_level == 'HIGH RISK':
            alert = "🔴 " + alert
        elif risk_level == 'MEDIUM RISK':
            alert = "🟡 " + alert
        elif risk_level == 'LOW RISK':
            alert = "🟢 " + alert
        else:
            alert = "✅ " + msg
            
        print(alert)
    
    # Evaluate each metric
    for metric, value in metrics_dict.items():
        metric_lower = metric.lower()
        
        # Handle dictionary metrics (precision, recall, f1_score)
        if isinstance(value, dict):
            print(f"\n=== {metric} Breakdown ===")
            
            for key, score_value in value.items():
                # Extract the type of score (class or averaging method)
                if 'class' in key:
                    # Handle cases like 'score_class_0' or 'class_0'
                    class_num = key.split('_')[-1]
                    prefix = f"class_{class_num} - "
                elif 'weighted' in key or 'macro' in key:
                    # Handle cases like 'score_weighted' or 'weighted'
                    avg_method = key.split('_')[-1] if 'score_' in key else key
                    prefix = f"{avg_method} - "
                else:
                    prefix = f"{key} - "
                
                risk_level = get_risk_level(metric_lower, score_value)
                print_alert(metric, score_value, risk_level, prefix)
        
        # Handle single value metrics
        else:
            risk_level = get_risk_level(metric_lower, value)
            print_alert(metric, value, risk_level)

"""
    Evaluate and print risk levels for various metrics based on predefined thresholds.
    Args:
        metrics_dict (dict): A dictionary where keys are metric names and values are their corresponding scores.
                             Values can be either single float values or dictionaries containing multiple scores.
    The function performs the following steps:
    1. Define thresholds for each metric, normalized to a 0-1 scale.
    2. Define a helper function `get_risk_level` to determine the risk level based on the metric value.
    3. Define a helper function `print_alert` to print an alert message based on the risk level.
    4. Iterate through each metric in `metrics_dict`:
       - If the metric value is a dictionary, print a breakdown of the scores.
       - For each score, determine the risk level using `get_risk_level` and print an alert using `print_alert`.
       - If the metric value is a single float, determine the risk level and print an alert.
    Risk Levels:
        - ACCEPTABLE: Metric value falls within the acceptable range.
        - LOW RISK: Metric value falls within the low-risk range.
        - MEDIUM RISK: Metric value falls within the medium-risk range.
        - HIGH RISK: Metric value falls within the high-risk range.
    Alerts:
        - ✅ MESSAGE: Metric value is acceptable and works fine.
        - 🟢 ALERT: Metric value is low risk.
        - 🟡 ALERT: Metric value is medium risk.
        - 🔴 ALERT: Metric value is high risk.
    Summary:
        This function helps in evaluating the performance of a model by categorizing various metrics into risk levels
        and printing appropriate alerts based on predefined thresholds.
"""
    