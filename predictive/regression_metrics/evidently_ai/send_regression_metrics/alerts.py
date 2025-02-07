def evaluate_metrics(metrics_dict):
    # Define thresholds for each metric
    thresholds = {
        'r_square_error': {
            'acceptable': (0.75, 1.0),
            'low_risk': (0.61, 0.74),
            'medium_risk': (0.41, 0.6),
            'high_risk': (0.0, 0.4)
        },
        'mean_absolute_error': {
            'acceptable': (0, 10),
            'low_risk': (11, 24),
            'medium_risk': (25, 49),
            'high_risk': (50, float('inf'))
        },
        'mean_absolute_percentage_error': {
            'acceptable': (0, 15),
            'low_risk': (16, 29),
            'medium_risk': (30, 49),
            'high_risk': (50, 100)
        },
            'mean_error': {
        'suspicious': 0.0,  # Perfect mean error may indicate overfitting
        'acceptable': (0.0, 0.1),       # 0-10%
        'low_risk': (0.11, 0.24),       # 11-24%
        'medium_risk': (0.25, 0.49),    # 25-49%
        'high_risk': (0.5, 1.0)         # 50-100%
    },
    'absolute_maximum_error': {
        'suspicious': 0.0,  # All predictions perfect is suspicious
        'acceptable': (0.0, 0.1),       # 0-10%
        'low_risk': (0.11, 0.24),       # 11-24%
        'medium_risk': (0.25, 0.49),    # 25-49%
        'high_risk': (0.5, 1.0)         # 50-100%
    },
    'root_mean_squared_error': {
        'suspicious': 0.0,  # Perfect RMSE is suspicious
        'acceptable': (0.0, 0.15),      # 0-15%
        'low_risk': (0.16, 0.29),       # 16-29%
        'medium_risk': (0.3, 0.49),     # 30-49%
        'high_risk': (0.5, 1.0)         # 50-100%
    },
    'std_dev_error': {
        'suspicious': 0.0,  # No variability in errors is suspicious
        'acceptable': (0.0, 0.1),       # 0-10%
        'low_risk': (0.11, 0.24),       # 11-24%
        'medium_risk': (0.25, 0.49),    # 25-49%
        'high_risk': (0.5, 1.0)         # 50-100%
    }
    }
    
    def get_risk_level(metric_name, value):
        if metric_name not in thresholds:
            return 'UNKNOWN'
        
        ranges = thresholds[metric_name]
        if ranges['acceptable'][0] <= value <= ranges['acceptable'][1]:
            return 'ACCEPTABLE'
        elif ranges['low_risk'][0] <= value <= ranges['low_risk'][1]:
            return 'LOW RISK'
        elif ranges['medium_risk'][0] <= value <= ranges['medium_risk'][1]:
            return 'MEDIUM RISK'
        elif value >= ranges['high_risk'][0]:
            return 'HIGH RISK'
        
        return 'UNKNOWN'
    
    def print_alert(metric_name, value, risk_level):
        alert = f"ALERT - {metric_name}: {value:.2f} - Risk Level: {risk_level}"
        msg = f"MESSAGE - {metric_name}: {value:.2f} - works fine and is {risk_level}"
        
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
        risk_level = get_risk_level(metric, value)
        print_alert(metric, value, risk_level)

"""
    Evaluates regression metrics against predefined thresholds and prints alerts based on risk levels.
    Parameters:
    metrics_dict (dict): A dictionary where keys are metric names and values are the corresponding metric values.
    The function performs the following steps:
    1. Defines thresholds for each metric, categorizing them into 'acceptable', 'low risk', 'medium risk', and 'high risk' levels.
    2. Defines a helper function `get_risk_level` to determine the risk level of a given metric value based on the predefined thresholds.
    3. Defines a helper function `print_alert` to print an alert message with appropriate risk level indicators.
    4. Iterates over each metric in `metrics_dict`, evaluates its risk level using `get_risk_level`, and prints the corresponding alert using `print_alert`.
    Risk Levels:
    - ACCEPTABLE: Metric value falls within the acceptable range.
    - LOW RISK: Metric value falls within the low risk range.
    - MEDIUM RISK: Metric value falls within the medium risk range.
    - HIGH RISK: Metric value falls within the high risk range.
    - UNKNOWN: Metric name is not recognized or value does not fall within any defined range.
    Example:
    metrics_dict = {
        'r_square_error': 0.65,
        'mean_absolute_error': 15,
        'mean_absolute_percentage_error': 20,
        'mean_error': 0.05,
        'absolute_maximum_error': 0.2,
        'root_mean_squared_error': 0.1,
        'std_dev_error': 0.3
    evaluate_metrics(metrics_dict)
"""