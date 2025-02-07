from utils import generate_performance_dict
from datetime import datetime
from all_metrics import get_classification_metrics

def fetch_model_performance() -> dict:
    try:
        data=get_classification_metrics()
        performance = generate_performance_dict(client_id="client_secret", metric_type="classification", created_at=datetime.utcnow().isoformat() + "Z", metric_id="string", name="string", type="string", description="string", source="string", data=data, collected_on="string", collected_by="string", authorized_user="string", test_mode=False)
        return performance
    except Exception as e:
        raise RuntimeError(f"Error fetching model performance: {str(e)}")
    
"""
        Fetches the model performance metrics based on the specified metric type.

        Returns:
            dict: A dictionary containing the performance metrics.

        Raises:
            RuntimeError: If there is an error fetching the model performance.

        Steps:
            1. Retrieve classification metrics data by calling `get_classification_metrics()`.
            2. Check the value of `metric_type`:
                - If `metric_type` is "regression":
                    - Generate a performance dictionary for regression metrics using `generate_performance_dict` with appropriate parameters.
                - Otherwise:
                    - Generate a performance dictionary for classification metrics using `generate_performance_dict` with appropriate parameters.
            3. Return the generated performance dictionary.
            4. If any exception occurs during the process, raise a `RuntimeError` with the error message."""