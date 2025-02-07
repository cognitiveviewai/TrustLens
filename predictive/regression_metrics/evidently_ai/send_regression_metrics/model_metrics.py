from utils import generate_performance_dict
from datetime import datetime
from all_metrics import get_regression_results

def fetch_regression_results() -> dict:
    try:
        data=get_regression_results()

        performance = generate_performance_dict(client_id="client_secret", metric_type="regression", created_at=datetime.utcnow().isoformat() + "Z", metric_id="string", name="string", type="string", description="string", source="string", data=data, collected_on="string", collected_by="string", authorized_user="string", test_mode=False)
        return performance
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {str(e)}")
    
"""
    Fetches regression results and generates a performance dictionary.

    This function attempts to retrieve regression results using the `get_regression_results` function.
    It then generates a performance dictionary with the retrieved data and additional metadata using
    the `generate_performance_dict` function. If an error occurs during the process, a RuntimeError
    is raised with the error message.

    Returns:
        dict: A dictionary containing performance metrics and metadata.

    Raises:
        RuntimeError: If there is an error fetching the regression results.

    Steps:
    1. Call the `get_regression_results` function to retrieve regression data.
    2. Generate a performance dictionary using the `generate_performance_dict` function with the following parameters:
        - client_id: "new_client105_evidently"
        - metric_type: "classification"
        - created_at: Current UTC time in ISO 8601 format with a "Z" suffix
        - metric_id: "string"
        - name: "string"
        - type: "string"
        - description: "string"
        - source: "string"
        - data: The retrieved regression data
        - collected_on: "2025-01-21T12:25:41.641Z"
        - collected_by: "string"
        - authorized_user: "string"
        - test_mode: False
    3. Return the generated performance dictionary.
    4. If an exception occurs, raise a RuntimeError with the error message.
"""