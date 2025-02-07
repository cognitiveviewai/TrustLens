from predictive.utils import generate_performance_dict
from datetime import datetime
from data import get_data_metrics

def fetch_data_results() -> dict:
    """
    Fetches data metrics and generates a performance dictionary.

    This function attempts to retrieve data metrics using the `get_data_metrics` function.
    It then generates a performance dictionary using the `generate_performance_dict` function
    with predefined parameters and the retrieved data.

    Returns:
        dict: A dictionary containing performance metrics.

    Raises:
        RuntimeError: If there is an error fetching data metrics or generating the performance dictionary.

    Steps:
    1. Call the `get_data_metrics` function to retrieve data metrics.
    2. Use the retrieved data to generate a performance dictionary by calling the `generate_performance_dict` function
       with the following parameters:
       - client_id: "string"
       - metric_type: "classification"
       - created_at: Current UTC time in ISO 8601 format with a "Z" suffix
       - metric_id: "string"
       - name: "string"
       - type: "string"
       - description: "string"
       - source: "string"
       - data: Retrieved data from `get_data_metrics`
       - collected_on: "2025-01-21T12:25:41.641Z"
       - collected_by: "string"
       - authorized_user: "string"
       - test_mode: False
    3. Return the generated performance dictionary.
    4. If any exception occurs during the process, raise a RuntimeError with the error message.
    """

    try:
        data=get_data_metrics()

        performance = generate_performance_dict(client_id="client_secret", metric_type="classification", created_at=datetime.utcnow().isoformat() + "Z", metric_id="string", name="string", type="string", description="string", source="string", data=data, collected_on="string", collected_by="string", authorized_user="string", test_mode=False)
        return performance
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {str(e)}")
    
