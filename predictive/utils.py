def generate_performance_dict(client_id: str, metric_type: str, created_at: str, metric_id: str, name: str, type: str, description: str, source: str, data: dict, collected_on: str, collected_by: str, authorized_user: str, test_mode: bool) -> dict:
    payload= {
        "client_id": client_id,
        "metric_type": metric_type,
        "created_at": created_at,
        "metric_id": metric_id,
        "name": name,
        "type": type,
        "description": description,
        "source": source,
        "data": data,
        "collected_on": collected_on,
        "collected_by": collected_by,
        "authorized_user": authorized_user,
        "test_mode": test_mode
    }
    return payload

"""
    Generate a performance dictionary with the provided parameters.

    Parameters:
    - client_id (str): Unique identifier for the client.
    - metric_type (str): Type of the metric being recorded.
    - created_at (str): Timestamp of when the metric was created.
    - metric_id (str): Unique identifier for the metric.
    - name (str): Name of the metric.
    - type (str): Type/category of the metric.
    - description (str): Detailed description of the metric.
    - source (str): Source from where the metric data is collected.
    - data (dict): Dictionary containing the metric data.
    - collected_on (str): Timestamp of when the data was collected.
    - collected_by (str): Identifier of the entity that collected the data.
    - authorized_user (str): User authorized to access the metric.
    - test_mode (bool): Flag indicating if the metric is in test mode.

    Returns:
    - dict: A dictionary containing all the provided parameters as key-value pairs.
"""

