import requests
from model_metrics import fetch_model_performance

from alerts import evaluate_metrics

def send_post_request(url, data, headers=None):

    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Endpoint of the evidence API where you need to do a POST method with the payload
url = ""


# Populate the payload according to the schema, you can only choose those metrics which are relevant for the model you use
payload = fetch_model_performance()
# Sending the POST request

response = send_post_request(url, payload)

simplified_metrics = {key: value['metric_value'] for key, value in payload["data"].items()}
evaluate_metrics(simplified_metrics)
"""
    Sends a POST request to the specified URL with the given data and optional headers.

    Args:
        url (str): The URL to which the POST request is sent.
        data (dict): The data to be sent in the POST request body as JSON.
        headers (dict, optional): Optional headers to include in the POST request.

    Returns:
        dict: The JSON response from the server.

    Steps:
        1. Sends a POST request to the specified URL.
        2. Includes the provided data in the request body as JSON.
        3. Optionally includes any provided headers in the request.
        4. Receives the response from the server.
        5. Parses the response as JSON and returns it.
"""



