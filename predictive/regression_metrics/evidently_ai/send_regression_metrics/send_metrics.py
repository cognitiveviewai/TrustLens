
import requests
from regression_metrics import fetch_regression_results

from alerts import evaluate_metrics

def send_post_request(url, data, headers=None):
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Endpoint of the evidence API where you need to do a POST method with the payload
url = ""


# Populate the payload according to the schema, you can only choose those metrics which are relevant for the model you use
payload = fetch_regression_results()
# Sending the POST request

response = send_post_request(url, payload)

simplified_metrics = {key: value for key, value in payload["data"].items()}
evaluate_metrics(simplified_metrics)
"""
    Sends a POST request to the specified URL with the given data and optional headers.

    Parameters:
    url (str): The URL to which the POST request is to be sent.
    data (dict): The data to be sent in the body of the POST request. This should be a dictionary that will be converted to JSON.
    headers (dict, optional): A dictionary of HTTP headers to send with the request. Defaults to None.

    Returns:
    dict: The JSON response from the server, parsed into a dictionary.

    Raises:
    requests.exceptions.RequestException: If there is an issue with the network request.
"""


