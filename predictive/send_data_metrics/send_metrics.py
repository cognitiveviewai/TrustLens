import requests
from data_metrics import fetch_data_results


def send_post_request(url, data, headers=None):
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Endpoint of the evidence API where you need to do a POST method with the payload
url = ""

# Populate the payload according to the schema, you can only choose those metrics which are relevant for the model you use
payload = fetch_data_results()
# Sending the POST request

# print(payload)
response = send_post_request(url, payload)
"""
    Sends a POST request to the specified URL with the given data and optional headers.

    Parameters:
    url (str): The URL to which the POST request is sent.
    data (dict): The data to be sent in the body of the POST request. This should be a dictionary that will be converted to JSON.
    headers (dict, optional): A dictionary of HTTP headers to send with the request. Defaults to None.

    Returns:
    dict: The JSON response from the server, parsed into a dictionary.

    Steps:
    1. The function takes three parameters: `url`, `data`, and an optional `headers`.
    2. It uses the `requests.post` method to send a POST request to the specified `url`.
    3. The `data` parameter is passed as JSON in the body of the request.
    4. If `headers` are provided, they are included in the request; otherwise, the default headers are used.
    5. The response from the server is captured in the `response` variable.
    6. The function then calls the `json` method on the `response` object to parse the JSON response from the server.
    7. Finally, the parsed JSON response is returned as a dictionary.
"""

