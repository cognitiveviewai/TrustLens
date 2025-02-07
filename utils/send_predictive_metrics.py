import json
import requests
from utils import send_payload
with open('/home/cv-011/copilot-connector/metrics_predictive.json') as f:
    predictive_data = json.load(f)

def send_post_request(url, data, headers=None):
    response = requests.post(url, json=data, headers=headers)
    return response.json()

# Endpoint of the evidence API where you need to do a POST method with the payload
url = "https://backend.cognitiveview.com/api/v1/evidence/"

# Populate the payload according to the schema, you can only choose those metrics which are relevant for the model you use
payload = send_payload("T-123456", "CL-123456", "C-123456", "O-123456", "Admin", "user@gmail.com", ["user1", "user2"], "Confidential", "Evidence Document", "This is an evidence document", "API", "1.0", "Source", "https://www.example.com/evidence_document", predictive_data)
# Sending the POST request
response = send_post_request(url, payload)
print(response)