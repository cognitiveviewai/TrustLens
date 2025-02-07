import json
import os
from datetime import datetime

def get_metrics(dict_name, value):
    metrics = {
        dict_name: {
            "metric_value": value,
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "model_id": "string",
            "metric_source": "string"
        }
    }

    # Define the file path
    file_path = os.path.join('/home/cv-011/copilot-connector/', 'metrics_predictive.json')

    # Load existing metrics if the file exists
    if os.path.exists(file_path):
        with open(file_path, 'r') as json_file:
            all_metrics = json.load(json_file)
    else:
        all_metrics = {}

    # Update the metrics
    all_metrics.update(metrics)

    # Save the updated metrics as a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(all_metrics, json_file, indent=4)

    return all_metrics
