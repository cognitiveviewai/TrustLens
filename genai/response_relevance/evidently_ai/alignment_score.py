# Here, is an example of how to generate alignment_score using evidently.ai. This works by generating embedding for both ground_truth, and response using a open-source deep learning encoder model.

import pandas as pd
from evidently.descriptors import SemanticSimilarity
from evidently.report import Report
from evidently.metric_preset import TextEvals

# Dummy data
assistant_log = [
    {
        "ground_truth": " Universe is a magical place.",
        "response": 'Universe is a place with full of magic.'
    }
]

# Converting the log into a pandas DataFrame
assistant_log = pd.DataFrame(assistant_log)

# Creating the "Report" object to generate semantic similarity between ground truth
report = Report(metrics=[
    TextEvals(column_name = "response", descriptors=[
        SemanticSimilarity(with_column = "ground_truth"),       
        ]
    ),
])

# Running the "Report" object to generate semanticSimilarity of the generated response, with the ground truth
report.run(reference_data = None, current_data = assistant_log)

# Extracting the alignment score from the 
alignment_score = float(report.as_dict()["metrics"][0]["result"]["current_characteristics"]["mean"])
print(alignment_score)

