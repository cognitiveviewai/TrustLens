# Here, is an example of how to implement biased content functionality using evidently.ai

import pandas as pd
from evidently.descriptors import BiasLLMEval
from evidently.metric_preset import TextEvals
from evidently.report import Report
import os

os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

# Dummy data
assistant_log = [
    {
        "response": 'I cannot answer you how to make lethal weapons.'
    }
]

# Converting the log into a pandas DataFrame
assistant_log = pd.DataFrame(assistant_log)

# Instantiating the "Report" object
report = Report(metrics=[
    TextEvals(column_name = "response", descriptors = [
        BiasLLMEval()
    ])
])

# Executing the "Report" object to generate results
report.run(reference_data = None, current_data = assistant_log)

# Extracting the intended value from the "Report" object after generating the metrics
biased_content = report.datasets().current['Bias category'].values[0]
print(f"Biased Content: {biased_content}")

