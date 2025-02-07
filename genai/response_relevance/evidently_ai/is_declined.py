# Here, is an example of how to implement "whether the response declined" functionality using evidently.ai

import pandas as pd
from evidently.descriptors import DeclineLLMEval
from evidently.metric_preset import TextEvals
from evidently.report import Report
import os

os.environ["OPENAI_API_KEY"] = 'YOUR API KEY'

# Dummy data
assistant_log = [
    {
        "response": "I can't answer how to make explosives."
    }
]

# Converting the log into a pandas DataFrame
assistant_log = pd.DataFrame(assistant_log)

# Instantiating the "Report" object
report = Report(metrics=[
    TextEvals(column_name = "response", descriptors = [
        DeclineLLMEval()
    ])
])

# Executing the "Report" object to generate results
report.run(reference_data = None, current_data = assistant_log)

# Extracting the intended value from the "Report" object after generating the metrics
eval = report.datasets().current['Decline category'].values[0]


print(eval)
