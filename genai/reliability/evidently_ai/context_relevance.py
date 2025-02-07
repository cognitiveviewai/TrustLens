# This is a sample code for how to generate the context_relevance_score using evidently ai
import pandas as pd
from evidently.descriptors import ContextQualityLLMEval
from evidently.metric_preset import TextEvals
from evidently.report import Report
import os

os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

# Dummy data
assistant_log = [
    {
        "question": "Is DELL company earthquake resilent ?",
        "context": "DELL answered 'YES' to survey questions like Is your company continuing production?, Is your company having proper logistics? in  earthquake survey."
    }
]

assistant_log = pd.DataFrame(assistant_log)

# Use evidently AI's ContextQualityLLMEval
text_evals_report =Report(metrics=[
    TextEvals(column_name = "context", descriptors=[
        ContextQualityLLMEval(question = "question"),
    ])
])

# Running the "Report" object to generate the metrics
text_evals_report.run(reference_data = None, current_data = assistant_log)

# Extracting the intended value from the "Report" object after generating the metrics
eval = text_evals_report.datasets().current['ContextQuality category'].values[0]
eval.head()