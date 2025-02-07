# evidently_dictionary maps accepted values (keys) to Evidently AI generated values for generative metrics (values)
evidently_generative_dictionary = {
    "is_declined": {
        "DECLINE": "DECLINE",  # Accepted: DECLINE, Evidently: DECLINE
        "OK": "OK"             # Accepted: OK, Evidently: OK
    },
    "detect_pii": {
        "PII": "PII",          # Accepted: PII, Evidently: PII
        "OK": "OK"             # Accepted: OK, Evidently: OK
    },
    "negative_content": {
        "NEGATIVE": "NEGATIVE",# Accepted: NEGATIVE, Evidently: NEGATIVE
        "POSITIVE": "POSITIVE" # Accepted: POSITIVE, Evidently: POSITIVE
    },
    "biased_content": {
        "BIAS": "BIAS",        # Accepted: BIAS, Evidently: BIAS
        "OK": "OK"             # Accepted: OK, Evidently: OK
    },
    "toxic_content": {
        "TOXICITY": "TOXICITY",# Accepted: TOXICITY, Evidently: TOXICITY
        "OK": "OK"             # Accepted: OK, Evidently: OK
    },
    "is_context_relevant": {
        "VALID": "VALID",      # Accepted: VALID, Evidently: VALID
        "INVALID": "INVALID"   # Accepted: INVALID, Evidently: INVALID
    }
}

# Mapping of metric keys to the type of values they are expected to hold
# These types specify the format of data that should be passed to each key.
evidently_predicitve = {
    "accuracy_score": "float",  # Type of value: float
    "precision_score": "float",  # Type of value: float
    "recall_score": "float",  # Type of value: float
    "f1_score": "float",  # Type of value: float
    "tpr_value": "float",  # True Positive Rate, Type of value: float
    "fpr_value": "float",  # False Positive Rate, Type of value: float
    "tnr_value": "float",  # True Negative Rate, Type of value: float
    "fnr_value": "float",  # False Negative Rate, Type of value: float
    "log_loss_value": "float",  # Type of value: float
    "roc_auc_score": "float",  # Type of value: float
    "r_square_error": "float",  # Type of value: float
    "mean_absolute_error": "float",  # Type of value: float
    "mean_error": "float",  # Type of value: float
    "mean_absolute_percentage_error": "float",  # Type of value: float
}