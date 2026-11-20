"""Custom MLflow pyfunc wrapper around a Hugging Face sentiment pipeline (Lab 9 style).

Only the Hugging Face model ID and settings are stored in the MLflow model.
The weights are pulled from the local Hugging Face cache when the model is loaded,
so MLflow artifacts stay tiny.

Input : DataFrame with a column "text"
Output: DataFrame with columns text, prediction (0=NEGATIVE, 1=POSITIVE), label, score
"""
import mlflow.pyfunc
import pandas as pd

LABEL_TO_INT = {"NEGATIVE": 0, "POSITIVE": 1}


def normalise_label(raw_label: str) -> str:
    """Map different models' label names to NEGATIVE / POSITIVE."""
    r = str(raw_label).upper()
    if r in ("POSITIVE", "POS", "LABEL_1"):
        return "POSITIVE"
    if r in ("NEGATIVE", "NEG", "LABEL_0"):
        return "NEGATIVE"
    raise ValueError(f"Unexpected label from model: {raw_label}")


class HFSentimentModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str, max_length: int = 128, device: str = "cpu"):
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.device = device
        self._pipe = None  # built in load_context, never pickled

    def load_context(self, context):
        from transformers import pipeline

        self._pipe = pipeline(
            "text-classification",
            model=self.hf_model_id,
            device=-1 if self.device == "cpu" else self.device,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(t) for t in model_input]

        outputs = self._pipe(texts, truncation=True, max_length=self.max_length)
        labels = [normalise_label(o["label"]) for o in outputs]
        return pd.DataFrame(
            {
                "text": texts,
                "prediction": [LABEL_TO_INT[l] for l in labels],
                "label": labels,
                "score": [float(o["score"]) for o in outputs],
            }
        )
