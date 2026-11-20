"""Shared settings used by evaluate.py, register_champion.py and predict_from_mlflow.py."""

STUDENT_USN = "1MSXXXXXXX"          # TODO: put your real USN here
# Port 5001 (not 5000): on macOS, AirPlay Receiver already uses port 5000.
TRACKING_URI = "http://127.0.0.1:5001"
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
TASK = "sentiment"
