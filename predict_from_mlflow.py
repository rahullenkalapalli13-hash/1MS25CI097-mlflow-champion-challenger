"""Loads a model from the MLflow registry by ALIAS and predicts on new data.

This script deliberately does NOT import transformers / Hugging Face.
It only knows the registered model name and an alias, not which version is the Champion.

Usage:
    python predict_from_mlflow.py                 # champion only
    python predict_from_mlflow.py --both          # champion and challenger
"""
import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

import config


def predict_with_alias(alias, df):
    client = MlflowClient()
    mv = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, alias)
    uri = f"models:/{config.REGISTERED_MODEL_NAME}@{alias}"
    model = mlflow.pyfunc.load_model(uri)
    meta = model.metadata.metadata or {}
    hf_id = meta.get("hf_model_id") or mv.tags.get("hf_model_id", "unknown")
    print(f"\nLoaded {uri}  ->  version {mv.version}  |  Hugging Face model: {hf_id}")

    out = model.predict(df[["text"]])
    result = df.copy()
    result["prediction"] = out["label"].values
    result["score"] = out["score"].round(4).values
    result["model_alias"] = alias
    result["model_version"] = mv.version
    result["hf_model_id"] = hf_id
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/new_data.csv")
    parser.add_argument("--out", default="outputs/new_data_predictions.csv")
    parser.add_argument("--both", action="store_true", help="also run the @challenger alias")
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.TRACKING_URI)
    df = pd.read_csv(args.data)

    aliases = ["champion", "challenger"] if args.both else ["champion"]
    frames = []
    for alias in aliases:
        res = predict_with_alias(alias, df)
        show = [c for c in ("text", "expected_label", "prediction", "score") if c in res.columns]
        print(res[show].to_string(index=False, max_colwidth=60))
        if "expected_label" in res.columns:
            print(f"Agreement with expected_label: {(res['expected_label'] == res['prediction']).mean():.0%}")
        frames.append(res)

    pd.concat(frames).to_csv(args.out, index=False)
    print(f"\nSaved predictions to {args.out}")


if __name__ == "__main__":
    main()
