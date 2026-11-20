"""Runs Model A (Champion) and Model B (Challenger) on the SAME inputs and logs everything to MLflow.

Usage:
    python evaluate.py                 # full run (200 samples)
    N_SAMPLES=10 python evaluate.py    # quick debug run
"""
import inspect
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from mlflow.models import infer_signature
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_recall_fscore_support)
from transformers import pipeline

import config
from hf_wrapper import HFSentimentModel, LABEL_TO_INT, normalise_label

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
N_SAMPLES = int(os.environ.get("N_SAMPLES", 200))
SEED = 42
MAX_LENGTH = 128
DEVICE = "cpu"
DATASET_NAME = "glue/sst2 (validation split)"

MODELS = {
    "A": dict(role="champion", short="distilbert-sst2",
              hf_model_id="distilbert-base-uncased-finetuned-sst-2-english"),
    "B": dict(role="challenger", short="roberta-large",
              hf_model_id="siebert/sentiment-roberta-large-english"),
}

# ---- CHAMPION RULE (frozen BEFORE the final run - same text as README.md) ----
PRIMARY_METRIC = "macro_f1"      # higher is better
MARGIN = 0.01                    # Challenger must beat Champion by >= 1 percentage point
MAX_LATENCY_RATIO = 10.0         # guardrail: reject Challenger if avg latency > 10x Champion's
# Tie-breaker: if |difference| < MARGIN, the FASTER model (lower avg_latency_ms) wins.
# ------------------------------------------------------------------------------

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_eval_set():
    ds = load_dataset("glue", "sst2", split="validation").to_pandas()
    df = ds.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)
    df = df.rename(columns={"sentence": "text"})[["idx", "text", "label"]]
    df.to_csv(DATA_DIR / "eval_inputs.csv", index=False)
    return df


def predict_all(clf, texts):
    """Run one text at a time so we can time each prediction."""
    clf(texts[0], truncation=True, max_length=MAX_LENGTH)  # warm-up (not timed)
    preds, labels, scores, latencies = [], [], [], []
    for t in texts:
        start = time.perf_counter()
        out = clf(t, truncation=True, max_length=MAX_LENGTH)[0]
        latencies.append(time.perf_counter() - start)
        lab = normalise_label(out["label"])
        labels.append(lab)
        preds.append(LABEL_TO_INT[lab])
        scores.append(float(out["score"]))
    return preds, labels, scores, latencies


def compute_metrics(y_true, y_pred, latencies):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "avg_latency_ms": float(np.mean(latencies) * 1000),
        "total_inference_s": float(np.sum(latencies)),
    }


def log_pyfunc_model(wrapper, input_example, signature, metadata):
    """Works with both old (artifact_path=) and new (name=) MLflow versions."""
    kwargs = dict(
        python_model=wrapper,
        input_example=input_example,
        signature=signature,
        metadata=metadata,
        code_paths=["hf_wrapper.py"],
        pip_requirements=[f"mlflow=={mlflow.__version__}",
                          f"transformers=={transformers.__version__}",
                          f"torch=={torch.__version__.split('+')[0]}",
                          "pandas"],
    )
    params = inspect.signature(mlflow.pyfunc.log_model).parameters
    if "name" in params:
        return mlflow.pyfunc.log_model(name="model", **kwargs)
    return mlflow.pyfunc.log_model(artifact_path="model", **kwargs)


def confusion_fig(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["NEG", "POS"], ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    set_seed(SEED)
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    eval_df = load_eval_set()
    texts, y_true = eval_df["text"].tolist(), eval_df["label"].tolist()
    print(f"Evaluation set: {len(eval_df)} samples saved to data/eval_inputs.csv")

    lib_versions = (f"transformers={transformers.__version__}; torch={torch.__version__}; "
                    f"mlflow={mlflow.__version__}")
    common_tags = {"student_usn": config.STUDENT_USN, "library_versions": lib_versions,
                   "dataset_name": DATASET_NAME}

    results, run_ids, pred_frames = {}, {}, {}

    with mlflow.start_run(run_name=f"compare_{config.TASK}") as parent:
        for key, spec in MODELS.items():
            run_name = f"{key}_{config.TASK}_{spec['short']}"
            print(f"\n=== {run_name} ({spec['hf_model_id']}) ===")
            clf = pipeline("text-classification", model=spec["hf_model_id"],
                           device=-1 if DEVICE == "cpu" else DEVICE)
            preds, labels, scores, lats = predict_all(clf, texts)
            metrics = compute_metrics(y_true, preds, lats)
            print({k: round(v, 4) for k, v in metrics.items()})

            pred_df = pd.DataFrame({"input": texts, "reference": y_true, "prediction": preds,
                                    "label": labels, "score": scores,
                                    "correct": [int(a == b) for a, b in zip(y_true, preds)],
                                    "latency_ms": [l * 1000 for l in lats]})
            pred_path = OUT_DIR / f"predictions_{spec['short']}.csv"
            pred_df.to_csv(pred_path, index=False)
            pred_frames[key] = pred_df

            # wrapper for MLflow (reuse the loaded pipeline for the signature, then drop it)
            wrapper = HFSentimentModel(spec["hf_model_id"], max_length=MAX_LENGTH, device=DEVICE)
            wrapper._pipe = clf
            example = pd.DataFrame({"text": texts[:3]})
            signature = infer_signature(example, wrapper.predict(None, example))
            wrapper._pipe = None

            with mlflow.start_run(run_name=run_name, nested=True) as child:
                mlflow.log_params({
                    "hf_model_id": spec["hf_model_id"], "task": config.TASK, "role": spec["role"],
                    "max_length": MAX_LENGTH, "batch_size": 1, "eval_samples": len(eval_df),
                    "seed": SEED, "device": DEVICE,
                })
                mlflow.set_tags(common_tags)
                mlflow.log_metrics(metrics)

                info = log_pyfunc_model(wrapper, example, signature,
                                        metadata={"hf_model_id": spec["hf_model_id"],
                                                  "role": spec["role"], "task": config.TASK})
                mlflow.set_tag("model_uri", info.model_uri)  # used by register_champion.py

                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))
                mlflow.log_dict({k: float(v) for k, v in metrics.items()}, "metrics_summary.json")
                (OUT_DIR / f"metrics_summary_{spec['short']}.json").write_text(json.dumps(metrics, indent=2))

                fig = confusion_fig(y_true, preds, f"{spec['short']} confusion matrix")
                mlflow.log_figure(fig, "confusion_matrix.png")
                fig.savefig(OUT_DIR / f"confusion_matrix_{spec['short']}.png", dpi=150)
                plt.close(fig)
                run_ids[key] = child.info.run_id

            results[key] = metrics
            del clf  # free memory before loading the next model

        # ------------------------- Champion-Challenger decision -------------------------
        mA, mB = results["A"], results["B"]
        delta = mB[PRIMARY_METRIC] - mA[PRIMARY_METRIC]
        latency_ratio = mB["avg_latency_ms"] / mA["avg_latency_ms"]
        if delta >= MARGIN:
            if latency_ratio <= MAX_LATENCY_RATIO:
                winner, reason = "B", f"Challenger beat Champion by {delta:.4f} (>= margin {MARGIN}) within latency guardrail."
            else:
                winner, reason = "A", (f"Challenger improved by {delta:.4f} but is {latency_ratio:.1f}x slower "
                                       f"(> {MAX_LATENCY_RATIO}x guardrail); Champion stays.")
        elif delta <= -MARGIN:
            winner, reason = "A", f"Challenger is worse by {abs(delta):.4f}; Champion stays."
        else:
            winner = "A" if mA["avg_latency_ms"] <= mB["avg_latency_ms"] else "B"
            reason = f"Difference {delta:.4f} is within margin {MARGIN}; tie-breaker -> faster model ({winner})."
        print(f"\nDECISION: winner = Model {winner}. {reason}")

        mlflow.log_params({"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                           "max_latency_ratio": MAX_LATENCY_RATIO, "direction": "higher_is_better",
                           "eval_samples": len(eval_df), "seed": SEED,
                           "model_A": MODELS["A"]["hf_model_id"], "model_B": MODELS["B"]["hf_model_id"]})
        mlflow.set_tags({**common_tags, "winner": winner,
                         "winner_model_id": MODELS[winner]["hf_model_id"],
                         "run_id_A": run_ids["A"], "run_id_B": run_ids["B"], "decision_reason": reason})
        mlflow.log_metrics({"delta_primary_metric": delta, "latency_ratio": latency_ratio,
                            **{f"A_{k}": v for k, v in mA.items()}, **{f"B_{k}": v for k, v in mB.items()}})

        # ------------------------------- Summary artifacts -------------------------------
        summary = {"A": mA, "B": mB, "delta_primary_metric": delta, "latency_ratio": latency_ratio,
                   "winner": winner, "reason": reason}
        mlflow.log_dict(summary, "metrics_summary.json")
        (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

        names = ["accuracy", "precision", "recall", "f1", "macro_f1"]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - 0.2, [mA[n] for n in names], 0.4, label=f"A {MODELS['A']['short']}")
        ax.bar(x + 0.2, [mB[n] for n in names], 0.4, label=f"B {MODELS['B']['short']}")
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Champion vs Challenger (higher is better)")
        ax.legend()
        fig.tight_layout()
        mlflow.log_figure(fig, "metrics_comparison.png")
        fig.savefig(OUT_DIR / "metrics_comparison.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.bar([MODELS["A"]["short"], MODELS["B"]["short"]], [mA["avg_latency_ms"], mB["avg_latency_ms"]],
               color=["tab:blue", "tab:orange"])
        ax.set_ylabel("avg latency per sample (ms) - lower is better")
        ax.set_title("Inference latency")
        fig.tight_layout()
        mlflow.log_figure(fig, "latency_comparison.png")
        fig.savefig(OUT_DIR / "latency_comparison.png", dpi=150)
        plt.close(fig)

        # qualitative examples: 5 cases where the models disagree (write your own comments!)
        a, b = pred_frames["A"], pred_frames["B"]
        dis = pd.DataFrame({"input": a["input"], "reference": a["reference"],
                            "pred_A": a["prediction"], "pred_B": b["prediction"]})
        dis = dis[dis["pred_A"] != dis["pred_B"]].head(5).copy()
        dis["correct_model"] = np.where(dis["pred_A"] == dis["reference"], "A", "B")
        dis["comment"] = ""  # TODO: fill in your own short comment for each row
        dis.to_csv(OUT_DIR / "disagreements.csv", index=False)
        mlflow.log_artifact(str(OUT_DIR / "disagreements.csv"))
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))
        if Path("README.md").exists():
            mlflow.log_artifact("README.md")
        mlflow.log_artifact("requirements.txt")
        print(f"\nDisagreements between models: {int((a['prediction'] != b['prediction']).sum())} of {len(a)}")
        print("Done. Open", config.TRACKING_URI)


if __name__ == "__main__":
    main()
