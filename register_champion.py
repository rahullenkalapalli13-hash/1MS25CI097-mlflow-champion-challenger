"""Registers both logged models in the MLflow Model Registry and sets the aliases.

Reads the latest compare_<task> run, looks at its `winner` tag, then:
    winner     -> alias "champion"
    the other  -> alias "challenger"
No run IDs are hard-coded, so it works on any machine after evaluate.py has been run.
"""
import mlflow
from mlflow import MlflowClient

import config


def main():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    client = MlflowClient()

    parents = mlflow.search_runs(
        experiment_names=[config.EXPERIMENT_NAME],
        filter_string=f"attributes.run_name = 'compare_{config.TASK}'",
        order_by=["attributes.start_time DESC"], max_results=1, output_format="list")
    if not parents:
        raise SystemExit("No compare run found. Run evaluate.py first.")
    parent = parents[0]
    winner = parent.data.tags["winner"]
    print(f"Latest compare run: {parent.info.run_id} | winner = Model {winner}")

    try:
        client.create_registered_model(config.REGISTERED_MODEL_NAME)
    except Exception:
        pass  # already exists

    for key in ("A", "B"):
        run = client.get_run(parent.data.tags[f"run_id_{key}"])
        mv = mlflow.register_model(run.data.tags["model_uri"], config.REGISTERED_MODEL_NAME)
        alias = "champion" if key == winner else "challenger"
        client.set_registered_model_alias(config.REGISTERED_MODEL_NAME, alias, mv.version)
        client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version, "hf_model_id", run.data.params["hf_model_id"])
        client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version, "source_run_id", run.info.run_id)
        client.set_model_version_tag(config.REGISTERED_MODEL_NAME, mv.version, "macro_f1", f"{run.data.metrics['macro_f1']:.4f}")
        print(f"Model {key} -> version {mv.version} -> alias '{alias}' ({run.data.params['hf_model_id']})")


if __name__ == "__main__":
    main()
