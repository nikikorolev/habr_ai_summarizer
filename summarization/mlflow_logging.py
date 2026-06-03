from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "habr-summarization"


def setup_mlflow(
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> str:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri is None:
        port = os.getenv("MLFLOW_PORT", "5050")
        uri = f"http://localhost:{port}"

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def _clean_metric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def log_summarization_results(
    results: pd.DataFrame,
    *,
    experiment_name: str = DEFAULT_EXPERIMENT,
    run_prefix: str = "summarization",
    sample_size: int | None = None,
    tracking_uri: str | None = None,
) -> str:
    uri = setup_mlflow(experiment_name=experiment_name, tracking_uri=tracking_uri)

    metrics_path = PROJECT_ROOT / "summarization" / "mlflow_results.csv"
    results.to_csv(metrics_path)

    with mlflow.start_run(run_name=f"{run_prefix}-summary") as parent_run:
        mlflow.log_params(
            {
                "task": "summarization",
                "models_count": len(results),
                "sample_size": sample_size if sample_size is not None else "unknown",
                "python_version": platform.python_version(),
            }
        )
        try:
            mlflow.log_artifact(str(metrics_path), artifact_path="tables")
        except ModuleNotFoundError as exc:
            mlflow.set_tag("artifact_logging_error", f"missing dependency: {exc.name}")

        best_model = None
        if "overall_score" in results.columns and len(results) > 0:
            best_model = str(results["overall_score"].astype(float).idxmax())
            mlflow.set_tag("best_model", best_model)

        for model_name, row in results.iterrows():
            with mlflow.start_run(
                run_name=f"{run_prefix}-{model_name}",
                nested=True,
            ):
                mlflow.set_tags(
                    {
                        "task": "summarization",
                        "model_name": str(model_name),
                        "model_type": str(row.get("type", "unknown")),
                        "is_best": str(model_name) == best_model,
                    }
                )

                for column, value in row.items():
                    metric = _clean_metric_value(value)
                    if metric is not None:
                        mlflow.log_metric(str(column), metric)
                    else:
                        mlflow.log_param(str(column), str(value))

        return parent_run.info.run_id
