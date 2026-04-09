from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _require_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment tracking. Install with `pip install -e .` or `pip install mlflow`."
        ) from exc
    return mlflow


def configure_tracking(tracking_uri: str | None = None, experiment_name: str = "artificial-data-generation"):
    mlflow = _require_mlflow()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_dataframe_artifact(df: pd.DataFrame, path: str | Path, artifact_path: str = "tables") -> Path:
    mlflow = _require_mlflow()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    mlflow.log_artifact(str(target), artifact_path=artifact_path)
    return target


def log_metrics(metrics: dict[str, float], prefix: str | None = None) -> None:
    mlflow = _require_mlflow()
    for key, value in metrics.items():
        if value is None:
            continue
        mlflow.log_metric(f"{prefix}_{key}" if prefix else key, float(value))


def log_params(params: dict[str, Any]) -> None:
    mlflow = _require_mlflow()
    mlflow.log_params(params)
