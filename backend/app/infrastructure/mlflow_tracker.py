from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, Iterable

import mlflow
import torch

from app.core.settings import settings

logger = logging.getLogger(__name__)


def mlflow_enabled() -> bool:
    return bool(settings.mlflow_enabled)


def configure_mlflow() -> None:
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_training_run(
    run_name: str,
    params: Dict[str, Any],
    training_summary: Dict[str, Any],
    comparison: Dict[str, Any],
    model: torch.nn.Module | None = None,
) -> str | None:
    if not mlflow_enabled():
        return None

    try:
        configure_mlflow()
        with mlflow.start_run(run_name=run_name) as run:
            _log_params(params)
            _log_training_summary(training_summary)
            _log_comparison_summary(comparison)

            if settings.mlflow_log_models and model is not None:
                _log_model_artifact(model)

            return run.info.run_id
    except Exception:
        logger.exception("MLflow logging failed")
        return None


def _log_params(params: Dict[str, Any]) -> None:
    cleaned = {key: value for key, value in params.items() if value is not None}
    if cleaned:
        mlflow.log_params(cleaned)


def _log_training_summary(training_summary: Dict[str, Any]) -> None:
    reward_curve = training_summary.get("reward_curve", [])

    for entry in _iter_reward_curve(reward_curve):
        step = int(entry.get("episode", 0))
        _log_metric("reward", entry.get("reward"), step)
        _log_metric("moving_average_reward", entry.get("moving_average_reward"), step)
        _log_metric("policy_loss", entry.get("policy_loss"), step)
        _log_metric("value_loss", entry.get("value_loss"), step)
        _log_metric("entropy", entry.get("entropy"), step)

    _log_metric("final_training_reward", training_summary.get("final_training_reward"), None)
    _log_metric("best_training_reward", training_summary.get("best_training_reward"), None)
    _log_metric("duration_seconds", training_summary.get("duration_seconds"), None)

    final_eval = training_summary.get("final_eval_metrics", {})
    if isinstance(final_eval, dict):
        for key, value in final_eval.items():
            _log_metric(f"final_eval_{key}", value, None)

    if reward_curve:
        mlflow.log_dict(reward_curve, "reward_curve.json")

    mlflow.log_dict(training_summary, "training_summary.json")


def _log_comparison_summary(comparison: Dict[str, Any]) -> None:
    if not comparison:
        return

    ppo_metrics = comparison.get("ppo", {})
    if isinstance(ppo_metrics, dict):
        for key, value in ppo_metrics.items():
            _log_metric(f"ppo_{key}", value, None)

    rule_metrics = comparison.get("rule", {})
    if isinstance(rule_metrics, dict):
        for key, value in rule_metrics.items():
            _log_metric(f"rule_{key}", value, None)

    deltas = comparison.get("deltas", {})
    if isinstance(deltas, dict):
        for key, value in deltas.items():
            _log_metric(f"delta_{key}", value, None)

    mlflow.log_dict(comparison, "comparison.json")


def _log_model_artifact(model: torch.nn.Module) -> None:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as temp_file:
            temp_path = temp_file.name
        torch.save(model.state_dict(), temp_path)
        mlflow.log_artifact(temp_path, artifact_path="models")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Failed to remove temp model artifact %s", temp_path)


def _log_metric(name: str, value: Any, step: int | None) -> None:
    if value is None:
        return
    try:
        metric_value = float(value)
    except (TypeError, ValueError):
        return

    if step is None:
        mlflow.log_metric(name, metric_value)
    else:
        mlflow.log_metric(name, metric_value, step=step)


def _iter_reward_curve(reward_curve: Iterable[Dict[str, Any]]):
    for entry in reward_curve:
        if isinstance(entry, dict):
            yield entry
