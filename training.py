from __future__ import annotations

import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from config import config


def train_and_log(
    train_path: Path | None = None,
    test_path: Path | None = None,
    metadata_path: Path | None = None,
    n_trials: int = 30,
) -> tuple[str, object]:
    """entrena XGBoost con Optuna y registra el modelo en MLflow."""

    train_path = train_path or config.paths.processed_dir / "df_train.csv"
    test_path = test_path or config.paths.processed_dir / "df_test.csv"
    metadata_path = metadata_path or config.paths.processed_dir / "preprocessing_metadata.json"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feature_columns = load_feature_columns(metadata_path, df_train)

    x_train, y_train = _xy(df_train, feature_columns)
    x_test, y_test = _xy(df_test, feature_columns)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "eval_metric": "logloss",
            "random_state": config.split.random_state,
            "n_jobs": -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], verbose=False)
        score = model.predict_proba(x_test)[:, 1]
        return roc_auc_score(y_test, score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = {
        **study.best_params,
        "eval_metric": "logloss",
        "random_state": config.split.random_state,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**best_params)
    model.fit(x_train, y_train)

    test_score = model.predict_proba(x_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_score)

    mlflow.set_tracking_uri(config.paths.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name="xgb_optuna") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("best_optuna_auc", study.best_value)
        mlflow.log_artifact(str(metadata_path), artifact_path="preprocessing")
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=config.model_name,
        )
        run_id = run.info.run_id

    config.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.paths.metrics_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_name": config.model_name,
                "test_auc": test_auc,
                "best_optuna_auc": study.best_value,
                "n_trials": n_trials,
                "feature_count": len(feature_columns),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    config.paths.model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(config.paths.model_dir / "xgb_model.json")
    return run_id, model


def load_model(model_path: Path | None = None) -> object:
    model_path = model_path or config.paths.model_dir / "xgb_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo entrenado: {model_path}")

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model


def load_feature_columns(metadata_path: Path, df_train: pd.DataFrame | None = None) -> list[str]:
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata["feature_columns"]

    if df_train is None:
        raise FileNotFoundError(f"No existe metadata de preprocesamiento: {metadata_path}")

    protected = set(config.columns.protected) | {"_source_file"}
    return [
        column
        for column in df_train.columns
        if column not in protected and pd.api.types.is_numeric_dtype(df_train[column])
    ]


def _xy(df: pd.DataFrame, feature_columns: list[str]):
    x = df[feature_columns].copy()
    y = df[config.columns.target].astype(int)
    return x, y
