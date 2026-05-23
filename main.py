from __future__ import annotations

import argparse
from pathlib import Path

from config import config, ensure_project_dirs
from preprocessing import list_csv_files, profile_raw_files, read_schema_sample, run_preprocessing


def inspect_data(data_dir: Path, sample_rows: int) -> None:
    ensure_project_dirs()
    files = list_csv_files(data_dir)
    if not files:
        print(f"No se encontraron CSV en: {data_dir}")
        print("Descarga el dataset del Drive indicado en README.md o usa --data-dir.")
        return

    print("\nArchivos encontrados")
    print(profile_raw_files(data_dir, sample_rows).to_string(index=False))

    first_file = files[0]
    print(f"\nEsquema muestral de: {first_file.name}")
    print(read_schema_sample(first_file, sample_rows).to_string(index=False))


def train_pipeline(
    data_dir: Path,
    validation_codmes: float | None,
    n_trials: int,
    max_retrain_attempts: int,
) -> None:
    import pandas as pd

    from training import train_and_log

    ensure_project_dirs()
    processed = run_preprocessing(data_dir=data_dir, validation_codmes=validation_codmes)
    run_id, model = train_and_log(
        train_path=Path(processed["train"]),
        test_path=Path(processed["test"]),
        metadata_path=Path(processed["metadata"]),
        n_trials=n_trials,
    )

    report = _score_and_monitor(processed, model)

    print(f"Run MLflow: {run_id}")
    print(f"PSI score: {report['score_psi']:.6f} ({report['score_psi_status']})")
    print(f"Reentrenamiento requerido: {report['requires_retraining']}")

    for attempt in range(max_retrain_attempts):
        if not report["requires_retraining"]:
            break
        print(f"Reentrenamiento automatico por alerta de monitoreo. Intento {attempt + 1}.")
        run_id, model = train_and_log(
            train_path=Path(processed["train"]),
            test_path=Path(processed["test"]),
            metadata_path=Path(processed["metadata"]),
            n_trials=n_trials,
        )
        report = _score_and_monitor(processed, model)
        print(f"Run MLflow reentrenado: {run_id}")
        print(f"PSI score reentrenado: {report['score_psi']:.6f} ({report['score_psi_status']})")


def _score_and_monitor(processed: dict[str, str], model: object) -> dict[str, object]:
    import pandas as pd

    from monitoring import run_monitoring
    from training import load_feature_columns

    feature_columns = load_feature_columns(Path(processed["metadata"]))
    df_train = pd.read_csv(processed["train"])
    df_val = pd.read_csv(processed["val"])

    train_scores = model.predict_proba(df_train[feature_columns])[:, 1]
    val_scores = model.predict_proba(df_val[feature_columns])[:, 1]
    return run_monitoring(
        train_scores=train_scores,
        val_scores=val_scores,
        y_val=df_val[config.columns.target].astype(int).to_numpy(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline CU Venta - Tarea 2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspecciona CSV sin cargar todo el dataset.")
    inspect_parser.add_argument("--data-dir", type=Path, default=config.paths.raw_dir)
    inspect_parser.add_argument("--sample-rows", type=int, default=5000)

    train_parser = subparsers.add_parser("train", help="Ejecuta preprocesamiento, Optuna, MLflow y monitoreo.")
    train_parser.add_argument("--data-dir", type=Path, default=config.paths.raw_dir)
    train_parser.add_argument("--validation-codmes", type=float, default=None)
    train_parser.add_argument("--n-trials", type=int, default=30)
    train_parser.add_argument("--max-retrain-attempts", type=int, default=1)

    subparsers.add_parser("inference", help="Pendiente etapa 3.")
    run_all_parser = subparsers.add_parser("run-all", help="Ejecuta el flujo disponible hasta fase 2.")
    run_all_parser.add_argument("--data-dir", type=Path, default=config.paths.raw_dir)
    run_all_parser.add_argument("--validation-codmes", type=float, default=None)
    run_all_parser.add_argument("--n-trials", type=int, default=30)
    run_all_parser.add_argument("--max-retrain-attempts", type=int, default=1)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "inspect-data":
        inspect_data(args.data_dir, args.sample_rows)
        return
    if args.command in {"train", "run-all"}:
        train_pipeline(args.data_dir, args.validation_codmes, args.n_trials, args.max_retrain_attempts)
        return

    raise NotImplementedError(f"Comando '{args.command}' se implementara en las siguientes etapas.")


if __name__ == "__main__":
    main()
