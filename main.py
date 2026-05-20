from __future__ import annotations

import argparse
from pathlib import Path

from config import config, ensure_project_dirs
from preprocessing import list_csv_files, profile_raw_files, read_schema_sample


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline CU Venta - Tarea 2")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-data", help="Inspecciona CSV sin cargar todo el dataset.")
    inspect_parser.add_argument("--data-dir", type=Path, default=config.paths.raw_dir)
    inspect_parser.add_argument("--sample-rows", type=int, default=5000)

    subparsers.add_parser("train", help="Pendiente etapa 2.")
    subparsers.add_parser("inference", help="Pendiente etapa 3.")
    subparsers.add_parser("run-all", help="Pendiente etapas 2 y 3.")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "inspect-data":
        inspect_data(args.data_dir, args.sample_rows)
        return

    raise NotImplementedError(f"Comando '{args.command}' se implementara en las siguientes etapas.")


if __name__ == "__main__":
    main()

