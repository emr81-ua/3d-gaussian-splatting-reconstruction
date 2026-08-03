from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_BLENDER_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "blender"
if str(_BLENDER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_BLENDER_SCRIPTS_DIR))

from procesar_pruebas_colmap import ensure_dataset_metadata
from run_colmap_pipeline import DEFAULT_COLMAP_EXE, list_images, verify_dense_structure


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procesa con COLMAP los datasets definidos por los JSON de experimentos de Blender."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "blender" / "experimentos",
        help="Carpeta con los JSON de experimentos.",
    )
    parser.add_argument(
        "--colmap-exe",
        type=Path,
        default=DEFAULT_COLMAP_EXE,
        help="Ruta a colmap.exe.",
    )
    parser.add_argument(
        "--output-name",
        default="dense",
        help="Nombre de la carpeta de salida de COLMAP dentro de cada dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra lo que se ejecutaria sin lanzar COLMAP.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Nombre concreto de experimento a procesar. Se puede repetir.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Si un experimento falla, continua con los siguientes y resume al final.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def dataset_dir_from_config(config: dict) -> Path:
    output_root = str(config.get("output_root", "pruebas"))
    experiment_name = str(config["experiment_name"])
    return (PROJECT_ROOT / output_root / experiment_name).resolve()


def build_command(dataset_dir: Path, args: argparse.Namespace) -> list[str]:
    images_dir = dataset_dir / "images"
    dense_dir = dataset_dir / args.output_name
    script_path = Path(__file__).resolve().parent / "run_colmap_pipeline.py"

    command = [
        sys.executable,
        str(script_path),
        str(images_dir),
        "--output-dir", str(dense_dir),
        "--colmap-exe", str(args.colmap_exe.resolve()),
    ]

    if args.dry_run:
        command.append("--dry-run")

    return command


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    if not config_dir.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de configuraciones: {config_dir}")

    config_paths = sorted(config_dir.glob("*.json"))
    if args.experiment:
        requested = set(args.experiment)
        config_paths = [path for path in config_paths if load_config(path)["experiment_name"] in requested]
    if not config_paths:
        raise FileNotFoundError(f"No se encontraron JSON de experimentos en: {config_dir}")

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    print(f"Experimentos detectados: {len(config_paths)}")

    for config_path in config_paths:
        config = load_config(config_path)
        dataset_dir = dataset_dir_from_config(config)
        experiment_name = str(config["experiment_name"])
        images_dir = dataset_dir / "images"
        dense_dir = dataset_dir / args.output_name

        print(f"\n=== Experimento: {experiment_name} ===")
        print(f"Config: {config_path}")
        print(f"Dataset: {dataset_dir}")

        if not images_dir.is_dir() or not list_images(images_dir):
            message = f"No hay imagenes renderizadas en {images_dir}"
            if args.keep_going:
                print(f"Error: {message}")
                failed.append(f"{experiment_name}: sin imagenes")
                continue
            raise FileNotFoundError(message)

        dataset_metadata_path = ensure_dataset_metadata(dataset_dir)
        if dataset_metadata_path is not None:
            print(f"Metadatos listos en: {dataset_metadata_path}")

        issues = verify_dense_structure(dense_dir)
        if not issues:
            print(f"Saltado: ya existe un dense en {dense_dir}")
            skipped.append(experiment_name)
            continue

        if dense_dir.exists() and any(dense_dir.iterdir()):
            print(f"Dense incompleto en {dense_dir}, se elimina y se vuelve a generar.")
            shutil.rmtree(dense_dir)

        command = build_command(dataset_dir, args)
        print(subprocess.list2cmdline(command))
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            processed.append(experiment_name)
            continue

        failed.append(f"{experiment_name}: codigo {completed.returncode}")
        if not args.keep_going:
            raise RuntimeError(f"Fallo COLMAP en {experiment_name} con codigo {completed.returncode}")

    processed_label = "Planificados" if args.dry_run else "Procesados"

    print("\nResumen")
    print(f"{processed_label}: {processed if processed else 'ninguno'}")
    print(f"Saltados: {skipped if skipped else 'ninguno'}")
    print(f"Fallidos: {failed if failed else 'ninguno'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
