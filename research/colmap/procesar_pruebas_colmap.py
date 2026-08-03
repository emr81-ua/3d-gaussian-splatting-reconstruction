from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_SCRIPTS_DIR = SCRIPT_DIR.parent / "blender"
if str(BLENDER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_SCRIPTS_DIR))

from experiment_metadata import derive_experiment_metadata, load_metadata, metadata_path, save_metadata, upgrade_metadata
from run_colmap_pipeline import (
    DEFAULT_COLMAP_EXE,
    analyze_sparse_model,
    evaluate_sparse_quality,
    list_images,
    verify_dense_structure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Procesa en lote los datasets de la carpeta pruebas y genera una carpeta dense "
            "utilizable para cada uno."
        )
    )
    parser.add_argument(
        "--pruebas-dir",
        type=Path,
        default=Path(r"C:\Users\emoky\Desktop\Universidad\TFG\pruebas"),
        help="Carpeta raiz que contiene los datasets de pruebas.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Nombre de dataset concreto a procesar. Se puede repetir.",
    )
    parser.add_argument(
        "--colmap-exe",
        type=Path,
        default=DEFAULT_COLMAP_EXE,
        help="Ruta a colmap.exe.",
    )
    parser.add_argument(
        "--camera-model",
        default="PINHOLE",
        help="Modelo de camara para COLMAP.",
    )
    parser.add_argument(
        "--no-single-camera",
        dest="single_camera",
        action="store_false",
        default=True,
        help="Permite intrinsecos distintos por imagen.",
    )
    parser.add_argument(
        "--no-gpu",
        dest="use_gpu",
        action="store_false",
        default=True,
        help="Fuerza CPU en SIFT.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=3200,
        help="Tamano maximo para SIFT extraction.",
    )
    parser.add_argument(
        "--undistort-max-image-size",
        type=int,
        default=-1,
        help="Tamano maximo para image_undistorter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra lo que se ejecutaria sin lanzar COLMAP.",
    )
    parser.add_argument(
        "--output-name",
        default="dense",
        help="Nombre de la carpeta de salida dentro de cada dataset.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Si un dataset falla, continua con los siguientes y resume al final.",
    )
    return parser.parse_args()


def find_dataset_dirs(pruebas_dir: Path, requested_names: list[str]) -> list[Path]:
    if requested_names:
        dataset_dirs = [pruebas_dir / name for name in requested_names]
    else:
        dataset_dirs = sorted(path for path in pruebas_dir.iterdir() if path.is_dir())

    valid_dirs: list[Path] = []
    for dataset_dir in dataset_dirs:
        images_dir = dataset_dir / "images"
        if images_dir.is_dir() and list_images(images_dir):
            valid_dirs.append(dataset_dir)
    return valid_dirs


def dense_is_ready(dense_dir: Path, colmap_exe: Path, input_image_count: int) -> bool:
    if not dense_dir.is_dir() or verify_dense_structure(dense_dir):
        return False
    sparse_dir = dense_dir / "sparse"
    try:
        metrics = analyze_sparse_model(colmap_exe, sparse_dir)
    except Exception:
        return False
    is_usable, _ = evaluate_sparse_quality(metrics, input_image_count)
    return is_usable


def dense_is_script_output(dense_dir: Path) -> bool:
    return (dense_dir / "colmap_run_summary.json").is_file()


def load_experiment_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def discover_config_path(dataset_dir: Path) -> Path | None:
    experiments_dir = Path(__file__).resolve().parent.parent / "blender" / "experimentos"
    exact_match = experiments_dir / f"{dataset_dir.name}.json"
    if exact_match.is_file():
        return exact_match

    image_count = len(list_images(dataset_dir / "images"))
    candidates: list[Path] = []
    for config_path in sorted(experiments_dir.glob("*.json")):
        config = load_experiment_config(config_path)
        expected = int(config["num_cameras_per_row"]) * len(config["vertical_offsets"])
        if expected == image_count:
            candidates.append(config_path)

    if len(candidates) == 1:
        return candidates[0]
    return None


def ensure_dataset_metadata(dataset_dir: Path) -> Path | None:
    current_metadata_path = metadata_path(dataset_dir)
    if current_metadata_path.is_file():
        current_metadata = load_metadata(dataset_dir)
        if current_metadata is not None:
            upgraded_metadata = upgrade_metadata(current_metadata)
            if upgraded_metadata != current_metadata:
                save_metadata(dataset_dir, upgraded_metadata)
        return current_metadata_path

    config_path = discover_config_path(dataset_dir)
    if config_path is None:
        return None

    config = load_experiment_config(config_path)
    metadata = derive_experiment_metadata(config, dataset_dir, source_config=str(config_path.resolve()))
    save_metadata(dataset_dir, metadata)
    return current_metadata_path


def build_command(dataset_dir: Path, args: argparse.Namespace) -> list[str]:
    images_dir = dataset_dir / "images"
    dense_dir = dataset_dir / args.output_name
    masks_dir = dataset_dir / "masks"
    script_path = Path(__file__).resolve().parent / "run_colmap_pipeline.py"

    command = [
        sys.executable,
        str(script_path),
        str(images_dir),
        "--output-dir",
        str(dense_dir),
        "--colmap-exe",
        str(args.colmap_exe.resolve()),
        "--camera-model",
        args.camera_model,
        "--max-image-size",
        str(args.max_image_size),
        "--undistort-max-image-size",
        str(args.undistort_max_image_size),
    ]

    if args.dry_run:
        command.append("--dry-run")
    if not args.single_camera:
        command.append("--no-single-camera")
    if not args.use_gpu:
        command.append("--no-gpu")
    if masks_dir.is_dir():
        command.extend(["--mask-path", str(masks_dir)])

    return command


def main() -> None:
    args = parse_args()
    pruebas_dir = args.pruebas_dir.resolve()

    if not pruebas_dir.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de pruebas: {pruebas_dir}")

    dataset_dirs = find_dataset_dirs(pruebas_dir, args.dataset)
    if not dataset_dirs:
        raise FileNotFoundError(f"No se encontraron datasets procesables en: {pruebas_dir}")

    print(f"Datasets detectados: {len(dataset_dirs)}")

    processed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for dataset_dir in dataset_dirs:
        dense_dir = dataset_dir / args.output_name
        print(f"\n=== Dataset: {dataset_dir.name} ===")

        dataset_metadata_path = ensure_dataset_metadata(dataset_dir)
        if dataset_metadata_path is not None:
            print(f"Metadatos listos en: {dataset_metadata_path}")
        else:
            print("Aviso: no se han podido inferir metadatos de captura para este dataset.")

        input_image_count = len(list_images(dataset_dir / "images"))

        if dense_is_ready(dense_dir, args.colmap_exe.resolve(), input_image_count):
            print(f"Saltado: ya existe un dense valido en {dense_dir}")
            skipped.append(dataset_dir.name)
            continue

        if dense_dir.exists() and any(dense_dir.iterdir()):
            if dense_is_script_output(dense_dir):
                print(f"Se sustituye una salida previa del script que no alcanza la calidad minima: {dense_dir}")
                shutil.rmtree(dense_dir)
            else:
                message = (
                    f"Existe una carpeta dense no vacia pero no valida en {dense_dir}. "
                    "No la voy a tocar automaticamente."
                )
                if args.keep_going:
                    print(f"Error: {message}")
                    failed.append(f"{dataset_dir.name}: dense existente no valido")
                    continue
                raise FileExistsError(message)

        command = build_command(dataset_dir, args)
        print(subprocess.list2cmdline(command))

        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            processed.append(dataset_dir.name)
            continue

        failed.append(f"{dataset_dir.name}: codigo {completed.returncode}")
        if not args.keep_going:
            raise RuntimeError(f"Fallo COLMAP en {dataset_dir.name} con codigo {completed.returncode}")

    processed_label = "Planificados" if args.dry_run else "Procesados"

    print("\nResumen")
    print(f"{processed_label}: {processed if processed else 'ninguno'}")
    print(f"Saltados: {skipped if skipped else 'ninguno'}")
    print(f"Fallidos: {failed if failed else 'ninguno'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
