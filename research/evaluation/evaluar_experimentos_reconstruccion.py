from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua en lote la calidad de reconstrucciones de experimentos Blender.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "blender" / "experimentos",
        help="Carpeta con los JSON de experimentos.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "pruebas" / "evaluation_summary.csv",
        help="CSV acumulado con los resultados.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continua aunque falle algun experimento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        raise FileNotFoundError(f"No se encontraron configuraciones en {config_dir}")

    evaluator = PROJECT_ROOT / "scripts" / "evaluation" / "evaluar_reconstruccion.py"
    processed: list[str] = []
    failed: list[str] = []

    for config_path in config_paths:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dataset_dir = (PROJECT_ROOT / config["output_root"] / config["experiment_name"]).resolve()
        print(f"\n=== Evaluando {config['experiment_name']} ===")
        command = [
            sys.executable,
            str(evaluator),
            str(dataset_dir),
            "--config-path",
            str(config_path),
            "--append-csv",
            str(args.csv.resolve()),
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            processed.append(config["experiment_name"])
            continue
        failed.append(config["experiment_name"])
        if not args.keep_going:
            raise RuntimeError(f"Fallo evaluando {config['experiment_name']}")

    print("\nResumen")
    print(f"Procesados: {processed if processed else 'ninguno'}")
    print(f"Fallidos: {failed if failed else 'ninguno'}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
