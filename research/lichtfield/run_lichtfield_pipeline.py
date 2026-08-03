from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LICHTFELD_EXE = Path(
    r"C:\Users\emoky\Desktop\Universidad\TFG\herramientas\LichtFeld-Studio\build\LichtFeld-Studio.exe"
)
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "scripts" / "blender" / "experimentos"
CONFIG_LOG_PATH = PROJECT_ROOT / "lichtfield_configs_log.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bloque 3 del pipeline: entrena LichtFeld Studio en modo headless para cada experimento."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Carpeta con los JSON de experimentos.",
    )
    parser.add_argument(
        "--lichtfeld-exe",
        type=Path,
        default=DEFAULT_LICHTFELD_EXE,
        help="Ruta a LichtFeld-Studio.exe.",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=15000,
        help="Numero de iteraciones de entrenamiento (default: 15000; usa 30000 para calidad final).",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Nombre concreto de experimento a entrenar. Se puede repetir.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Si un experimento falla, continua con los siguientes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Vuelve a entrenar aunque ya exista un splat en la carpeta de salida.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8"))


def dataset_dir_from_config(config: dict) -> Path:
    return (PROJECT_ROOT / config.get("output_root", "pruebas") / config["experiment_name"]).resolve()


def latest_splat(output_dir: Path) -> Path | None:
    splats = sorted(output_dir.glob("splat_*.ply"))
    return splats[-1] if splats else None


def append_run_log(
    lichtfeld_exe: Path,
    iterations: int,
    experiment_results: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"FECHA/HORA : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"LICHTFELD  : {lichtfeld_exe}")
    lines.append(f"ITERACIONES: {iterations}")
    lines.append("")
    lines.append(f"{'EXPERIMENTO':<30} {'RESULTADO':<12} SALIDA")
    lines.append("-" * 72)
    for r in experiment_results:
        lines.append(f"{r['name']:<30} {r['result']:<12} {r.get('splat', r.get('note', ''))}")
    lines.append("")

    with CONFIG_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Configuracion guardada en: {CONFIG_LOG_PATH}")


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    lichtfeld_exe = args.lichtfeld_exe.resolve()

    if not config_dir.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de configuraciones: {config_dir}")
    if not lichtfeld_exe.is_file():
        raise FileNotFoundError(f"No existe LichtFeld-Studio.exe en: {lichtfeld_exe}")

    config_paths = sorted(config_dir.glob("*.json"))
    if args.experiment:
        requested = set(args.experiment)
        config_paths = [p for p in config_paths if load_config(p)["experiment_name"] in requested]
    if not config_paths:
        raise FileNotFoundError(f"No se encontraron JSON de experimentos en: {config_dir}")

    trained: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    experiment_results: list[dict] = []

    print(f"Experimentos detectados: {len(config_paths)}")
    print(f"Iteraciones: {args.iter}")

    for config_path in config_paths:
        config = load_config(config_path)
        experiment_name = config["experiment_name"]
        dataset_dir = dataset_dir_from_config(config)
        dense_dir = dataset_dir / "dense"
        output_dir = dense_dir / "output"

        print(f"\n=== Experimento: {experiment_name} ===")

        if not dense_dir.is_dir():
            msg = f"No existe dense en {dense_dir}. Ejecuta primero run_colmap_bloque2.bat."
            if args.keep_going:
                print(f"Saltado: {msg}")
                skipped.append(f"{experiment_name}: sin dense")
                experiment_results.append({"name": experiment_name, "result": "SALTADO", "note": "sin dense"})
                continue
            raise FileNotFoundError(msg)

        splat = latest_splat(output_dir)
        if splat is not None and not args.force:
            print(f"Saltado: ya existe {splat.name} en {output_dir}")
            skipped.append(experiment_name)
            experiment_results.append({"name": experiment_name, "result": "SALTADO", "splat": str(splat)})
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            str(lichtfeld_exe),
            "--data-path", str(dense_dir),
            "--output-path", str(output_dir),
            "--iter", str(args.iter),
            "--headless",
            "--train",
            "--no-splash",
        ]

        print(subprocess.list2cmdline(command))
        completed = subprocess.run(command, check=False)

        if completed.returncode == 0:
            splat = latest_splat(output_dir)
            if splat:
                print(f"Entrenamiento completado. Splat: {splat.name}")
                trained.append(experiment_name)
                experiment_results.append({"name": experiment_name, "result": "ENTRENADO", "splat": str(splat)})
            else:
                print("Entrenamiento completado (no se encontro splat_*.ply en la salida).")
                trained.append(experiment_name)
                experiment_results.append({"name": experiment_name, "result": "ENTRENADO", "note": "sin splat_*.ply"})
        else:
            failed.append(f"{experiment_name}: codigo {completed.returncode}")
            experiment_results.append({"name": experiment_name, "result": "FALLIDO", "note": f"codigo {completed.returncode}"})
            if not args.keep_going:
                append_run_log(lichtfeld_exe, args.iter, experiment_results)
                raise RuntimeError(f"LichtFeld fallo en {experiment_name} con codigo {completed.returncode}")

    print("\nResumen")
    print(f"Entrenados: {trained if trained else 'ninguno'}")
    print(f"Saltados:   {skipped if skipped else 'ninguno'}")
    print(f"Fallidos:   {failed if failed else 'ninguno'}")

    append_run_log(lichtfeld_exe, args.iter, experiment_results)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
