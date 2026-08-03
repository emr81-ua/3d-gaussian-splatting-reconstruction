from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_SCRIPTS_DIR = SCRIPT_DIR.parent / "blender"
if str(BLENDER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_SCRIPTS_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_COLMAP_EXE = Path(
    r"C:\Users\emoky\Desktop\Universidad\TFG\herramientas\colmap-x64-windows-cuda\bin\colmap.exe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bloque 2: ejecuta COLMAP (feature extraction, matching, mapper, undistort) y genera la carpeta dense."
    )
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--colmap-exe", type=Path, default=DEFAULT_COLMAP_EXE)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"\n[COLMAP] {subprocess.list2cmdline(command)}")
    if dry_run:
        return
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"El comando fallo con codigo {completed.returncode}")


def find_sparse_model_dir(mapper_output_dir: Path) -> Path:
    zero_dir = mapper_output_dir / "0"
    if zero_dir.is_dir():
        return zero_dir
    candidates = sorted(p for p in mapper_output_dir.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("COLMAP mapper no genero ningun modelo sparse.")
    return candidates[0]


def verify_dense_structure(output_dir: Path) -> list[str]:
    issues: list[str] = []
    for folder in ("images", "sparse", "stereo"):
        if not (output_dir / folder).is_dir():
            issues.append(f"Falta la carpeta {folder}/")
    sparse = output_dir / "sparse"
    for fname in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse / fname).is_file():
            issues.append(f"Falta sparse/{fname}")
    return issues


def analyze_sparse_model(colmap_exe: Path, sparse_dir: Path) -> dict:
    result = subprocess.run(
        [str(colmap_exe), "model_analyzer", "--path", str(sparse_dir)],
        check=False, capture_output=True, text=True,
    )
    text = result.stdout + result.stderr
    metrics: dict = {}
    for key, pat in {
        "registered_images": r"Registered images:\s+(\d+)",
        "points": r"Points:\s+(\d+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([0-9.]+)px",
        "mean_observations_per_image": r"Mean observations per image:\s+([0-9.]+)",
    }.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            metrics[key] = int(v) if v.isdigit() else float(v)
    return metrics


def evaluate_sparse_quality(metrics: dict, input_image_count: int) -> tuple[bool, list[str]]:
    issues = []
    reg = int(metrics.get("registered_images", 0))
    pts = int(metrics.get("points", 0))
    if reg < input_image_count * 0.85:
        issues.append(f"Solo {reg}/{input_image_count} imagenes registradas")
    if pts < max(200, input_image_count * 25):
        issues.append(f"Solo {pts} puntos sparse (minimo recomendado {max(200, input_image_count * 25)})")
    return not issues, issues


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else image_dir.parent / "dense")
    colmap_exe = args.colmap_exe.resolve()

    if not colmap_exe.is_file():
        raise FileNotFoundError(f"No existe colmap.exe en: {colmap_exe}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de imagenes: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No se encontraron imagenes en: {image_dir}")

    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing:
            raise FileExistsError(f"La carpeta dense ya existe y no esta vacia: {output_dir}")
    elif not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir.parent / f".colmap_tmp_{uuid4().hex[:8]}"
    database_path = tmp_dir / "database.db"
    mapper_output_dir = tmp_dir / "sparse"

    if not args.dry_run:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        mapper_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Paso 1: feature extraction (ajustado para renders sintéticos de Blender)
        run_command([
            str(colmap_exe), "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--SiftExtraction.peak_threshold", "0.004",
            "--SiftExtraction.max_num_features", "16384",
        ], dry_run=args.dry_run)

        # Paso 2: feature matching (defaults de COLMAP)
        run_command([
            str(colmap_exe), "exhaustive_matcher",
            "--database_path", str(database_path),
        ], dry_run=args.dry_run)

        # Paso 3: mapper (ajustado para renders sintéticos de Blender, un solo modelo)
        run_command([
            str(colmap_exe), "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(mapper_output_dir),
            "--Mapper.multiple_models", "0",
            "--Mapper.init_min_num_inliers", "30",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.15",
        ], dry_run=args.dry_run)

        if args.dry_run:
            print(f"\nDry run completado. Salida prevista: {output_dir}")
            return

        sparse_model_dir = find_sparse_model_dir(mapper_output_dir)

        # Paso 4: image_undistorter → genera la carpeta dense
        run_command([
            str(colmap_exe), "image_undistorter",
            "--image_path", str(image_dir),
            "--input_path", str(sparse_model_dir),
            "--output_path", str(output_dir),
            "--output_type", "COLMAP",
            "--copy_policy", "copy",
        ], dry_run=False)

        # Limpiar archivos no estandar que COLMAP deja en sparse/ y que pueden
        # confundir a LichtFeld (frames.bin, rigs.bin no son parte del formato base)
        for unwanted in ["frames.bin", "rigs.bin"]:
            (output_dir / "sparse" / unwanted).unlink(missing_ok=True)

        shutil.copy2(database_path, output_dir / "database.db")

    finally:
        if not args.dry_run and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    issues = verify_dense_structure(output_dir)
    if issues:
        raise RuntimeError("Dense generado con problemas:\n- " + "\n- ".join(issues))

    sparse_metrics = {}
    try:
        sparse_metrics = analyze_sparse_model(colmap_exe, output_dir / "sparse")
    except Exception:
        pass

    is_ok, quality_issues = evaluate_sparse_quality(sparse_metrics, len(images))

    print("\nBloque 2 completado.")
    print(f"Imagenes: {len(images)}")
    print(f"Registradas: {sparse_metrics.get('registered_images', '?')}/{len(images)}")
    print(f"Puntos sparse: {sparse_metrics.get('points', '?')}")
    print(f"Error medio de reproyeccion: {sparse_metrics.get('mean_reprojection_error', '?')} px")
    if quality_issues:
        print("Avisos de calidad:")
        for q in quality_issues:
            print(f"  - {q}")
    print(f"Dense en: {output_dir}")

    (output_dir / "colmap_run_summary.json").write_text(
        json.dumps({
            "timestamp": datetime.now().isoformat(),
            "reconstruction_mode": "mapper",
            "input_image_count": len(images),
            "sparse_metrics": sparse_metrics,
            "quality_issues": quality_issues,
        }, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
