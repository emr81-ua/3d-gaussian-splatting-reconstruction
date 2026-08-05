"""
reconstruct.py - Full photo -> 3D model pipeline (Gaussian Splatting).

Usage:
    python reconstruct.py <input> [options]

    <input>   A .zip of photos, or a folder containing photos.

Options:
    --output DIR          Output folder (default: ./output/<name>).
    --iter N              3DGS training iterations (default 15000).
    --max-gaussians N     Cap on the number of Gaussians (--max-cap in
                          LichtFeld). Lower it if you run out of VRAM.
                          Default 500000.
    --colmap-exe PATH     Path to the COLMAP executable (auto-detected).
    --lichtfeld-exe PATH  Path to LichtFeld-Studio (auto-detected).
    --skip-training       Run COLMAP only (sparse cloud), skip 3DGS training.

Tool detection order (COLMAP and LichtFeld):
    1. the explicit --*-exe argument
    2. the COLMAP_EXE / LICHTFELD_EXE environment variable
    3. a copy placed under ./tools/
    4. the system PATH

Output:
    <output>/images/          input photos
    <output>/dense/           COLMAP reconstruction (poses + sparse cloud)
    <output>/dense/output/    training splats
    <output>/model.ply        final 3DGS model (copy of the last splat)
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- #
#  Tool discovery
# --------------------------------------------------------------------------- #
def _find_tool(explicit, env_var: str, path_names: list[str], glob_patterns: list[str]) -> Path | None:
    if explicit and Path(explicit).is_file():
        return Path(explicit).resolve()
    env_val = os.environ.get(env_var)
    if env_val and Path(env_val).is_file():
        return Path(env_val).resolve()
    for pattern in glob_patterns:
        for match in sorted(PROJECT_ROOT.glob(pattern)):
            if match.is_file():
                return match.resolve()
    for name in path_names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    return None


def find_colmap(explicit) -> Path:
    tool = _find_tool(
        explicit, "COLMAP_EXE",
        ["colmap", "colmap.exe"],
        ["tools/**/colmap.exe", "tools/**/colmap"],
    )
    if not tool:
        raise FileNotFoundError(
            "COLMAP not found. Install it and add it to PATH, set the COLMAP_EXE "
            "environment variable, or pass --colmap-exe. See the README."
        )
    return tool


def find_lichtfeld(explicit) -> Path:
    tool = _find_tool(
        explicit, "LICHTFELD_EXE",
        ["LichtFeld-Studio", "LichtFeld-Studio.exe"],
        ["tools/**/LichtFeld-Studio.exe", "tools/**/LichtFeld-Studio"],
    )
    if not tool:
        raise FileNotFoundError(
            "LichtFeld-Studio not found. Build/download it, set the LICHTFELD_EXE "
            "environment variable, or pass --lichtfeld-exe. See the README."
        )
    return tool


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


_NO_WINDOW = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW: sin consolas extra (útil desde la GUI)


def run(command: list[str]) -> None:
    print(f"    $ {subprocess.list2cmdline(command)}", flush=True)
    completed = subprocess.run(command, check=False, creationflags=_NO_WINDOW)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def collect_images_into(dest_images: Path, source: Path) -> int:
    dest_images.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in sorted(source.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            target = dest_images / p.name
            if target.exists():
                target = dest_images / f"{p.parent.name}_{p.name}"
            shutil.copy2(p, target)
            count += 1
    return count


def prepare_input(entrada: Path, work_images: Path) -> int:
    if entrada.is_file() and entrada.suffix.lower() == ".zip":
        log(f"Extracting {entrada.name}")
        tmp = work_images.parent / f".unzip_{uuid4().hex[:8]}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(entrada) as zf:
                zf.extractall(tmp)
            return collect_images_into(work_images, tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    elif entrada.is_dir():
        return collect_images_into(work_images, entrada)
    raise FileNotFoundError(f"Input is neither a .zip nor a folder: {entrada}")


def find_sparse_model_dir(mapper_output_dir: Path) -> Path:
    zero_dir = mapper_output_dir / "0"
    if zero_dir.is_dir():
        return zero_dir
    candidates = sorted(p for p in mapper_output_dir.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError("COLMAP mapper produced no sparse model.")
    return candidates[0]


def analyze_sparse(colmap_exe: Path, sparse_dir: Path) -> dict:
    result = subprocess.run(
        [str(colmap_exe), "model_analyzer", "--path", str(sparse_dir)],
        check=False, capture_output=True, text=True, creationflags=_NO_WINDOW,
    )
    text = result.stdout + result.stderr
    metrics: dict = {}
    for key, pat in {
        "registered_images": r"Registered images:\s+(\d+)",
        "points": r"Points:\s+(\d+)",
        "mean_reprojection_error": r"Mean reprojection error:\s+([0-9.]+)px",
    }.items():
        m = re.search(pat, text)
        if m:
            v = m.group(1)
            metrics[key] = int(v) if v.isdigit() else float(v)
    return metrics


# --------------------------------------------------------------------------- #
#  COLMAP  (parameters tuned in the thesis; work for real photos too)
# --------------------------------------------------------------------------- #
def run_colmap(colmap_exe: Path, image_dir: Path, dense_dir: Path,
               peak_threshold: float = 0.004, max_features: int = 16384,
               matcher: str = "exhaustive") -> dict:
    tmp = dense_dir.parent / f".colmap_tmp_{uuid4().hex[:8]}"
    database = tmp / "database.db"
    mapper_out = tmp / "sparse"
    tmp.mkdir(parents=True, exist_ok=True)
    mapper_out.mkdir(parents=True, exist_ok=True)

    try:
        log("COLMAP 1/4 - Feature extraction (SIFT)")
        run([str(colmap_exe), "feature_extractor",
             "--database_path", str(database),
             "--image_path", str(image_dir),
             "--SiftExtraction.peak_threshold", str(peak_threshold),
             "--SiftExtraction.max_num_features", str(max_features)])

        log(f"COLMAP 2/4 - Feature matching ({matcher})")
        run([str(colmap_exe), f"{matcher}_matcher",
             "--database_path", str(database)])

        log("COLMAP 3/4 - Reconstruction (mapper)")
        run([str(colmap_exe), "mapper",
             "--database_path", str(database),
             "--image_path", str(image_dir),
             "--output_path", str(mapper_out),
             "--Mapper.multiple_models", "0",
             "--Mapper.init_min_num_inliers", "30",
             "--Mapper.abs_pose_min_num_inliers", "15",
             "--Mapper.abs_pose_min_inlier_ratio", "0.15"])

        sparse_model = find_sparse_model_dir(mapper_out)

        log("COLMAP 4/4 - Undistort (3DGS-ready format)")
        run([str(colmap_exe), "image_undistorter",
             "--image_path", str(image_dir),
             "--input_path", str(sparse_model),
             "--output_path", str(dense_dir),
             "--output_type", "COLMAP",
             "--copy_policy", "copy"])

        for unwanted in ("frames.bin", "rigs.bin"):
            (dense_dir / "sparse" / unwanted).unlink(missing_ok=True)

        shutil.copy2(database, dense_dir / "database.db")
        return analyze_sparse(colmap_exe, dense_dir / "sparse")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
#  LichtFeld Studio  (3DGS training)
# --------------------------------------------------------------------------- #
def run_training(lichtfeld_exe: Path, dense_dir: Path, iterations: int, max_gaussians: int) -> Path:
    output_dir = dense_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"3DGS - Training in LichtFeld Studio ({iterations} iters, max {max_gaussians:,} Gaussians)")
    run([str(lichtfeld_exe),
         "--data-path", str(dense_dir),
         "--output-path", str(output_dir),
         "--iter", str(iterations),
         "--max-cap", str(max_gaussians),
         "--headless", "--train", "--no-splash"])
    splats = sorted(output_dir.glob("splat_*.ply"))
    if not splats:
        raise RuntimeError("Training finished but no splat_*.ply was produced.")
    return splats[-1]


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full photo -> 3D model pipeline (Gaussian Splatting).")
    parser.add_argument("input", type=Path, help="A .zip of photos, or a folder of photos.")
    parser.add_argument("--output", type=Path, default=None, help="Output folder.")
    parser.add_argument("--iter", type=int, default=15000, help="3DGS training iterations.")
    parser.add_argument("--max-gaussians", type=int, default=500000,
                        help="Cap on Gaussians (LichtFeld --max-cap). Lower it if you run out of VRAM.")
    parser.add_argument("--peak-threshold", type=float, default=0.004,
                        help="SIFT peak_threshold. Lower detects more features (good for low-texture).")
    parser.add_argument("--max-features", type=int, default=16384,
                        help="Max SIFT features per image (more = more detail, slower).")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive",
                        help="Feature matcher: exhaustive (unordered photos) or sequential (video/ordered).")
    parser.add_argument("--colmap-exe", type=Path, default=None)
    parser.add_argument("--lichtfeld-exe", type=Path, default=None)
    parser.add_argument("--skip-training", action="store_true", help="COLMAP only, no 3DGS training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entrada = args.input.resolve()

    name = entrada.stem if entrada.suffix.lower() == ".zip" else entrada.name
    output = (args.output.resolve() if args.output else (PROJECT_ROOT / "output" / name))
    images_dir = output / "images"
    dense_dir = output / "dense"

    colmap_exe = find_colmap(args.colmap_exe)
    print(f"COLMAP:    {colmap_exe}")
    lichtfeld_exe = None
    if not args.skip_training:
        lichtfeld_exe = find_lichtfeld(args.lichtfeld_exe)
        print(f"LichtFeld: {lichtfeld_exe}")

    if dense_dir.exists() and any(dense_dir.iterdir()):
        raise FileExistsError(f"A reconstruction already exists in {dense_dir}. Delete it or use another --output.")

    t0 = datetime.now()

    n = prepare_input(entrada, images_dir)
    if n == 0:
        raise FileNotFoundError("No images found in the input.")
    log(f"{n} photos ready in {images_dir}")

    metrics = run_colmap(colmap_exe, images_dir, dense_dir,
                         peak_threshold=args.peak_threshold,
                         max_features=args.max_features,
                         matcher=args.matcher)
    print(f"    Registered: {metrics.get('registered_images','?')}/{n}  |  "
          f"points: {metrics.get('points','?')}  |  "
          f"reproj. error: {metrics.get('mean_reprojection_error','?')} px")

    if args.skip_training:
        log("Done (COLMAP only).")
        return

    splat = run_training(lichtfeld_exe, dense_dir, args.iter, args.max_gaussians)
    final_model = output / "model.ply"
    shutil.copy2(splat, final_model)

    dt = str(datetime.now() - t0).split(".")[0]
    log("PIPELINE COMPLETE")
    print(f"    3D model: {final_model}")
    print(f"    Splat:    {splat}")
    print(f"    Time:     {dt}")


if __name__ == "__main__":
    main()
