from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_SCRIPTS_DIR = SCRIPT_DIR.parent / "blender"
if str(BLENDER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_SCRIPTS_DIR))

from experiment_metadata import derive_experiment_metadata, load_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLENDER_EXE = Path(r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe")
DEFAULT_COLMAP_EXE = Path(
    r"C:\Users\emoky\Desktop\Universidad\TFG\herramientas\colmap-x64-windows-cuda\bin\colmap.exe"
)


@dataclass
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua objetivamente la calidad de una reconstruccion 3D del pipeline."
    )
    parser.add_argument("dataset_dir", type=Path, help="Carpeta del experimento dentro de pruebas.")
    parser.add_argument("--config-path", type=Path, default=None, help="JSON del experimento de Blender.")
    parser.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER_EXE, help="Ruta a blender.exe.")
    parser.add_argument("--colmap-exe", type=Path, default=DEFAULT_COLMAP_EXE, help="Ruta a colmap.exe.")
    parser.add_argument("--gt-sample-count", type=int, default=20000, help="Puntos GT a muestrear del modelo.")
    parser.add_argument("--output-json", type=Path, default=None, help="Ruta del informe JSON.")
    parser.add_argument("--append-csv", type=Path, default=None, help="CSV acumulado de resultados.")
    return parser.parse_args()


def discover_config_path(dataset_dir: Path, metadata: dict | None, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if metadata and metadata.get("source_config"):
        candidate = Path(metadata["source_config"])
        if candidate.is_file():
            return candidate.resolve()
    experiments_dir = PROJECT_ROOT / "scripts" / "blender" / "experimentos"
    candidate = experiments_dir / f"{dataset_dir.name}.json"
    if candidate.is_file():
        return candidate.resolve()
    images_dir = dataset_dir / "images"
    image_count = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.jpeg")))
    matching_candidates: list[Path] = []
    for config_path in sorted(experiments_dir.glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        expected_count = int(config["num_cameras_per_row"]) * len(config["vertical_offsets"])
        if expected_count == image_count:
            matching_candidates.append(config_path.resolve())
    if len(matching_candidates) == 1:
        return matching_candidates[0]
    raise FileNotFoundError("No se ha podido localizar el JSON del experimento para evaluar contra el modelo GT.")


def choose_sparse_dir(dense_dir: Path) -> Path:
    sparse_dir = dense_dir / "sparse"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"No existe sparse en {dense_dir}")
    if (sparse_dir / "images.bin").is_file():
        return sparse_dir
    zero_dir = sparse_dir / "0"
    if (zero_dir / "images.bin").is_file():
        return zero_dir
    raise FileNotFoundError(f"No se encontro un modelo sparse valido en {sparse_dir}")


def read_c_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.index(b"\x00", offset)
    return data[offset:end].decode("utf-8"), end + 1


def read_colmap_images_bin(path: Path) -> list[dict]:
    data = path.read_bytes()
    offset = 0
    num_images = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    images: list[dict] = []

    for _ in range(num_images):
        image_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        qvec = struct.unpack_from("<4d", data, offset)
        offset += 32
        tvec = struct.unpack_from("<3d", data, offset)
        offset += 24
        camera_id = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        name, offset = read_c_string(data, offset)
        num_points2d = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + int(num_points2d) * 24
        images.append(
            {
                "image_id": image_id,
                "camera_id": camera_id,
                "name": name,
                "qvec": np.array(qvec, dtype=np.float64),
                "tvec": np.array(tvec, dtype=np.float64),
            }
        )
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
            [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
            [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def camera_center_from_image(image_record: dict) -> np.ndarray:
    rotation = qvec_to_rotmat(image_record["qvec"])
    return -rotation.T @ image_record["tvec"]


def read_colmap_points3d_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    offset = 0
    num_points = struct.unpack_from("<Q", data, offset)[0]
    offset += 8
    points = np.empty((num_points, 3), dtype=np.float64)

    for i in range(num_points):
        offset += 8  # point3D_id
        xyz = struct.unpack_from("<3d", data, offset)
        offset += 24
        points[i] = xyz
        offset += 3  # rgb
        offset += 8  # error
        track_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + int(track_length) * 8
    return points


def read_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY incompleto.")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        payload = f.read()

    vertex_count = None
    properties: list[str] = []
    is_binary = False
    inside_vertex = False
    for line in header_lines:
        if line.startswith("format binary_little_endian"):
            is_binary = True
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            inside_vertex = True
            continue
        if inside_vertex and line.startswith("element "):
            inside_vertex = False
        if inside_vertex and line.startswith("property "):
            properties.append(line)
    if not is_binary or vertex_count is None:
        raise ValueError("Solo se soportan PLY binary_little_endian con vertices.")

    type_map = {
        "char": ("b", 1),
        "uchar": ("B", 1),
        "short": ("h", 2),
        "ushort": ("H", 2),
        "int": ("i", 4),
        "uint": ("I", 4),
        "float": ("f", 4),
        "double": ("d", 8),
    }
    fmt_parts = ["<"]
    xyz_indices: dict[str, int] = {}
    for idx, prop in enumerate(properties):
        _, prop_type, prop_name = prop.split()
        symbol, _ = type_map[prop_type]
        fmt_parts.append(symbol)
        if prop_name in {"x", "y", "z"}:
            xyz_indices[prop_name] = idx
    fmt = "".join(fmt_parts)
    stride = struct.calcsize(fmt)
    unpack = struct.Struct(fmt).unpack_from
    points = np.empty((vertex_count, 3), dtype=np.float64)
    xi, yi, zi = xyz_indices["x"], xyz_indices["y"], xyz_indices["z"]
    for i in range(vertex_count):
        values = unpack(payload, i * stride)
        points[i] = (float(values[xi]), float(values[yi]), float(values[zi]))
    return points


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / src.shape[0]
    u, singular_values, vt = np.linalg.svd(covariance)
    s_matrix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_matrix[-1, -1] = -1
    rotation = u @ s_matrix @ vt
    variance = np.mean(np.sum(src_centered * src_centered, axis=1))
    scale = float(np.trace(np.diag(singular_values) @ s_matrix) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation)


def nearest_neighbor_metrics(pred_points: np.ndarray, gt_points: np.ndarray) -> dict[str, float]:
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    pred_to_gt = gt_tree.query(pred_points, k=1)[0]
    gt_to_pred = pred_tree.query(gt_points, k=1)[0]

    chamfer = float(pred_to_gt.mean() + gt_to_pred.mean())
    return {
        "pred_to_gt_mean": float(pred_to_gt.mean()),
        "pred_to_gt_rmse": float(np.sqrt(np.mean(pred_to_gt ** 2))),
        "pred_to_gt_p90": float(np.percentile(pred_to_gt, 90)),
        "pred_to_gt_p95": float(np.percentile(pred_to_gt, 95)),
        "pred_to_gt_max": float(pred_to_gt.max()),
        "gt_to_pred_mean": float(gt_to_pred.mean()),
        "gt_to_pred_p90": float(np.percentile(gt_to_pred, 90)),
        "gt_to_pred_p95": float(np.percentile(gt_to_pred, 95)),
        "gt_to_pred_max": float(gt_to_pred.max()),
        "chamfer_l1": chamfer,
    }


def quality_score(metrics: dict) -> float:
    reg = float(metrics["colmap"]["registered_ratio"])
    reproj = float(metrics["colmap"]["mean_reprojection_error_px"])
    geom = float(metrics["geometry"]["pred_to_gt_p90_normalized"])

    reg_score = max(0.0, min(1.0, reg))
    reproj_score = max(0.0, min(1.0, 1.0 - reproj / 3.0))
    geom_score = max(0.0, min(1.0, 1.0 - geom / 0.2))
    return round(100.0 * (0.4 * reg_score + 0.2 * reproj_score + 0.4 * geom_score), 2)


def parse_model_analyzer_output(output_text: str) -> dict[str, float]:
    patterns = {
        "registered_images": r"Registered images:\s+(\d+)",
        "points": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([0-9.]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([0-9.]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([0-9.]+)px",
    }
    result: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text)
        if not match:
            raise RuntimeError(f"No se pudo leer la metrica '{key}' del model_analyzer.")
        value = match.group(1)
        result[key] = float(value) if "." in value else int(value)
    return result


def run_model_analyzer(colmap_exe: Path, sparse_dir: Path) -> dict[str, float]:
    completed = subprocess.run(
        [str(colmap_exe), "model_analyzer", "--path", str(sparse_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    return parse_model_analyzer_output(completed.stdout + "\n" + completed.stderr)


def ensure_gt_points(dataset_dir: Path, config_path: Path, blender_exe: Path, sample_count: int) -> Path:
    eval_dir = dataset_dir / "evaluation"
    gt_path = eval_dir / f"gt_surface_points_{sample_count}.npz"
    if gt_path.is_file():
        return gt_path

    script_path = PROJECT_ROOT / "scripts" / "blender" / "export_gt_surface_points_blender.py"
    command = [
        str(blender_exe),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--config-path",
        str(config_path),
        "--output-path",
        str(gt_path),
        "--sample-count",
        str(sample_count),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Fallo exportando puntos GT desde Blender:\n" + completed.stdout + "\n" + completed.stderr
        )
    return gt_path


def latest_splat_ply(dense_dir: Path) -> Path | None:
    output_dir = dense_dir / "output"
    if not output_dir.is_dir():
        return None
    splats = sorted(output_dir.glob("splat_*.ply"))
    return splats[-1] if splats else None


def append_csv(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    dense_dir = dataset_dir / "dense"
    metadata = load_metadata(dataset_dir)
    config_path = discover_config_path(dataset_dir, metadata, args.config_path)
    if metadata is None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metadata = derive_experiment_metadata(config, dataset_dir, source_config=str(config_path))
    sparse_dir = choose_sparse_dir(dense_dir)

    gt_npz_path = ensure_gt_points(dataset_dir, config_path, args.blender_exe.resolve(), args.gt_sample_count)
    gt_points = np.load(gt_npz_path)["points"]
    model_scale = float(np.max(np.ptp(gt_points, axis=0)))

    sparse_images = read_colmap_images_bin(sparse_dir / "images.bin")
    sparse_points = read_colmap_points3d_bin(sparse_dir / "points3D.bin")

    if not metadata or not metadata.get("cameras"):
        raise RuntimeError("Faltan metadatos de camaras GT para poder alinear la reconstruccion.")

    gt_camera_map = {
        camera["image_name"]: np.array(camera["position"], dtype=np.float64)
        for camera in metadata["cameras"]
    }
    matched_pred = []
    matched_gt = []
    for image_record in sparse_images:
        if image_record["name"] in gt_camera_map:
            matched_pred.append(camera_center_from_image(image_record))
            matched_gt.append(gt_camera_map[image_record["name"]])
    if len(matched_pred) < 3:
        raise RuntimeError("No hay suficientes camaras emparejadas para alinear la reconstruccion.")

    matched_pred_arr = np.vstack(matched_pred)
    matched_gt_arr = np.vstack(matched_gt)
    transform = umeyama_similarity(matched_pred_arr, matched_gt_arr)
    aligned_sparse_points = transform.apply(sparse_points)
    aligned_camera_centers = transform.apply(matched_pred_arr)
    camera_alignment_rmse = float(np.sqrt(np.mean(np.sum((aligned_camera_centers - matched_gt_arr) ** 2, axis=1))))

    geometry = nearest_neighbor_metrics(aligned_sparse_points, gt_points)
    geometry["pred_to_gt_mean_normalized"] = geometry["pred_to_gt_mean"] / model_scale
    geometry["pred_to_gt_p90_normalized"] = geometry["pred_to_gt_p90"] / model_scale
    geometry["chamfer_l1_normalized"] = geometry["chamfer_l1"] / model_scale

    colmap_metrics = run_model_analyzer(args.colmap_exe.resolve(), sparse_dir)
    colmap_metrics["registered_ratio"] = float(colmap_metrics["registered_images"]) / float(len(gt_camera_map))
    colmap_metrics["camera_alignment_rmse"] = camera_alignment_rmse
    colmap_metrics["camera_alignment_rmse_normalized"] = camera_alignment_rmse / model_scale

    result = {
        "dataset_dir": str(dataset_dir),
        "config_path": str(config_path),
        "model_name": metadata["capture"]["model_name"] if metadata else None,
        "input_image_count": len(gt_camera_map),
        "gt_sample_count": int(gt_points.shape[0]),
        "reference_model_scale": model_scale,
        "colmap": colmap_metrics,
        "geometry": geometry,
        "alignment": {
            "scale": transform.scale,
            "rotation": transform.rotation.tolist(),
            "translation": transform.translation.tolist(),
        },
        "artifacts": {
            "sparse_dir": str(sparse_dir),
            "gt_points_npz": str(gt_npz_path),
            "lfs_splat_ply": str(latest_splat_ply(dense_dir)) if latest_splat_ply(dense_dir) else None,
        },
    }

    splat_path = latest_splat_ply(dense_dir)
    if splat_path is not None:
        splat_points = read_ply_xyz(splat_path)
        aligned_splat_points = transform.apply(splat_points)
        lfs_geometry = nearest_neighbor_metrics(aligned_splat_points, gt_points)
        lfs_geometry["pred_to_gt_mean_normalized"] = lfs_geometry["pred_to_gt_mean"] / model_scale
        lfs_geometry["pred_to_gt_p90_normalized"] = lfs_geometry["pred_to_gt_p90"] / model_scale
        lfs_geometry["chamfer_l1_normalized"] = lfs_geometry["chamfer_l1"] / model_scale
        result["lfs_geometry"] = lfs_geometry

    score_source = {
        "colmap": result["colmap"],
        "geometry": result.get("lfs_geometry", result["geometry"]),
    }
    result["quality_score_100"] = quality_score(score_source)

    output_json = args.output_json.resolve() if args.output_json else dataset_dir / "evaluation" / "reconstruction_quality.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    if args.append_csv:
        flat_row = {
            "dataset_dir": str(dataset_dir),
            "quality_score_100": result["quality_score_100"],
            "registered_ratio": result["colmap"]["registered_ratio"],
            "colmap_points": int(result["colmap"]["points"]),
            "mean_reprojection_error_px": result["colmap"]["mean_reprojection_error_px"],
            "camera_alignment_rmse_normalized": result["colmap"]["camera_alignment_rmse_normalized"],
            "sparse_pred_to_gt_p90_normalized": result["geometry"]["pred_to_gt_p90_normalized"],
            "sparse_chamfer_l1_normalized": result["geometry"]["chamfer_l1_normalized"],
            "lfs_pred_to_gt_p90_normalized": result.get("lfs_geometry", {}).get("pred_to_gt_p90_normalized"),
            "lfs_chamfer_l1_normalized": result.get("lfs_geometry", {}).get("chamfer_l1_normalized"),
        }
        append_csv(args.append_csv.resolve(), flat_row)

    print(f"Informe guardado en: {output_json}")
    print(f"Quality score: {result['quality_score_100']:.2f}/100")
    print(f"Registered ratio: {result['colmap']['registered_ratio']:.3f}")
    print(f"Sparse p90 normalizado: {result['geometry']['pred_to_gt_p90_normalized']:.5f}")
    if "lfs_geometry" in result:
        print(f"LFS p90 normalizado: {result['lfs_geometry']['pred_to_gt_p90_normalized']:.5f}")


if __name__ == "__main__":
    main()
