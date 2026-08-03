from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0
WORLD_UP = (0.0, 0.0, 1.0)
METADATA_VERSION = 5


def normalize(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(component * component for component in vec))
    if norm == 0.0:
        raise ValueError("No se puede normalizar un vector nulo.")
    return tuple(component / norm for component in vec)


def cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def subtract(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def rotation_matrix_to_qvec(r: list[list[float]]) -> tuple[float, float, float, float]:
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2][1] - r[1][2]) / s
        qy = (r[0][2] - r[2][0]) / s
        qz = (r[1][0] - r[0][1]) / s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        qw = (r[2][1] - r[1][2]) / s
        qx = 0.25 * s
        qy = (r[0][1] + r[1][0]) / s
        qz = (r[0][2] + r[2][0]) / s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        qw = (r[0][2] - r[2][0]) / s
        qx = (r[0][1] + r[1][0]) / s
        qy = 0.25 * s
        qz = (r[1][2] + r[2][1]) / s
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        qw = (r[1][0] - r[0][1]) / s
        qx = (r[0][2] + r[2][0]) / s
        qy = (r[1][2] + r[2][1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / norm, qx / norm, qy / norm, qz / norm


def look_at_rotation(camera_position: tuple[float, float, float],
                     target_position: tuple[float, float, float]) -> list[list[float]]:
    # COLMAP convention: +Z toward scene, +Y down, +X right
    forward = normalize(subtract(target_position, camera_position))
    z_axis = forward
    x_axis = normalize(cross(z_axis, WORLD_UP))
    y_axis = normalize(cross(z_axis, x_axis))
    return [
        [x_axis[0], y_axis[0], z_axis[0]],
        [x_axis[1], y_axis[1], z_axis[1]],
        [x_axis[2], y_axis[2], z_axis[2]],
    ]


def rotation_c2w_to_w2c(rotation_c2w: list[list[float]]) -> list[list[float]]:
    return [
        [rotation_c2w[0][0], rotation_c2w[1][0], rotation_c2w[2][0]],
        [rotation_c2w[0][1], rotation_c2w[1][1], rotation_c2w[2][1]],
        [rotation_c2w[0][2], rotation_c2w[1][2], rotation_c2w[2][2]],
    ]


def translation_w2c(rotation_w2c: list[list[float]],
                    translation_c2w: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        -dot(tuple(rotation_w2c[0]), translation_c2w),
        -dot(tuple(rotation_w2c[1]), translation_c2w),
        -dot(tuple(rotation_w2c[2]), translation_c2w),
    )


def derive_intrinsics(config: dict[str, Any]) -> dict[str, float]:
    # Blender renders tienen píxeles cuadrados con sensor fit=Auto.
    # Blender usa sensor_width para determinar la focal en píxeles en ambos ejes,
    # por lo que fx = fy = lens_mm / sensor_width_mm * resolution_x.
    width = int(config["resolution_x"])
    height = int(config["resolution_y"])
    f_pix = float(config["lens_mm"]) / SENSOR_WIDTH_MM * width
    return {
        "width": width,
        "height": height,
        "fx": f_pix,
        "fy": f_pix,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "sensor_width_mm": SENSOR_WIDTH_MM,
    }


def build_camera_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    num_cameras_per_row = int(config["num_cameras_per_row"])
    radius = float(config["radius"])
    vertical_offsets = config["vertical_offsets"]
    target_heights = config["target_heights"]

    cam_index = 0
    for row, height in enumerate(vertical_offsets):
        target_z = float(target_heights[row])
        for column in range(num_cameras_per_row):
            angle = (column / num_cameras_per_row) * 2.0 * math.pi
            camera_position = (
                radius * math.cos(angle),
                radius * math.sin(angle),
                float(height),
            )
            target_position = (0.0, 0.0, target_z)
            rotation_c2w = look_at_rotation(camera_position, target_position)
            rotation_w2c = rotation_c2w_to_w2c(rotation_c2w)
            translation = translation_w2c(rotation_w2c, camera_position)
            qw, qx, qy, qz = rotation_matrix_to_qvec(rotation_w2c)

            records.append(
                {
                    "image_name": f"Camera_{cam_index:02d}.png",
                    "camera_name": f"Camera_{cam_index:02d}",
                    "row": row,
                    "column": column,
                    "angle_deg": math.degrees(angle),
                    "position": [round(value, 12) for value in camera_position],
                    "target": [round(value, 12) for value in target_position],
                    "qvec": [round(qw, 12), round(qx, 12), round(qy, 12), round(qz, 12)],
                    "tvec": [round(translation[0], 12), round(translation[1], 12), round(translation[2], 12)],
                }
            )
            cam_index += 1

    return records


def recommended_colmap_settings(camera_count: int) -> dict[str, Any]:
    # Los renders de Blender tienen menos textura que fotos reales.
    # Necesitan extraer más features (peak_threshold bajo, más features máximas)
    # y umbrales de mapper más permisivos para poder inicializar y registrar cámaras.
    return {
        "use_known_poses": False,
        "feature_extractor": {
            "camera_model": "PINHOLE",
            "single_camera": True,
            "use_gpu": True,
            "max_image_size": 3200,
            "max_num_features": 16384,
            "peak_threshold": 0.004,
            "estimate_affine_shape": True,
            "domain_size_pooling": True,
        },
        "matcher": {
            "guided_matching": True,
            "max_ratio": 0.85,
        },
        "triangulator": {
            "clear_points": True,
            "refine_intrinsics": False,
            "min_num_matches": 5,
            "tri_min_angle": 0.5,
            "ignore_two_view_tracks": False,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
        },
        "mapper_fallback": {
            "multiple_models": False,
            "init_min_num_inliers": 30,
            "abs_pose_min_num_inliers": 15,
            "abs_pose_min_inlier_ratio": 0.15,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
        },
    }


def derive_experiment_metadata(config: dict[str, Any], dataset_dir: Path, *,
                               source_config: str | None = None) -> dict[str, Any]:
    intrinsics = derive_intrinsics(config)
    cameras = build_camera_records(config)
    metadata = {
        "metadata_version": METADATA_VERSION,
        "experiment_name": config["experiment_name"],
        "dataset_dir": str(dataset_dir),
        "source_config": source_config,
        "pose_source": "inferred_from_config",
        "capture": {
            "model_name": config["model_name"],
            "output_root": config["output_root"],
            "num_cameras_per_row": int(config["num_cameras_per_row"]),
            "radius": float(config["radius"]),
            "vertical_offsets": list(config["vertical_offsets"]),
            "target_heights": list(config["target_heights"]),
            "lens_mm": float(config["lens_mm"]),
            "resolution_x": int(config["resolution_x"]),
            "resolution_y": int(config["resolution_y"]),
            "camera_count": len(cameras),
        },
        "intrinsics": intrinsics,
        "cameras": cameras,
        "colmap": recommended_colmap_settings(len(cameras)),
    }
    return metadata


def upgrade_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(metadata)
    capture = upgraded.get("capture", {})
    camera_count = int(capture.get("camera_count") or len(upgraded.get("cameras", [])))
    if all(key in capture for key in ("resolution_x", "resolution_y", "lens_mm")):
        intrinsics_config = {
            "resolution_x": int(capture["resolution_x"]),
            "resolution_y": int(capture["resolution_y"]),
            "lens_mm": float(capture["lens_mm"]),
        }
        upgraded["intrinsics"] = derive_intrinsics(intrinsics_config)
    if all(key in capture for key in ("num_cameras_per_row", "radius", "vertical_offsets", "target_heights")):
        upgraded["cameras"] = build_camera_records(capture)
        camera_count = len(upgraded["cameras"])
    if camera_count > 0:
        upgraded["colmap"] = recommended_colmap_settings(camera_count)
    upgraded["metadata_version"] = METADATA_VERSION
    return upgraded


def metadata_path(dataset_dir: Path) -> Path:
    return dataset_dir / "experiment_metadata.json"


def load_metadata(dataset_dir: Path) -> dict[str, Any] | None:
    path = metadata_path(dataset_dir)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_metadata(dataset_dir: Path, metadata: dict[str, Any]) -> Path:
    path = metadata_path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path
