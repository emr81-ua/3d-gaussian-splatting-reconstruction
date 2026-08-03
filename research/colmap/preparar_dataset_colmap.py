from __future__ import annotations

import argparse
import math
import shutil
import struct
from pathlib import Path


def parse_intrinsics(path: Path) -> dict[str, dict[str, float]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 5 != 0:
        raise ValueError("Intrinsics.cfg no tiene bloques de 5 lineas.")

    intrinsics: dict[str, dict[str, float]] = {}
    for i in range(0, len(lines), 5):
        camera_id = lines[i]
        cx = float(lines[i + 1])
        cy = float(lines[i + 2])
        fx = float(lines[i + 3])
        fy = float(lines[i + 4])
        intrinsics[camera_id] = {"cx": cx, "cy": cy, "fx": fx, "fy": fy}
    return intrinsics


def parse_calibration(path: Path) -> tuple[list[str], dict[str, list[float]]]:
    camera_ids: dict[int, str] = {}
    matrices: dict[int, list[float]] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("##"):
            continue

        if line.startswith("idCam:"):
            key, value = line.split("=", 1)
            index = int(key.split(":")[1])
            camera_ids[index] = value.strip()
        elif line.startswith("matCam:"):
            key, value = line.split("=", 1)
            index = int(key.split(":")[1])
            matrices[index] = [float(token.strip()) for token in value.split(",")]

    if set(camera_ids) != set(matrices):
        raise ValueError("Calibration.cfg no contiene el mismo numero de idCam y matCam.")

    ordered_ids = [camera_ids[i] for i in sorted(camera_ids)]
    ordered_matrices = {camera_ids[i]: matrices[i] for i in sorted(camera_ids)}
    return ordered_ids, ordered_matrices


def mat4_to_rt_colmap(values: list[float]) -> tuple[list[list[float]], list[float]]:
    if len(values) != 16:
        raise ValueError("Cada matriz de camara debe tener 16 valores.")

    # El dataset parece guardar matrices camera-to-world en orden fila.
    r_c2w = [
        [values[0], values[1], values[2]],
        [values[4], values[5], values[6]],
        [values[8], values[9], values[10]],
    ]
    t_c2w = [values[3], values[7], values[11]]

    r_w2c = [
        [r_c2w[0][0], r_c2w[1][0], r_c2w[2][0]],
        [r_c2w[0][1], r_c2w[1][1], r_c2w[2][1]],
        [r_c2w[0][2], r_c2w[1][2], r_c2w[2][2]],
    ]
    t_w2c = [
        -(r_w2c[0][0] * t_c2w[0] + r_w2c[0][1] * t_c2w[1] + r_w2c[0][2] * t_c2w[2]),
        -(r_w2c[1][0] * t_c2w[0] + r_w2c[1][1] * t_c2w[1] + r_w2c[1][2] * t_c2w[2]),
        -(r_w2c[2][0] * t_c2w[0] + r_w2c[2][1] * t_c2w[1] + r_w2c[2][2] * t_c2w[2]),
    ]
    return r_w2c, t_w2c


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


def detect_image_size(image_path: Path) -> tuple[int, int]:
    with image_path.open("rb") as f:
        signature = f.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"La imagen {image_path.name} no es PNG o no tiene una cabecera valida.")
    width = struct.unpack(">I", signature[16:20])[0]
    height = struct.unpack(">I", signature[20:24])[0]
    return width, height


def parse_binary_ply_xyz(path: Path) -> list[tuple[float, float, float]]:
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

    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError("Solo se soporta PLY binary_little_endian 1.0.")

    vertex_count = None
    properties: list[str] = []
    inside_vertex = False
    for line in header_lines:
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
            inside_vertex = True
            continue
        if inside_vertex and line.startswith("element "):
            inside_vertex = False
        if inside_vertex and line.startswith("property "):
            properties.append(line)

    if vertex_count is None:
        raise ValueError("El PLY no define vertices.")

    fmt_parts: list[str] = ["<"]
    xyz_indices: dict[str, int] = {}
    type_sizes = {
        "char": ("b", 1),
        "uchar": ("B", 1),
        "short": ("h", 2),
        "ushort": ("H", 2),
        "int": ("i", 4),
        "uint": ("I", 4),
        "float": ("f", 4),
        "double": ("d", 8),
    }

    for idx, prop in enumerate(properties):
        _, prop_type, prop_name = prop.split()
        if prop_type not in type_sizes:
            raise ValueError(f"Tipo de propiedad no soportado en PLY: {prop_type}")
        fmt_parts.append(type_sizes[prop_type][0])
        if prop_name in {"x", "y", "z"}:
            xyz_indices[prop_name] = idx

    if set(xyz_indices) != {"x", "y", "z"}:
        raise ValueError("No se encontraron las propiedades x, y, z en el PLY.")

    fmt = "".join(fmt_parts)
    stride = struct.calcsize(fmt)
    expected_size = vertex_count * stride
    if len(payload) < expected_size:
        raise ValueError("El payload del PLY es mas corto de lo esperado.")

    points: list[tuple[float, float, float]] = []
    unpack = struct.Struct(fmt).unpack_from
    xi = xyz_indices["x"]
    yi = xyz_indices["y"]
    zi = xyz_indices["z"]
    for offset in range(0, expected_size, stride):
        values = unpack(payload, offset)
        points.append((float(values[xi]), float(values[yi]), float(values[zi])))
    return points


def build_dataset(input_dir: Path, output_dir: Path) -> None:
    intrinsics = parse_intrinsics(input_dir / "Intrinsics.cfg")
    ordered_ids, matrices = parse_calibration(input_dir / "calibration.cfg")

    image_files = sorted(input_dir.glob("*-Color-1-calibrated.png"))
    image_map = {path.name.split("-Color-1-calibrated.png")[0]: path for path in image_files}

    missing_images = [camera_id for camera_id in ordered_ids if camera_id not in image_map]
    missing_intrinsics = [camera_id for camera_id in ordered_ids if camera_id not in intrinsics]
    if missing_images:
        raise ValueError(f"Faltan imagenes para las camaras: {missing_images}")
    if missing_intrinsics:
        raise ValueError(f"Faltan intrinsecos para las camaras: {missing_intrinsics}")

    output_images = output_dir / "images"
    output_sparse = output_dir / "sparse" / "0"
    output_images.mkdir(parents=True, exist_ok=True)
    output_sparse.mkdir(parents=True, exist_ok=True)

    sample_width, sample_height = detect_image_size(image_map[ordered_ids[0]])

    cameras_lines = []
    images_lines = []
    for image_idx, camera_id in enumerate(ordered_ids, start=1):
        src_image = image_map[camera_id]
        dst_image = output_images / src_image.name
        shutil.copy2(src_image, dst_image)

        params = intrinsics[camera_id]
        cameras_lines.append(
            f"{image_idx} PINHOLE {sample_width} {sample_height} "
            f"{params['fx']:.12f} {params['fy']:.12f} {params['cx']:.12f} {params['cy']:.12f}"
        )

        r_w2c, t_w2c = mat4_to_rt_colmap(matrices[camera_id])
        qw, qx, qy, qz = rotation_matrix_to_qvec(r_w2c)
        images_lines.append(
            f"{image_idx} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} "
            f"{t_w2c[0]:.12f} {t_w2c[1]:.12f} {t_w2c[2]:.12f} {image_idx} {src_image.name}"
        )
        images_lines.append("0 0 0")
    points = parse_binary_ply_xyz(input_dir / "newPC.ply")
    points_lines = [
        f"{idx} {x:.9f} {y:.9f} {z:.9f} 255 255 255 0.0"
        for idx, (x, y, z) in enumerate(points, start=1)
    ]

    (output_sparse / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (output_sparse / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (output_sparse / "points3D.txt").write_text("\n".join(points_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte el dataset calibrado del TFG a una carpeta compatible con COLMAP/LichtFeld Studio."
    )
    parser.add_argument("input_dir", type=Path, help="Carpeta con las PNG, CFG y PLY del dataset.")
    parser.add_argument("output_dir", type=Path, help="Carpeta de salida para el dataset COLMAP.")
    args = parser.parse_args()

    build_dataset(args.input_dir.resolve(), args.output_dir.resolve())
    print(f"Dataset COLMAP generado en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
