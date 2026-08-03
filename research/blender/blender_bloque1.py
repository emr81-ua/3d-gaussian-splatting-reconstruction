from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_metadata import derive_experiment_metadata, save_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Bloque 1 del pipeline: importa modelo, crea camaras y renderiza imagenes."
    )
    parser.add_argument("--config-path", required=True, help="Ruta al JSON del experimento")
    parser.add_argument("--save-blend", action="store_true", help="Guardar un .blend final junto a las imagenes")
    return parser.parse_args(argv)


def load_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_fields = [
        "experiment_name",
        "model_name",
        "output_root",
        "num_cameras_per_row",
        "radius",
        "vertical_offsets",
        "target_heights",
        "lens_mm",
        "resolution_x",
        "resolution_y",
    ]
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Faltan campos obligatorios en el JSON: {missing}")
    if len(config["vertical_offsets"]) != len(config["target_heights"]):
        raise ValueError("vertical_offsets y target_heights deben tener la misma longitud.")
    return config


def resolve_model_path(model_name: str) -> Path:
    return (PROJECT_ROOT / "herramientas" / "modelos" / model_name).resolve()


def build_output_paths(config: dict) -> tuple[Path, Path]:
    experiment_dir = (PROJECT_ROOT / config["output_root"] / config["experiment_name"]).resolve()
    images_dir = experiment_dir / "images"
    return experiment_dir, images_dir


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for block_collection in (
        bpy.data.meshes,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.actions,
    ):
        for block in list(block_collection):
            if block.users == 0:
                block_collection.remove(block)


def import_fbx(model_path: Path) -> list[bpy.types.Object]:
    before_names = {obj.name for obj in bpy.data.objects}
    bpy.ops.import_scene.fbx(filepath=str(model_path))
    imported = [obj for obj in bpy.data.objects if obj.name not in before_names]
    if not imported:
        raise RuntimeError("No se ha importado ningun objeto desde el FBX.")
    return imported


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        if obj.type not in {"MESH", "ARMATURE", "EMPTY"}:
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)
            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)

    if not math.isfinite(min_corner.x):
        raise RuntimeError("No se pudo calcular el bounding box del modelo.")

    return min_corner, max_corner


def get_root_objects(imported_objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    imported_names = {obj.name for obj in imported_objects}
    roots = [obj for obj in imported_objects if obj.parent is None or obj.parent.name not in imported_names]
    return roots or imported_objects


def center_model(imported_objects: list[bpy.types.Object]) -> dict[str, float]:
    bpy.context.view_layer.update()
    min_corner, max_corner = scene_bounds(imported_objects)
    roots = get_root_objects(imported_objects)

    offset = Vector((
        (min_corner.x + max_corner.x) / 2.0,
        (min_corner.y + max_corner.y) / 2.0,
        min_corner.z,
    ))

    for obj in roots:
        obj.location -= offset

    bpy.context.view_layer.update()
    min_corner, max_corner = scene_bounds(imported_objects)
    dims = max_corner - min_corner
    return {
        "width": float(dims.x),
        "depth": float(dims.y),
        "height": float(dims.z),
        "min_z": float(min_corner.z),
        "max_z": float(max_corner.z),
    }


def setup_lighting(model_info: dict[str, float], radius: float) -> None:
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    bpy.context.scene.world.use_nodes = True

    top_z = max(model_info["height"] + 2.0, 4.0)

    key_data = bpy.data.lights.new(name="KeyLight", type="AREA")
    key_data.energy = 3000
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    bpy.context.collection.objects.link(key_obj)
    key_obj.location = (radius, -radius, top_z)

    fill_data = bpy.data.lights.new(name="FillLight", type="AREA")
    fill_data.energy = 1500
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.location = (-radius, radius * 0.5, model_info["height"] * 0.7 + 1.0)


def clear_cameras_and_targets() -> None:
    for obj in list(bpy.data.objects):
        if obj.type == "CAMERA" or obj.name.startswith("Target_"):
            bpy.data.objects.remove(obj, do_unlink=True)


def create_cameras(config: dict) -> int:
    clear_cameras_and_targets()

    num_cameras_per_row = int(config["num_cameras_per_row"])
    radius = float(config["radius"])
    vertical_offsets = config["vertical_offsets"]
    target_heights = config["target_heights"]
    lens_mm = float(config["lens_mm"])

    angles_deg = config.get("camera_angles_deg")
    if angles_deg is not None and len(angles_deg) != num_cameras_per_row:
        raise ValueError(
            f"camera_angles_deg debe tener {num_cameras_per_row} entradas, tiene {len(angles_deg)}."
        )

    cam_index = 0
    for row, height in enumerate(vertical_offsets):
        target_z = float(target_heights[row])

        for i in range(num_cameras_per_row):
            angle = math.radians(float(angles_deg[i])) if angles_deg else (i / num_cameras_per_row) * 2 * math.pi
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = float(height)

            cam_data = bpy.data.cameras.new(f"Camera_{cam_index:02d}")
            cam_data.lens = lens_mm
            cam_obj = bpy.data.objects.new(f"Camera_{cam_index:02d}", cam_data)
            bpy.context.collection.objects.link(cam_obj)
            cam_obj.location = (x, y, z)

            target = bpy.data.objects.new(f"Target_{cam_index:02d}", None)
            bpy.context.collection.objects.link(target)
            target.location = (0, 0, target_z)

            constraint = cam_obj.constraints.new(type="TRACK_TO")
            constraint.target = target
            constraint.track_axis = "TRACK_NEGATIVE_Z"
            constraint.up_axis = "UP_Y"

            cam_index += 1

    return cam_index


def export_exact_camera_metadata(config: dict) -> list[dict]:
    camera_objects = sorted((obj for obj in bpy.data.objects if obj.type == "CAMERA"), key=lambda obj: obj.name)
    records: list[dict] = []

    for index, cam_obj in enumerate(camera_objects):
        camera_data = cam_obj.data
        depsgraph = bpy.context.evaluated_depsgraph_get()
        matrix_world = cam_obj.evaluated_get(depsgraph).matrix_world.copy()
        rotation_c2w = matrix_world.to_3x3()
        location = matrix_world.to_translation()
        rotation_w2c = rotation_c2w.transposed()
        translation = -(rotation_w2c @ location)
        qw, qx, qy, qz = rotation_w2c.to_quaternion()

        records.append(
            {
                "image_name": f"{cam_obj.name}.png",
                "camera_name": cam_obj.name,
                "row": index // int(config["num_cameras_per_row"]),
                "column": index % int(config["num_cameras_per_row"]),
                "position": [float(location.x), float(location.y), float(location.z)],
                "target": [0.0, 0.0, float(config["target_heights"][index // int(config["num_cameras_per_row"])])],
                "qvec": [float(qw), float(qx), float(qy), float(qz)],
                "tvec": [float(translation.x), float(translation.y), float(translation.z)],
                "lens_mm": float(camera_data.lens),
                "sensor_width_mm": float(camera_data.sensor_width),
                "sensor_height_mm": float(camera_data.sensor_height),
            }
        )

    return records


def render_all_cameras(output_folder: Path, config: dict) -> None:
    os.makedirs(output_folder, exist_ok=True)

    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.resolution_x = int(config["resolution_x"])
    scene.render.resolution_y = int(config["resolution_y"])
    scene.render.resolution_percentage = 100

    cameras = sorted((obj for obj in bpy.data.objects if obj.type == "CAMERA"), key=lambda obj: obj.name)

    if not cameras:
        print("No se encontraron camaras en la escena.")
        return

    print(f"Se encontraron {len(cameras)} camaras. Iniciando captura...")

    for index, cam in enumerate(cameras, start=1):
        scene.camera = cam
        filename = f"{cam.name}.png"
        filepath = output_folder / filename
        scene.render.filepath = str(filepath)
        bpy.ops.render.render(write_still=True)
        print(f"Captura {index}/{len(cameras)}: {filename}")

    print(f"Proceso completado. {len(cameras)} imagenes guardadas en: {output_folder}")


def images_already_rendered(output_folder: Path, expected_count: int) -> bool:
    if not output_folder.is_dir():
        return False
    existing = [p for p in output_folder.iterdir() if p.suffix.lower() == ".png"]
    return len(existing) >= expected_count


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el JSON de configuracion: {config_path}")

    config = load_config(config_path)
    model_path = resolve_model_path(config["model_name"])
    experiment_dir, output_folder = build_output_paths(config)

    expected_count = int(config["num_cameras_per_row"]) * len(config["vertical_offsets"])
    if images_already_rendered(output_folder, expected_count):
        print(f"Saltado: ya existen {expected_count} imagenes en {output_folder}")
        sys.exit(0)

    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo en herramientas/modelos: {model_path}")
    if model_path.suffix.lower() != ".fbx":
        raise ValueError("Por ahora este script solo soporta modelos .fbx")

    clear_scene()
    imported_objects = import_fbx(model_path)
    model_info = center_model(imported_objects)
    setup_lighting(model_info, float(config["radius"]))
    total_cameras = create_cameras(config)

    print(f"Modelo importado: {model_path.name}")
    print(f"Experimento: {config['experiment_name']}")
    print(f"Camaras creadas: {total_cameras}")
    print(f"Dimensiones aproximadas del modelo: {model_info}")
    print(f"Carpeta de salida: {experiment_dir}")

    metadata = derive_experiment_metadata(config, experiment_dir, source_config=str(config_path))
    metadata["pose_source"] = "blender_export"
    metadata["cameras"] = export_exact_camera_metadata(config)
    metadata["capture"]["model_bounds"] = model_info
    metadata_path = save_metadata(experiment_dir, metadata)
    print(f"Metadatos del experimento guardados en: {metadata_path}")

    render_all_cameras(output_folder, config)

    if args.save_blend:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        blend_path = experiment_dir / f"{model_path.stem}_{config['experiment_name']}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"Escena guardada en: {blend_path}")


if __name__ == "__main__":
    main()
