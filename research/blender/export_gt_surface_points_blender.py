from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import bpy
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from blender_bloque1 import center_model, clear_scene, import_fbx, load_config, resolve_model_path


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Muestrea puntos de la superficie GT del modelo en Blender.")
    parser.add_argument("--config-path", required=True, help="Ruta al JSON del experimento.")
    parser.add_argument("--output-path", required=True, help="Ruta del .npz de salida.")
    parser.add_argument("--sample-count", type=int, default=20000, help="Numero de puntos a muestrear.")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria.")
    return parser.parse_args(argv)


def triangulated_world_triangles(imported_objects: list[bpy.types.Object]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for obj in imported_objects:
        if obj.type != "MESH":
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        mesh.calc_loop_triangles()
        matrix = eval_obj.matrix_world.copy()

        try:
            vertices = [matrix @ vertex.co for vertex in mesh.vertices]
            for tri in mesh.loop_triangles:
                a = vertices[tri.vertices[0]]
                b = vertices[tri.vertices[1]]
                c = vertices[tri.vertices[2]]
                triangles.append(
                    (
                        np.array((a.x, a.y, a.z), dtype=np.float64),
                        np.array((b.x, b.y, b.z), dtype=np.float64),
                        np.array((c.x, c.y, c.z), dtype=np.float64),
                    )
                )
        finally:
            eval_obj.to_mesh_clear()

    if not triangles:
        raise RuntimeError("No se encontraron triangulos de malla para muestrear.")
    return triangles


def triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


def sample_surface_points(triangles: list[tuple[np.ndarray, np.ndarray, np.ndarray]], sample_count: int, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)

    areas = np.array([triangle_area(a, b, c) for a, b, c in triangles], dtype=np.float64)
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise RuntimeError("El area total de la malla es cero.")
    probabilities = areas / total_area
    indices = np.random.choice(len(triangles), size=sample_count, p=probabilities)

    samples = np.empty((sample_count, 3), dtype=np.float64)
    for i, tri_index in enumerate(indices):
        a, b, c = triangles[int(tri_index)]
        r1 = np.sqrt(np.random.rand())
        r2 = np.random.rand()
        u = 1.0 - r1
        v = r1 * (1.0 - r2)
        w = r1 * r2
        samples[i] = u * a + v * b + w * c
    return samples


def main() -> None:
    args = parse_args()
    config_path = Path(args.config_path).resolve()
    output_path = Path(args.output_path).resolve()

    config = load_config(config_path)
    model_path = resolve_model_path(config["model_name"])
    if not model_path.is_file():
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    clear_scene()
    imported_objects = import_fbx(model_path)
    model_info = center_model(imported_objects)
    triangles = triangulated_world_triangles(imported_objects)
    points = sample_surface_points(triangles, sample_count=args.sample_count, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points,
        sample_count=np.array([args.sample_count], dtype=np.int32),
        bbox_min=points.min(axis=0),
        bbox_max=points.max(axis=0),
        model_width=np.array([model_info["width"]], dtype=np.float64),
        model_depth=np.array([model_info["depth"]], dtype=np.float64),
        model_height=np.array([model_info["height"]], dtype=np.float64),
    )
    print(f"Puntos GT exportados a: {output_path}")


if __name__ == "__main__":
    main()
