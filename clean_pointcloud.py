"""
clean_pointcloud.py - Limpieza de la nube dispersa de COLMAP.

Entre COLMAP y el entrenamiento, filtra `points3D.bin` para centrarse en el
objeto principal y descartar el fondo. Reescribe SOLO points3D.bin dejando
cameras.bin e images.bin intactos, para que LichtFeld siga entrenando igual.

Dos etapas:

  ETAPA 1 - FONDO (quitar lo que no es el objeto)
    * Recorte por geometría de cámaras: como todas las fotos miran al objeto,
      su punto de convergencia es el centro; se descarta lo que cae fuera del
      "anillo" de cámaras (paredes, suelo lejano, habitación).
    * Clúster principal: se queda con el grupo de puntos conectado más grande
      (el objeto) y tira las islas de fondo sueltas.

  ETAPA 2 - PULIDO (limpiar la nube del objeto)
    * Filtro COLMAP: descarta puntos vistos por pocas cámaras o con error alto.
    * Outliers estadísticos (SOR): descarta puntos aislados según sus vecinos.

Uso:
    python clean_pointcloud.py <ruta a .../dense/sparse>  [opciones]

Requiere numpy y scipy (solo para este paso opcional).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

try:
    import numpy as np
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "La limpieza necesita numpy y scipy:  pip install numpy scipy\n"
        f"(import falló: {e})"
    )

_POINT_FMT = "<QdddBBBdQ"                 # id, xyz, rgb, error, track_len
_POINT_SIZE = struct.calcsize(_POINT_FMT)  # 51 bytes


# --------------------------------------------------------------------------- #
#  points3D.bin  (lectura / escritura)
# --------------------------------------------------------------------------- #
def read_points3D(path: Path):
    data = path.read_bytes()
    off = 0
    (num,) = struct.unpack_from("<Q", data, off); off += 8
    ids = np.empty(num, np.uint64)
    xyz = np.empty((num, 3), np.float64)
    rgb = np.empty((num, 3), np.uint8)
    err = np.empty(num, np.float64)
    tlen = np.empty(num, np.uint64)
    tracks: list[bytes] = []
    for i in range(num):
        pid, x, y, z, r, g, b, e, tl = struct.unpack_from(_POINT_FMT, data, off)
        off += _POINT_SIZE
        ids[i] = pid; xyz[i] = (x, y, z); rgb[i] = (r, g, b); err[i] = e; tlen[i] = tl
        nbytes = int(tl) * 8
        tracks.append(data[off:off + nbytes]); off += nbytes
    return ids, xyz, rgb, err, tlen, tracks


def write_points3D(path: Path, keep, ids, xyz, rgb, err, tlen, tracks):
    idx = np.nonzero(keep)[0]
    out = bytearray()
    out += struct.pack("<Q", len(idx))
    for i in idx:
        out += struct.pack(_POINT_FMT, int(ids[i]), float(xyz[i, 0]), float(xyz[i, 1]),
                           float(xyz[i, 2]), int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]),
                           float(err[i]), int(tlen[i]))
        out += tracks[i]
    path.write_bytes(bytes(out))


def write_ply(path: Path, xyz, rgb):
    n = len(xyz)
    head = ("ply\nformat ascii 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n")
    with path.open("w") as f:
        f.write(head)
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# --------------------------------------------------------------------------- #
#  images.bin  (centros y direcciones de las cámaras)
# --------------------------------------------------------------------------- #
def read_camera_geometry(images_bin: Path):
    """Devuelve (centros Nx3, direcciones Nx3) de las cámaras registradas."""
    data = images_bin.read_bytes()
    off = 0
    (num,) = struct.unpack_from("<Q", data, off); off += 8
    centers, dirs = [], []
    for _ in range(num):
        off += 4  # image_id (uint32)
        qw, qx, qy, qz, tx, ty, tz = struct.unpack_from("<7d", data, off); off += 56
        off += 4  # camera_id (uint32)
        end = data.index(b"\x00", off); off = end + 1          # nombre (null-terminated)
        (n2d,) = struct.unpack_from("<Q", data, off); off += 8
        off += n2d * 24                                        # puntos 2D (x,y,point3D_id)

        # rotación (COLMAP: qvec = [w,x,y,z], mundo->cámara)
        n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 or 1.0
        w, x, y, z = qw / n, qx / n, qy / n, qz / n
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])
        t = np.array([tx, ty, tz])
        centers.append(-R.T @ t)          # centro de la cámara en el mundo
        dirs.append(R.T @ np.array([0.0, 0.0, 1.0]))  # eje óptico (hacia la escena)
    return np.array(centers), np.array(dirs)


def scene_center(centers, dirs):
    """Punto donde convergen los ejes ópticos (centro del objeto)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in zip(centers, dirs):
        d = d / (np.linalg.norm(d) or 1.0)
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ C
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return centers.mean(axis=0)


# --------------------------------------------------------------------------- #
#  Filtros
# --------------------------------------------------------------------------- #
def largest_cluster(points, eps):
    """Máscara del componente conexo más grande (grafo de vecinos a distancia eps)."""
    if len(points) < 3:
        return np.ones(len(points), bool)
    tree = cKDTree(points)
    graph = tree.sparse_distance_matrix(tree, max_distance=eps, output_type="coo_matrix")
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return np.ones(len(points), bool)
    biggest = np.bincount(labels).argmax()
    return labels == biggest


def clean_sparse(sparse_dir: Path, *, min_track=3, max_error=2.0, std_ratio=2.0,
                 crop_factor=0.9, cluster=True, cluster_eps_factor=4.0, knn=8,
                 min_keep_frac=0.02, export_ply=True) -> dict:
    sparse_dir = Path(sparse_dir)
    p3 = sparse_dir / "points3D.bin"
    if not p3.is_file():
        raise FileNotFoundError(f"No existe {p3}")

    ids, xyz, rgb, err, tlen, tracks = read_points3D(p3)
    N = len(ids)
    keep = np.ones(N, bool)
    st = {"total": N}

    # -------- ETAPA 2a: filtro COLMAP (barato, quita basura obvia) --------
    keep &= tlen.astype(np.int64) >= min_track
    keep &= err <= max_error
    st["tras_colmap"] = int(keep.sum())

    # -------- ETAPA 1a: FONDO por geometría de cámaras (recorte al anillo) --------
    imgs = sparse_dir / "images.bin"
    center = None
    if imgs.is_file():
        try:
            centers, dirs = read_camera_geometry(imgs)
            center = scene_center(centers, dirs)
            ring = np.median(np.linalg.norm(centers - center, axis=1))
            surv = np.nonzero(keep)[0]
            dist = np.linalg.norm(xyz[surv] - center, axis=1)
            keep[surv[dist > crop_factor * ring]] = False
            st["ring_radius"] = float(ring)
        except Exception as ex:
            st["ring_error"] = str(ex)
    st["tras_fondo_anillo"] = int(keep.sum())

    # -------- ETAPA 1b: FONDO por clúster principal (el objeto conectado) --------
    if cluster:
        surv = np.nonzero(keep)[0]
        if len(surv) > 50:
            tree = cKDTree(xyz[surv])
            nn = tree.query(xyz[surv], k=2)[0][:, 1]
            eps = float(np.median(nn)) * cluster_eps_factor
            big = largest_cluster(xyz[surv], eps)
            keep[surv[~big]] = False
    st["tras_fondo_cluster"] = int(keep.sum())

    # -------- ETAPA 2b: PULIDO (outliers estadísticos, SOR) --------
    surv = np.nonzero(keep)[0]
    if len(surv) > knn + 1:
        tree = cKDTree(xyz[surv])
        d, _ = tree.query(xyz[surv], k=knn + 1)
        md = d[:, 1:].mean(axis=1)
        thr = md.mean() + std_ratio * md.std()
        keep[surv[md > thr]] = False
    st["tras_pulido"] = int(keep.sum())

    n_final = int(keep.sum())
    floor = max(200, int(N * min_keep_frac))
    if n_final < floor:
        raise RuntimeError(
            f"Limpieza demasiado agresiva: quedan {n_final}/{N} puntos (mínimo {floor}). "
            f"Aborto. Sube --crop-factor o desactiva el clúster con --no-cluster.")

    backup = sparse_dir / "points3D_original.bin"
    if not backup.is_file():
        backup.write_bytes(p3.read_bytes())
    if export_ply:
        write_ply(sparse_dir / "points3D_before.ply", xyz, rgb)
        write_ply(sparse_dir / "points3D_clean.ply", xyz[keep], rgb[keep])
    write_points3D(p3, keep, ids, xyz, rgb, err, tlen, tracks)

    st["final"] = n_final
    st["removed"] = N - n_final
    return st


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Limpia la nube dispersa de COLMAP (points3D.bin).")
    ap.add_argument("sparse_dir", type=Path, help="Carpeta .../dense/sparse")
    ap.add_argument("--min-track", type=int, default=3)
    ap.add_argument("--max-error", type=float, default=2.0)
    ap.add_argument("--crop-factor", type=float, default=0.9,
                    help="Fondo: radio (fracción del anillo de cámaras) a conservar. Menor = más agresivo.")
    ap.add_argument("--no-cluster", action="store_true", help="No filtrar por clúster principal.")
    ap.add_argument("--cluster-eps-factor", type=float, default=4.0)
    ap.add_argument("--std-ratio", type=float, default=2.0)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--no-ply", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    print(f">>> Limpiando {a.sparse_dir}")
    s = clean_sparse(a.sparse_dir, min_track=a.min_track, max_error=a.max_error,
                     std_ratio=a.std_ratio, crop_factor=a.crop_factor,
                     cluster=not a.no_cluster, cluster_eps_factor=a.cluster_eps_factor,
                     knn=a.knn, export_ply=not a.no_ply)
    print(f"    Total inicial      : {s['total']}")
    print(f"    Tras filtro COLMAP : {s['tras_colmap']}")
    print(f"    Tras fondo (anillo): {s['tras_fondo_anillo']}"
          + (f"   [radio anillo={s['ring_radius']:.2f}]" if 'ring_radius' in s else ""))
    print(f"    Tras fondo (clúster): {s['tras_fondo_cluster']}")
    print(f"    Tras pulido (SOR)  : {s['tras_pulido']}")
    print(f"    FINAL              : {s['final']}  ({100*s['removed']/s['total']:.1f}% eliminado)")
    print("    Backup + .ply de inspección generados.")


if __name__ == "__main__":
    main()
