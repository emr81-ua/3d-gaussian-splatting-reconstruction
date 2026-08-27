"""
clean_pointcloud.py - Limpieza de la nube dispersa de COLMAP.

Entre COLMAP y el entrenamiento, filtra `points3D.bin` para centrarse en el
objeto y descartar fondo / floaters. Reescribe SOLO points3D.bin dejando
cameras.bin e images.bin intactos, para que LichtFeld siga entrenando igual.

Tres filtros (combinables):
  1. COLMAP        -> descarta puntos vistos por pocas cámaras o con error alto.
  2. Outliers (SOR)-> descarta puntos aislados según sus vecinos (scipy KD-tree).
  3. Recorte       -> se queda con un radio alrededor del centro del objeto.

Uso:
    python clean_pointcloud.py <ruta a .../dense/sparse>  [opciones]

Deja:
    points3D.bin              (la nube limpia; sustituye la original)
    points3D_original.bin     (copia de seguridad de la original)
    points3D_before.ply       (nube original, para inspección visual)
    points3D_clean.ply        (nube limpia, para inspección visual)

Requiere numpy y scipy (solo para este paso opcional).
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

try:
    import numpy as np
    from scipy.spatial import cKDTree
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "La limpieza necesita numpy y scipy:  pip install numpy scipy\n"
        f"(import falló: {e})"
    )

# Un registro de points3D.bin: id(Q) + xyz(ddd) + rgb(BBB) + error(d) + track_len(Q)
_POINT_FMT = "<QdddBBBdQ"
_POINT_SIZE = struct.calcsize(_POINT_FMT)  # 51 bytes


# --------------------------------------------------------------------------- #
#  Lectura / escritura del formato binario de COLMAP
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
        nbytes = int(tl) * 8  # cada elemento del track: image_id(I) + point2D_idx(I)
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
#  Filtros
# --------------------------------------------------------------------------- #
def compute_mask(xyz, err, tlen, *, min_track, max_error, std_ratio, crop_quantile, knn):
    N = len(xyz)
    keep = np.ones(N, bool)
    stats = {"total": N}

    # 1) filtro COLMAP: track suficiente + error de reproyección bajo
    keep &= tlen.astype(np.int64) >= min_track
    keep &= err <= max_error
    stats["tras_colmap"] = int(keep.sum())

    # 2) outliers estadísticos (SOR) sobre los supervivientes
    surv = np.nonzero(keep)[0]
    if len(surv) > knn + 1:
        tree = cKDTree(xyz[surv])
        d, _ = tree.query(xyz[surv], k=knn + 1)   # incluye el propio punto
        mean_d = d[:, 1:].mean(axis=1)
        thr = mean_d.mean() + std_ratio * mean_d.std()
        keep[surv[mean_d > thr]] = False
    stats["tras_sor"] = int(keep.sum())

    # 3) recorte robusto alrededor del centro del objeto
    surv = np.nonzero(keep)[0]
    if len(surv) > 10:
        center = np.median(xyz[surv], axis=0)
        dist = np.linalg.norm(xyz[surv] - center, axis=1)
        rmax = np.quantile(dist, crop_quantile)
        keep[surv[dist > rmax]] = False
    stats["tras_crop"] = int(keep.sum())

    return keep, stats


def clean_sparse(sparse_dir: Path, *, min_track=3, max_error=2.0, std_ratio=2.0,
                 crop_quantile=0.98, knn=8, min_keep_frac=0.05, export_ply=True) -> dict:
    sparse_dir = Path(sparse_dir)
    p3 = sparse_dir / "points3D.bin"
    if not p3.is_file():
        raise FileNotFoundError(f"No existe {p3}")

    ids, xyz, rgb, err, tlen, tracks = read_points3D(p3)
    keep, stats = compute_mask(xyz, err, tlen, min_track=min_track, max_error=max_error,
                               std_ratio=std_ratio, crop_quantile=crop_quantile, knn=knn)

    n_final = int(keep.sum())
    floor = max(200, int(stats["total"] * min_keep_frac))
    if n_final < floor:
        raise RuntimeError(
            f"Limpieza demasiado agresiva: quedan {n_final}/{stats['total']} puntos "
            f"(mínimo {floor}). Aborto para no dañar el modelo. Sube --crop-quantile "
            f"o baja --std-ratio / --min-track.")

    # backup del original (solo la primera vez)
    backup = sparse_dir / "points3D_original.bin"
    if not backup.is_file():
        backup.write_bytes(p3.read_bytes())

    if export_ply:
        write_ply(sparse_dir / "points3D_before.ply", xyz, rgb)
        write_ply(sparse_dir / "points3D_clean.ply", xyz[keep], rgb[keep])

    write_points3D(p3, keep, ids, xyz, rgb, err, tlen, tracks)
    stats["final"] = n_final
    stats["removed"] = stats["total"] - n_final
    return stats


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="Limpia la nube dispersa de COLMAP (points3D.bin).")
    ap.add_argument("sparse_dir", type=Path, help="Carpeta .../dense/sparse")
    ap.add_argument("--min-track", type=int, default=3, help="Mín. cámaras que ven el punto.")
    ap.add_argument("--max-error", type=float, default=2.0, help="Máx. error de reproyección (px).")
    ap.add_argument("--std-ratio", type=float, default=2.0, help="Agresividad del SOR (menor = más agresivo).")
    ap.add_argument("--crop-quantile", type=float, default=0.98, help="Recorte: se queda con este cuantil de radio.")
    ap.add_argument("--knn", type=int, default=8, help="Vecinos para el SOR.")
    ap.add_argument("--no-ply", action="store_true", help="No exportar los .ply de inspección.")
    return ap.parse_args()


def main():
    a = parse_args()
    print(f">>> Limpiando {a.sparse_dir}")
    s = clean_sparse(a.sparse_dir, min_track=a.min_track, max_error=a.max_error,
                     std_ratio=a.std_ratio, crop_quantile=a.crop_quantile, knn=a.knn,
                     export_ply=not a.no_ply)
    print(f"    Total inicial : {s['total']}")
    print(f"    Tras COLMAP   : {s['tras_colmap']}  (-{s['total']-s['tras_colmap']})")
    print(f"    Tras SOR      : {s['tras_sor']}  (-{s['tras_colmap']-s['tras_sor']})")
    print(f"    Tras recorte  : {s['tras_crop']}  (-{s['tras_sor']-s['tras_crop']})")
    print(f"    FINAL         : {s['final']}  ({100*s['removed']/s['total']:.1f}% eliminado)")
    print("    Backup en points3D_original.bin  |  .ply de inspección generados.")


if __name__ == "__main__":
    main()
