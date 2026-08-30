"""
crop_splat.py - Recorte del modelo 3DGS entrenado (.ply) para quitar fondo residual.

El entrenamiento con máscaras ya evita casi todo el fondo, pero pueden quedar
gaussianas sueltas (flotantes lejos del sujeto) o casi transparentes. Este paso
las elimina SOBRE EL MODELO FINAL:

  * Recorte espacial: descarta gaussianas fuera de una esfera alrededor del
    sujeto (centro = convergencia de las cámaras, radio = fracción del anillo).
  * Opacidad mínima: descarta gaussianas casi invisibles (sigmoid(opacity) < umbral).
  * (opcional) Clúster principal sobre las gaussianas restantes.

Lee/mantiene TODAS las propiedades del .ply de 3DGS (f_dc, f_rest, scale, rot,
opacity). Escala/centro se estiman con images.bin de COLMAP (mismo método que
clean_pointcloud). Guarda un .ply nuevo y deja el original intacto.

Uso:
    python crop_splat.py <model.ply> --sparse <.../dense/sparse> [opciones]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Lector / escritor de PLY binario (little-endian) genérico
# --------------------------------------------------------------------------- #
_PLY_TYPE = {
    "float": ("f4", 4), "float32": ("f4", 4),
    "double": ("f8", 8), "float64": ("f8", 8),
    "uchar": ("u1", 1), "uint8": ("u1", 1),
    "char": ("i1", 1), "int8": ("i1", 1),
    "ushort": ("u2", 2), "uint16": ("u2", 2),
    "short": ("i2", 2), "int16": ("i2", 2),
    "uint": ("u4", 4), "uint32": ("u4", 4),
    "int": ("i4", 4), "int32": ("i4", 4),
}


def read_ply(path: Path):
    data = path.read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii", "replace").splitlines()
    fmt, count, props = None, 0, []
    for line in header:
        t = line.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element" and t[1] == "vertex":
            count = int(t[2])
        elif t[0] == "property":
            props.append((t[2], t[1]))  # (name, type)
    if fmt != "binary_little_endian":
        raise ValueError(f"Solo soporto binary_little_endian (este es {fmt})")
    dt = np.dtype([(name, "<" + _PLY_TYPE[typ][0]) for name, typ in props])
    arr = np.frombuffer(data, dtype=dt, count=count, offset=end)
    return arr, props, header


def write_ply(path: Path, arr, props, header):
    # reescribe cabecera cambiando solo el nº de vértices
    out_header = []
    for line in header:
        if line.startswith("element vertex"):
            out_header.append(f"element vertex {len(arr)}")
        else:
            out_header.append(line)
    blob = ("\n".join(out_header) + "\n").encode("ascii")
    with path.open("wb") as f:
        f.write(blob)
        f.write(arr.tobytes())


# --------------------------------------------------------------------------- #
#  Geometría de cámaras (centro y radio del sujeto)  -- reutiliza clean_pointcloud
# --------------------------------------------------------------------------- #
def subject_center_radius(sparse_dir: Path):
    from clean_pointcloud import read_camera_geometry, scene_center
    centers, dirs = read_camera_geometry(sparse_dir / "images.bin")
    c = scene_center(centers, dirs)
    ring = float(np.median(np.linalg.norm(centers - c, axis=1)))
    return c, ring


# --------------------------------------------------------------------------- #
#  Recorte
# --------------------------------------------------------------------------- #
def crop(model_ply: Path, sparse_dir: Path, *, crop_factor=0.8, min_opacity=0.05,
         cluster=False, cluster_eps_factor=5.0, sor_std=0.0, knn=12, max_scale_pct=0.0, out=None):
    arr, props, header = read_ply(Path(model_ply))
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    N = len(arr)
    keep = np.ones(N, bool)
    st = {"total": N}

    # --- opacidad mínima (opacity guardada en logit; prob = sigmoid) ---
    if "opacity" in arr.dtype.names and min_opacity > 0:
        prob = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))
        keep &= prob >= min_opacity
    st["tras_opacidad"] = int(keep.sum())

    # --- recorte espacial alrededor del sujeto (solo quita flotantes lejanos) ---
    if sparse_dir is not None:
        c, ring = subject_center_radius(Path(sparse_dir))
        dist = np.linalg.norm(xyz - c, axis=1)
        before = int(keep.sum())
        spatial = keep & (dist <= crop_factor * ring)
        # Salvaguarda: un centro de escena mal estimado podría recortar el sujeto.
        # Si el recorte espacial se llevaría más del 50%, no lo aplicamos.
        if before > 0 and int(spatial.sum()) < 0.5 * before:
            st["espacial_omitido"] = True
        else:
            keep = spatial
        st["radio_anillo"] = ring
        st["radio_recorte"] = crop_factor * ring
    st["tras_espacial"] = int(keep.sum())

    # --- clúster principal (opcional) ---
    if cluster and keep.sum() > 50:
        from scipy.spatial import cKDTree
        from scipy.sparse.csgraph import connected_components
        idx = np.nonzero(keep)[0]
        pts = xyz[idx]
        tree = cKDTree(pts)
        nn = tree.query(pts, k=2)[0][:, 1]
        eps = float(np.median(nn)) * cluster_eps_factor
        g = tree.sparse_distance_matrix(tree, max_distance=eps, output_type="coo_matrix")
        n_c, labels = connected_components(g, directed=False)
        if n_c > 1:
            biggest = np.bincount(labels).argmax()
            main = labels == biggest
            # Salvaguarda: solo nos quedamos con el componente principal si es la mayoría
            # (evita colapsar el modelo si el objeto quedó fragmentado por un eps pequeño).
            if int(main.sum()) >= 0.5 * len(idx):
                keep[idx[~main]] = False
            else:
                st["cluster_omitido"] = True
    st["tras_cluster"] = int(keep.sum())

    # --- quita gaussianas gigantes (los bokeh de fondo suelen ser las más grandes) ---
    if max_scale_pct and max_scale_pct > 0 and "scale_0" in arr.dtype.names:
        idx = np.nonzero(keep)[0]
        sc = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], 1).astype(np.float64)).mean(1)
        if len(idx) > 10:
            thr = np.percentile(sc[idx], 100.0 - max_scale_pct)
            keep[idx[sc[idx] > thr]] = False
    st["tras_escala"] = int(keep.sum())

    # --- outliers estadísticos (SOR): quita flotantes aislados (bokeh de fondo) ---
    if sor_std and sor_std > 0:
        from scipy.spatial import cKDTree
        idx = np.nonzero(keep)[0]
        if len(idx) > knn + 1:
            pts = xyz[idx]
            tree = cKDTree(pts)
            d, _ = tree.query(pts, k=knn + 1)
            md = d[:, 1:].mean(axis=1)              # dist. media a los k vecinos
            thr = md.mean() + sor_std * md.std()
            keep[idx[md > thr]] = False
    st["tras_sor"] = int(keep.sum())

    out = Path(out) if out else Path(model_ply).with_name(Path(model_ply).stem + "_cropped.ply")
    write_ply(out, arr[keep], props, header)
    st["final"] = int(keep.sum())
    st["removed"] = N - st["final"]
    st["out"] = str(out)
    return st


def parse_args():
    ap = argparse.ArgumentParser(description="Recorta fondo residual de un modelo 3DGS (.ply).")
    ap.add_argument("model_ply", type=Path, help="Modelo entrenado (splat_XXXX.ply / model.ply)")
    ap.add_argument("--sparse", type=Path, default=None, help="Carpeta .../dense/sparse (para centro y radio)")
    ap.add_argument("--crop-factor", type=float, default=0.8, help="Radio a conservar (fracción del anillo). Menor = más agresivo.")
    ap.add_argument("--min-opacity", type=float, default=0.05, help="Descarta gaussianas con opacidad < umbral.")
    ap.add_argument("--cluster", action="store_true", help="Además, quedarse con el clúster principal.")
    ap.add_argument("--cluster-eps", type=float, default=5.0, help="Factor de eps para el clúster (menor = separa islas cercanas).")
    ap.add_argument("--max-scale-pct", type=float, default=0.0, help="Quita el X%% de gaussianas mas grandes (bokeh). 0 = off.")
    ap.add_argument("--sor", type=float, default=0.0,
                    help="Outliers estadísticos: quita gaussianas aisladas (flotantes). Menor = más agresivo (p.ej. 1.0-2.0). 0 = off.")
    ap.add_argument("--knn", type=int, default=12, help="Vecinos para el SOR.")
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    a = parse_args()
    print(f">>> Recortando {a.model_ply}")
    s = crop(a.model_ply, a.sparse, crop_factor=a.crop_factor, min_opacity=a.min_opacity,
             cluster=a.cluster, cluster_eps_factor=a.cluster_eps, sor_std=a.sor, knn=a.knn, max_scale_pct=a.max_scale_pct, out=a.out)
    print(f"    Total               : {s['total']}")
    print(f"    Tras opacidad       : {s['tras_opacidad']}")
    print(f"    Tras recorte espacial: {s['tras_espacial']}"
          + (f"   [radio={s.get('radio_recorte',0):.2f} de anillo={s.get('radio_anillo',0):.2f}]"
             if 'radio_recorte' in s else ""))
    print(f"    Tras clúster        : {s['tras_cluster']}")
    print(f"    Tras SOR            : {s['tras_sor']}")
    print(f"    FINAL               : {s['final']}  ({100*s['removed']/max(1,s['total']):.1f}% eliminado)")
    print(f"    Guardado -> {s['out']}")


if __name__ == "__main__":
    main()
