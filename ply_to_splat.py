"""
ply_to_splat.py - Convierte un modelo 3DGS (.ply) al formato compacto .splat
usado por el visor web (32 bytes por gaussiana).

Formato .splat (por gaussiana, 32 bytes):
    position : float32[3]   (x, y, z)
    scale    : float32[3]   (exp de los scale del ply, ya lineal)
    color    : uint8[4]     (r, g, b, a)   a = opacidad (sigmoid)
    rotation : uint8[4]     (cuaternión normalizado, byte = q*128+128)

Uso:
    python ply_to_splat.py <model.ply> [--out model.splat] [--max N]

--max submuestrea a N gaussianas (para un visor web más ligero/fluido).
Se conservan las de mayor opacidad*tamaño (las que más aportan al render).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def ply_to_splat(model_ply: Path, out: Path, max_splats: int | None = None) -> dict:
    from crop_splat import read_ply  # reutiliza el lector de PLY binario
    arr, _, _ = read_ply(Path(model_ply))
    n = len(arr)

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float32)
    scale = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], 1)).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float32)))
    color = 0.5 + SH_C0 * np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], 1).astype(np.float32)
    color = np.clip(color, 0.0, 1.0)
    rot = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], 1).astype(np.float32)
    rot /= (np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9)

    # submuestreo por importancia (opacidad * volumen)
    if max_splats and n > max_splats:
        importance = opacity * scale.prod(axis=1)
        keep = np.argsort(-importance)[:max_splats]
        keep.sort()
        xyz, scale, opacity, color, rot = xyz[keep], scale[keep], opacity[keep], color[keep], rot[keep]
        n = max_splats

    buf = bytearray(n * 32)
    view = memoryview(buf)
    np.frombuffer(view, dtype=np.float32).reshape(n, 8)[:, 0:3] = xyz
    np.frombuffer(view, dtype=np.float32).reshape(n, 8)[:, 3:6] = scale
    rgba = np.empty((n, 4), np.uint8)
    rgba[:, 0:3] = np.clip(color * 255, 0, 255).astype(np.uint8)
    rgba[:, 3] = np.clip(opacity * 255, 0, 255).astype(np.uint8)
    rot_b = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)
    u8 = np.frombuffer(view, dtype=np.uint8).reshape(n, 32)
    u8[:, 24:28] = rgba
    u8[:, 28:32] = rot_b

    Path(out).write_bytes(bytes(buf))
    return {"in": n if not max_splats else min(n, max_splats), "total_in": len(arr),
            "out": str(out), "bytes": len(buf)}


def parse_args():
    ap = argparse.ArgumentParser(description="Convierte .ply de 3DGS a .splat compacto para el visor web.")
    ap.add_argument("model_ply", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max", type=int, default=None, help="Submuestrea a N gaussianas (visor web más ligero).")
    return ap.parse_args()


def main():
    a = parse_args()
    out = a.out or a.model_ply.with_suffix(".splat")
    print(f">>> {a.model_ply} -> {out}")
    s = ply_to_splat(a.model_ply, out, max_splats=a.max)
    print(f"    gaussianas: {s['total_in']} -> {s['in']}   ({s['bytes']/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
