"""
make_masks.py - Genera máscaras del sujeto para el entrenamiento enmascarado.

Por cada imagen de <images> crea una máscara en <masks> con el MISMO nombre
(blanco = sujeto, negro = fondo). LichtFeld las busca automáticamente en una
carpeta `masks/` junto a `images/` y, con `--mask-mode ignore`, deja el fondo
fuera de la función de pérdida: el modelo final no dibuja el fondo.

Uso:
    python make_masks.py <carpeta_images> [--out <carpeta_masks>] [--model u2net_human_seg]

Requiere: rembg  (pip install "rembg[cpu]")
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_masks(images_dir: Path, masks_dir: Path, model: str = "u2net_human_seg",
               post_process: bool = True) -> dict:
    from rembg import new_session, remove
    from PIL import Image

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in _EXTS)
    if not imgs:
        raise FileNotFoundError(f"No hay imágenes en {images_dir}")

    session = new_session(model)
    n = len(imgs)
    ok, empty = 0, []
    for i, p in enumerate(imgs, 1):
        data = p.read_bytes()
        # only_mask=True -> devuelve la máscara en escala de grises (sujeto=blanco)
        mask_bytes = remove(data, session=session, only_mask=True,
                            post_process_mask=post_process)
        m = Image.open(io.BytesIO(mask_bytes)).convert("L")
        # cuánto sujeto ha encontrado (para avisar si sale casi vacía)
        hist = m.histogram()
        white_frac = sum(hist[128:]) / float(m.width * m.height)
        if white_frac < 0.01:
            empty.append(p.name)
        out = masks_dir / (p.stem + ".png")
        m.save(out)
        ok += 1
        print(f"  [{i:>3}/{n}] {p.name:<28} sujeto={white_frac*100:5.1f}%", flush=True)

    return {"total": n, "ok": ok, "empty": empty, "masks_dir": str(masks_dir)}


def parse_args():
    ap = argparse.ArgumentParser(description="Genera máscaras del sujeto (rembg) para LichtFeld.")
    ap.add_argument("images_dir", type=Path, help="Carpeta con las imágenes de entrada")
    ap.add_argument("--out", type=Path, default=None,
                    help="Carpeta de salida de máscaras (por defecto: <images_dir>/../masks)")
    ap.add_argument("--model", default="u2net_human_seg",
                    help="Modelo rembg (u2net_human_seg para personas; u2net genérico)")
    ap.add_argument("--no-post", action="store_true", help="Desactiva el suavizado de bordes de rembg")
    return ap.parse_args()


def main():
    a = parse_args()
    out = a.out or (a.images_dir.parent / "masks")
    print(f">>> Segmentando {a.images_dir}  (modelo: {a.model})")
    print(f"    Máscaras -> {out}")
    s = make_masks(a.images_dir, out, model=a.model, post_process=not a.no_post)
    print(f"\n    Hechas {s['ok']}/{s['total']} máscaras.")
    if s["empty"]:
        print(f"    ⚠ {len(s['empty'])} casi vacías (no encontró sujeto): {', '.join(s['empty'][:8])}"
              + (" …" if len(s['empty']) > 8 else ""))
    print("    Revisa la carpeta de máscaras antes de entrenar.")


if __name__ == "__main__":
    main()
