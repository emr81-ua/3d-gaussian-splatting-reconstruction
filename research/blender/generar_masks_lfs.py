from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


MAX_SIZE = 1024
DEFAULT_CHECKPOINT = Path(r"C:\Users\emoky\Desktop\Universidad\TFG\scripts\sam_vit_b_01ec64.pth")


def resize_for_sam(image: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image, scale


def select_best_mask(masks: list[dict], h: int, w: int) -> np.ndarray | None:
    if not masks:
        return None

    cx, cy = w // 2, h // 2
    center_h = int(h * 0.3)
    center_w = int(w * 0.3)
    y_start = cy - center_h // 2
    y_end = cy + center_h // 2
    x_start = cx - center_w // 2
    x_end = cx + center_w // 2

    best_mask = None
    best_center_pixels = 0

    for m in masks:
        mask = m["segmentation"].astype(np.uint8)
        area = mask.sum()
        if area < 0.02 * h * w or area > 0.95 * h * w:
            continue

        center_pixels = mask[y_start:y_end, x_start:x_end].sum()
        if center_pixels > best_center_pixels:
            best_center_pixels = center_pixels
            best_mask = mask

    if best_mask is None:
        for m in masks:
            mask = m["segmentation"].astype(np.uint8)
            center_pixels = mask[y_start:y_end, x_start:x_end].sum()
            if center_pixels > 0:
                if best_mask is None or mask.sum() > best_mask.sum():
                    best_mask = mask

    if best_mask is None:
        largest = max(masks, key=lambda m: m["segmentation"].sum())
        best_mask = largest["segmentation"].astype(np.uint8)

    return best_mask


def generate_mask(image_path: Path, mask_generator: SamAutomaticMaskGenerator) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]
    image_small, scale = resize_for_sam(image)
    hs, ws = image_small.shape[:2]
    image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)
    if not masks:
        return np.zeros((h, w), dtype=np.uint8)

    best_mask = select_best_mask(masks, hs, ws)
    if best_mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    mask = (best_mask > 0).astype(np.uint8) * 255
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # LFS masked losses keep pixels where mask == 1, so subject should be white.
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera mascaras para LichtField Studio con sujeto en blanco y fondo en negro.")
    parser.add_argument("input_folder", type=Path, help="Carpeta de imagenes.")
    parser.add_argument("output_folder", type=Path, help="Carpeta de mascaras.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Ruta al checkpoint de SAM.")
    args = parser.parse_args()

    args.output_folder.mkdir(parents=True, exist_ok=True)

    print("Cargando modelo SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=8,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.90,
        min_mask_region_area=1000,
    )

    image_files = sorted(
        path for path in args.input_folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print("=" * 60)
    print(" GENERACION DE MASCARAS PARA LFS")
    print(" Sujeto blanco, fondo negro")
    print("=" * 60)
    print(f"\nEncontradas {len(image_files)} imagenes\n")

    total_start = time.perf_counter()
    errores: list[str] = []

    for idx, image_path in enumerate(image_files, start=1):
        out_path = args.output_folder / f"{image_path.name}.png"
        start = time.perf_counter()
        mask = generate_mask(image_path, mask_generator)
        if mask is None:
            errores.append(image_path.name)
            print(f"  ERROR [{idx}/{len(image_files)}] {image_path.name}")
            continue

        cv2.imwrite(str(out_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elapsed = time.perf_counter() - start
        print(f"  OK [{idx}/{len(image_files)}] {image_path.name:<40} -> {out_path.name:<45} {elapsed:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print(" RESUMEN")
    print("=" * 60)
    print(f"  Imagenes procesadas : {len(image_files)}")
    print(f"  Errores             : {len(errores)}")
    print(f"  Tiempo total        : {total_elapsed:.2f}s")
    if errores:
        print("  Archivos con error:")
        for error in errores:
            print(f"    - {error}")


if __name__ == "__main__":
    main()
