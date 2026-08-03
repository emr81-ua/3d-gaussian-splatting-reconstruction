import os
import time
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------- CONFIG --------
input_folder = r"C:\Users\emoky\Desktop\Universidad\TFG\images"
output_folder = r"C:\Users\emoky\Desktop\Universidad\TFG\masks"
os.makedirs(output_folder, exist_ok=True)

sam_checkpoint = r"C:\Users\emoky\Desktop\Universidad\TFG\scripts\sam_vit_b_01ec64.pth"
MAX_SIZE = 1024

# -------- INICIALIZAR SAM --------
print("Cargando modelo SAM...")
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=8,
    pred_iou_thresh=0.85,
    stability_score_thresh=0.90,
    min_mask_region_area=1000
)

# -------- FUNCIONES --------
def resize_for_sam(image):
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    return image, scale

def select_best_mask(masks, h, w):
    """
    Selecciona la máscara que MÁS píxeles tiene en el CENTRO de la imagen.
    Asume que el objeto de interés siempre está centrado.
    """
    if not masks:
        return None
    
    cx, cy = w // 2, h // 2
    
    # Definir región central (30% del centro de la imagen)
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
        
        # Filtro básico: descartar máscaras muy pequeñas o enormes
        if area < 0.02 * h * w or area > 0.95 * h * w:
            continue
        
        # Contar cuántos píxeles de la máscara están en el CENTRO
        center_region = mask[y_start:y_end, x_start:x_end]
        center_pixels = center_region.sum()
        
        # La máscara con MÁS píxeles en el centro es la ganadora
        if center_pixels > best_center_pixels:
            best_center_pixels = center_pixels
            best_mask = mask
    
    # Si no encontramos ninguna, usar la más grande que toque el centro
    if best_mask is None:
        for m in masks:
            mask = m["segmentation"].astype(np.uint8)
            center_region = mask[y_start:y_end, x_start:x_end]
            center_pixels = center_region.sum()
            
            if center_pixels > 0:  # Al menos toca el centro
                if best_mask is None or mask.sum() > best_mask.sum():
                    best_mask = mask
    
    # Último recurso: la más grande de todas
    if best_mask is None:
        largest = max(masks, key=lambda m: m["segmentation"].sum())
        best_mask = largest["segmentation"].astype(np.uint8)
    
    return best_mask

def generate_mask(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  ⚠️ ERROR: No se pudo leer {image_path}")
        return None
    
    h, w = image.shape[:2]

    # Procesar con SAM
    image_small, scale = resize_for_sam(image)
    hs, ws = image_small.shape[:2]
    image_rgb = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image_rgb)

    if not masks:
        return np.zeros((h, w), dtype=np.uint8)

    best_mask = select_best_mask(masks, hs, ws)
    if best_mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    # Convertir a binario 0/255
    mask = (best_mask > 0).astype(np.uint8) * 255

    # Redimensionar a tamaño original
    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Dilatación ligera
    # Dilatación ligera
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 🔁 INVERTIR: objeto blanco, fondo negro (COLMAP-friendly)
    mask = 255 - mask

    return mask

def verify_mask_dimensions(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        return False, "Archivo no encontrado"
    
    if img.shape[:2] != mask.shape[:2]:
        return False, f"img={img.shape[:2]} mask={mask.shape[:2]}"
    
    return True, "OK"

# -------- RUN --------
print("="*60)
print(" GENERACIÓN DE MÁSCARAS PARA COLMAP")
print(" Estrategia: Objeto siempre centrado")
print(" Formato: IMG.jpg → IMG.jpg.png")
print("="*60)

total_start = time.perf_counter()
tiempos = {}
errores = []

image_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"\nEncontradas {len(image_files)} imágenes\n")

for i, file in enumerate(image_files, 1):
    in_path = os.path.join(input_folder, file)
    mask_filename = file + ".png"
    out_path = os.path.join(output_folder, mask_filename)

    img_start = time.perf_counter()
    mask = generate_mask(in_path)
    
    if mask is None:
        errores.append(file)
        continue
    
    cv2.imwrite(out_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    img_elapsed = time.perf_counter() - img_start

    ok, msg = verify_mask_dimensions(in_path, out_path)
    status = "OK" if ok else "NOT OK"
    
    tiempos[file] = img_elapsed
    print(f"  {status} [{i}/{len(image_files)}] {file:<40} → {mask_filename:<45} {img_elapsed:.2f}s")
    
    if not ok:
        print(f"ADVERTENCIA: {msg}")
        errores.append(file)

total_elapsed = time.perf_counter() - total_start

# -------- RESUMEN --------
print("\n" + "="*60)
print(" RESUMEN")
print("="*60)
print(f"  Imágenes procesadas : {len(image_files)}")
print(f"  Máscaras generadas  : {len(tiempos)}")
print(f"  Errores             : {len(errores)}")
print(f"  Tiempo total        : {total_elapsed:.2f}s")

if tiempos:
    promedio = total_elapsed / len(tiempos)
    más_lenta  = max(tiempos, key=tiempos.get)
    más_rápida = min(tiempos, key=tiempos.get)
    print(f"  Promedio/imagen     : {promedio:.2f}s")
    print(f"  Más rápida          : {más_rápida} ({tiempos[más_rápida]:.2f}s)")
    print(f"  Más lenta           : {más_lenta} ({tiempos[más_lenta]:.2f}s)")

if errores:
    print(f"\nArchivos con errores:")
    for e in errores:
        print(f"  - {e}")

