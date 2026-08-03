from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLENDER_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "blender"
if str(BLENDER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(BLENDER_SCRIPTS_DIR))

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "scripts" / "blender" / "experimentos"
DEFAULT_PRUEBAS_DIR = PROJECT_ROOT / "pruebas"
REPORT_PATH = PROJECT_ROOT / "evaluacion_experimentos.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Muestra el estado y las metricas de todos los experimentos del pipeline."
    )
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--pruebas-dir", type=Path, default=DEFAULT_PRUEBAS_DIR)
    parser.add_argument("--save", action="store_true", help="Guarda el informe en evaluacion_experimentos.txt")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def splat_info(output_dir: Path) -> tuple[str, str]:
    if not output_dir.is_dir():
        return "NO", ""
    splats = sorted(output_dir.glob("splat_*.ply"))
    if not splats:
        return "NO", ""
    splat = splats[-1]
    size_mb = splat.stat().st_size / 1_048_576
    return "SI", f"{splat.name} ({size_mb:.0f} MB)"


def colmap_estado(dense_dir: Path) -> str:
    summary_path = dense_dir / "colmap_run_summary.json"
    if not dense_dir.is_dir():
        return "pendiente"
    if not summary_path.is_file():
        return "carpeta vacia" if not any(dense_dir.iterdir()) else "sin resumen"
    return "ok"


def format_row(cols: list[str], widths: list[int]) -> str:
    return "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))


def build_report(config_dir: Path, pruebas_dir: Path) -> str:
    config_paths = sorted(config_dir.glob("*.json"))
    if not config_paths:
        return "No se encontraron JSONs de experimentos."

    rows: list[dict] = []
    for config_path in config_paths:
        cfg = load_json(config_path)
        if cfg is None:
            continue

        name = cfg.get("experiment_name", config_path.stem)
        n_per_row = int(cfg.get("num_cameras_per_row", 0))
        n_rows = len(cfg.get("vertical_offsets", []))
        total = n_per_row * n_rows
        radius = cfg.get("radius", "?")
        lens = cfg.get("lens_mm", "?")

        dataset_dir = pruebas_dir / name
        dense_dir = dataset_dir / "dense"
        output_dir = dense_dir / "output"

        estado_colmap = colmap_estado(dense_dir)
        splat_ok, splat_desc = splat_info(output_dir)

        colmap_reg = ""
        colmap_pts = ""
        colmap_obs = ""
        colmap_err = ""
        colmap_mode = ""

        if estado_colmap == "ok":
            summary = load_json(dense_dir / "colmap_run_summary.json") or {}
            total_imgs = summary.get("input_image_count", total)
            reg_imgs = summary.get("registered_image_count", 0)
            colmap_reg = f"{reg_imgs}/{total_imgs}"
            metrics = summary.get("sparse_metrics") or {}
            colmap_pts = str(int(metrics.get("points", 0)))
            colmap_obs = f"{float(metrics.get('mean_observations_per_image', 0)):.0f}"
            colmap_err = f"{float(metrics.get('mean_reprojection_error', 0)):.3f}px"
            colmap_mode = summary.get("reconstruction_mode", "")
            if colmap_mode == "known_poses_triangulation":
                colmap_mode = "known_poses"

        rows.append({
            "name": name,
            "total": total,
            "radius": radius,
            "lens": lens,
            "filas": n_rows,
            "colmap": estado_colmap,
            "reg": colmap_reg,
            "puntos": colmap_pts,
            "obs": colmap_obs,
            "err": colmap_err,
            "mode": colmap_mode,
            "splat": splat_ok,
            "splat_desc": splat_desc,
        })

    lines: list[str] = []
    sep = "=" * 110

    lines.append(sep)
    lines.append(f"  EVALUACION DE EXPERIMENTOS  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(sep)

    # ── Tabla resumen ──────────────────────────────────────────────────────────
    headers = ["EXPERIMENTO", "CAMS", "RADIO", "LENS", "FILAS", "COLMAP", "REG", "PUNTOS", "OBS/IMG", "ERR", "SPLAT"]
    widths =  [26,             5,      6,       6,      5,       12,       7,     7,        7,         8,     5]
    lines.append("")
    lines.append(format_row(headers, widths))
    lines.append("  " + "-" * 106)
    for r in rows:
        cols = [
            r["name"], r["total"], r["radius"], f"{r['lens']}mm",
            r["filas"], r["colmap"], r["reg"], r["puntos"],
            r["obs"], r["err"], r["splat"],
        ]
        lines.append(format_row([str(c) for c in cols], widths))

    # ── Detalle por experimento con COLMAP hecho ───────────────────────────────
    done = [r for r in rows if r["colmap"] == "ok"]
    if done:
        lines.append("")
        lines.append(sep)
        lines.append("  DETALLE  (solo experimentos con COLMAP completado)")
        lines.append(sep)
        for r in done:
            lines.append("")
            lines.append(f"  {r['name']}")
            lines.append(f"    Configuracion  : {r['total']} camaras  |  {r['filas']} filas  |  radio {r['radius']} m  |  {r['lens']} mm")
            lines.append(f"    Modo COLMAP    : {r['mode']}")
            lines.append(f"    Registradas    : {r['reg']}")
            lines.append(f"    Puntos sparse  : {r['puntos']}")
            lines.append(f"    Obs/imagen     : {r['obs']}")
            lines.append(f"    Error reproyec.: {r['err']}")
            lines.append(f"    Modelo 3DGS    : {r['splat_desc'] if r['splat_desc'] else 'NO entrenado'}")

    # ── Pendientes ─────────────────────────────────────────────────────────────
    pending = [r for r in rows if r["colmap"] != "ok"]
    if pending:
        lines.append("")
        lines.append(sep)
        lines.append("  PENDIENTES  (sin COLMAP completado)")
        lines.append(sep)
        for r in pending:
            lines.append(f"  {r['name']:<28}  {r['total']} cams  |  radio {r['radius']}  |  {r['lens']} mm  |  filas {r['filas']}  →  {r['colmap']}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args.config_dir.resolve(), args.pruebas_dir.resolve())
    print(report)
    if args.save:
        REPORT_PATH.write_text(report + "\n", encoding="utf-8")
        print(f"\nInforme guardado en: {REPORT_PATH}")


if __name__ == "__main__":
    main()
