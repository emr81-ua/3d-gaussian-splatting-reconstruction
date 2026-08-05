"""
gui.py - Ventana (GUI) para el pipeline foto -> modelo 3D.

Envuelve reconstruct.py: eliges el .zip de fotos, ajustas los parametros clave
de COLMAP y del Gaussian Splatting, y le das a "Reconstruir". El progreso se ve
en vivo en el panel inferior.

Solo usa tkinter (libreria estandar de Python) -- no hay nada que instalar.

    python gui.py
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

HERE = Path(__file__).resolve().parent
RECONSTRUCT = HERE / "reconstruct.py"
CONFIG_PATH = HERE / ".gui_config.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# --------------------------------------------------------------------------- #
#  Detección de herramientas (suave: devuelve "" si no la encuentra)
# --------------------------------------------------------------------------- #
def _detect(env_var: str, names: list[str], globs: list[str]) -> str:
    import shutil
    if os.environ.get(env_var) and Path(os.environ[env_var]).is_file():
        return os.environ[env_var]
    for g in globs:
        for m in sorted(HERE.glob(g)):
            if m.is_file():
                return str(m)
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    return ""


def detect_colmap() -> str:
    return _detect("COLMAP_EXE", ["colmap", "colmap.exe"],
                   ["tools/**/colmap.exe", "tools/**/colmap"])


def detect_lichtfeld() -> str:
    return _detect("LICHTFELD_EXE", ["LichtFeld-Studio", "LichtFeld-Studio.exe"],
                   ["tools/**/LichtFeld-Studio.exe", "tools/**/LichtFeld-Studio"])


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Ventana principal
# --------------------------------------------------------------------------- #
class App:
    PAD = {"padx": 6, "pady": 4}

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Reconstrucción 3D · Gaussian Splatting")
        root.minsize(720, 640)

        cfg = load_config()
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue[str] = queue.Queue()

        # ---- variables ----
        self.v_input = tk.StringVar()
        self.v_output = tk.StringVar()
        self.v_peak = tk.StringVar(value="0.004")
        self.v_maxfeat = tk.StringVar(value="16384")
        self.v_matcher = tk.StringVar(value="exhaustive")
        self.v_iter = tk.StringVar(value="15000")
        self.v_maxg = tk.StringVar(value="500000")
        self.v_colmap = tk.StringVar(value=cfg.get("colmap", "") or detect_colmap())
        self.v_licht = tk.StringVar(value=cfg.get("lichtfeld", "") or detect_lichtfeld())

        self._build()

    # ---- construcción de la interfaz ----
    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        main = ttk.Frame(r, padding=10)
        main.grid(sticky="nsew")
        main.columnconfigure(0, weight=1)
        r.rowconfigure(0, weight=1)

        row = 0
        ttk.Label(main, text="Reconstrucción 3D a partir de fotos",
                  font=("Segoe UI", 14, "bold")).grid(row=row, column=0, sticky="w", pady=(0, 8))

        # --- Entrada ---
        row += 1
        f = ttk.LabelFrame(main, text="1 · Fotos de entrada", padding=8)
        f.grid(row=row, column=0, sticky="ew", pady=4)
        f.columnconfigure(0, weight=1)
        ttk.Entry(f, textvariable=self.v_input).grid(row=0, column=0, columnspan=2, sticky="ew", **self.PAD)
        ttk.Button(f, text="Elegir .zip…", command=self._pick_zip).grid(row=0, column=2, **self.PAD)
        ttk.Button(f, text="Elegir carpeta…", command=self._pick_folder).grid(row=0, column=3, **self.PAD)
        ttk.Label(f, text="Un .zip con las fotos, o una carpeta de fotos.",
                  foreground="#666").grid(row=1, column=0, columnspan=4, sticky="w", padx=6)

        # --- COLMAP ---
        row += 1
        f = ttk.LabelFrame(main, text="2 · COLMAP (poses de cámara)", padding=8)
        f.grid(row=row, column=0, sticky="ew", pady=4)
        for c in (1, 3):
            f.columnconfigure(c, weight=1)
        self._field(f, 0, 0, "Sensibilidad SIFT (peak_threshold):", self.v_peak,
                    "Menor = detecta más puntos. Útil con poca textura. Def. 0.004")
        self._field(f, 1, 0, "Máx. puntos por foto (max features):", self.v_maxfeat,
                    "Más = más detalle, más lento/memoria. Def. 16384")
        ttk.Label(f, text="Emparejador:").grid(row=2, column=0, sticky="w", **self.PAD)
        ttk.Combobox(f, textvariable=self.v_matcher, width=16, state="readonly",
                     values=["exhaustive", "sequential"]).grid(row=2, column=1, sticky="w", **self.PAD)
        ttk.Label(f, text="exhaustive: fotos sueltas · sequential: vídeo/orden.",
                  foreground="#666").grid(row=2, column=2, columnspan=2, sticky="w", padx=6)

        # --- Gaussian Splatting ---
        row += 1
        f = ttk.LabelFrame(main, text="3 · Gaussian Splatting (entrenamiento)", padding=8)
        f.grid(row=row, column=0, sticky="ew", pady=4)
        for c in (1, 3):
            f.columnconfigure(c, weight=1)
        self._field(f, 0, 0, "Iteraciones:", self.v_iter,
                    "Más = mejor calidad, más tiempo. Def. 15000 (prueba: 2000).")
        self._field(f, 1, 0, "Máx. gaussianas:", self.v_maxg,
                    "Más = más detalle, más VRAM. Baja a 300000 si te quedas sin memoria.")

        # --- Herramientas ---
        row += 1
        f = ttk.LabelFrame(main, text="4 · Herramientas (se recuerdan)", padding=8)
        f.grid(row=row, column=0, sticky="ew", pady=4)
        f.columnconfigure(1, weight=1)
        ttk.Label(f, text="COLMAP:").grid(row=0, column=0, sticky="w", **self.PAD)
        ttk.Entry(f, textvariable=self.v_colmap).grid(row=0, column=1, sticky="ew", **self.PAD)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._pick_exe(self.v_colmap)).grid(row=0, column=2, **self.PAD)
        ttk.Label(f, text="LichtFeld:").grid(row=1, column=0, sticky="w", **self.PAD)
        ttk.Entry(f, textvariable=self.v_licht).grid(row=1, column=1, sticky="ew", **self.PAD)
        ttk.Button(f, text="…", width=3,
                   command=lambda: self._pick_exe(self.v_licht)).grid(row=1, column=2, **self.PAD)

        # --- Botón + salida ---
        row += 1
        bar = ttk.Frame(main)
        bar.grid(row=row, column=0, sticky="ew", pady=(8, 4))
        bar.columnconfigure(1, weight=1)
        self.btn = ttk.Button(bar, text="▶  Reconstruir", command=self._start)
        self.btn.grid(row=0, column=0, sticky="w")
        self.status = ttk.Label(bar, text="Listo.", foreground="#0a7")
        self.status.grid(row=0, column=1, sticky="w", padx=12)

        row += 1
        main.rowconfigure(row, weight=1)
        self.log = scrolledtext.ScrolledText(main, height=14, state="disabled",
                                             bg="#0f1524", fg="#d7e0f5", insertbackground="#d7e0f5",
                                             font=("Consolas", 9))
        self.log.grid(row=row, column=0, sticky="nsew", pady=(4, 0))

    def _field(self, parent, r, c, label, var, help_text):
        ttk.Label(parent, text=label).grid(row=r, column=c, sticky="w", **self.PAD)
        ttk.Entry(parent, textvariable=var, width=14).grid(row=r, column=c + 1, sticky="w", **self.PAD)
        ttk.Label(parent, text=help_text, foreground="#666").grid(
            row=r, column=c + 2, columnspan=2, sticky="w", padx=6)

    # ---- selectores ----
    def _pick_zip(self):
        p = filedialog.askopenfilename(title="Elige el .zip de fotos",
                                       filetypes=[("Zip", "*.zip"), ("Todos", "*.*")])
        if p:
            self.v_input.set(p)

    def _pick_folder(self):
        p = filedialog.askdirectory(title="Elige la carpeta de fotos")
        if p:
            self.v_input.set(p)

    def _pick_exe(self, var):
        p = filedialog.askopenfilename(title="Selecciona el ejecutable",
                                       filetypes=[("Ejecutable", "*.exe"), ("Todos", "*.*")])
        if p:
            var.set(p)

    # ---- log ----
    def _write(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    # ---- lanzar ----
    def _start(self):
        if self.proc is not None:
            return
        entrada = self.v_input.get().strip()
        if not entrada or not Path(entrada).exists():
            messagebox.showerror("Falta la entrada", "Elige un .zip o una carpeta de fotos válida.")
            return
        colmap = self.v_colmap.get().strip()
        licht = self.v_licht.get().strip()
        if not colmap or not Path(colmap).is_file():
            messagebox.showerror("Falta COLMAP", "Indica la ruta a colmap.exe (sección 4).")
            return
        if not licht or not Path(licht).is_file():
            messagebox.showerror("Falta LichtFeld", "Indica la ruta a LichtFeld-Studio.exe (sección 4).")
            return

        save_config({"colmap": colmap, "lichtfeld": licht})

        # carpeta de salida: la elegida, o output/<nombre> evitando pisar una previa
        name = Path(entrada).stem if entrada.lower().endswith(".zip") else Path(entrada).name
        out = self.v_output.get().strip()
        if not out:
            base = HERE / "output" / name
            candidate, i = base, 2
            while (candidate / "dense").exists():
                candidate = base.parent / f"{name}_{i}"
                i += 1
            out = str(candidate)

        cmd = [sys.executable, str(RECONSTRUCT), entrada,
               "--output", out,
               "--iter", self.v_iter.get().strip() or "15000",
               "--max-gaussians", self.v_maxg.get().strip() or "500000",
               "--peak-threshold", self.v_peak.get().strip() or "0.004",
               "--max-features", self.v_maxfeat.get().strip() or "16384",
               "--matcher", self.v_matcher.get().strip() or "exhaustive",
               "--colmap-exe", colmap,
               "--lichtfeld-exe", licht]

        self.btn.configure(state="disabled")
        self.status.configure(text="Procesando…", foreground="#c80")
        self._write(f"$ {' '.join(cmd)}\n\n")
        threading.Thread(target=self._run, args=(cmd, out), daemon=True).start()
        self.root.after(100, self._pump)

    def _run(self, cmd, out):
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, creationflags=_NO_WINDOW)
            for line in self.proc.stdout:
                self.q.put(line)
            self.proc.wait()
            code = self.proc.returncode
        except Exception as e:
            self.q.put(f"\n[ERROR] {e}\n")
            code = -1
        self.proc = None
        self.q.put(("__DONE__", code, out))

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == "__DONE__":
                    _, code, out = item
                    self.btn.configure(state="normal")
                    if code == 0:
                        self.status.configure(text=f"✅ Listo · modelo en {out}", foreground="#0a7")
                        self._write(f"\n>>> COMPLETADO. Modelo: {out}\\model.ply\n")
                    else:
                        self.status.configure(text=f"❌ Terminó con errores (código {code})", foreground="#c33")
                    return
                else:
                    self._write(item)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
