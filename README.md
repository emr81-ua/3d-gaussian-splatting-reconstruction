# Photo → 3D · Gaussian Splatting pipeline

**Turn a handful of photos into a clean 3D model you can explore in real time.** One command takes a `.zip` of images and runs the whole pipeline — camera-pose estimation (COLMAP), **automatic background removal**, and 3D Gaussian Splatting training (LichtFeld Studio) — and hands you a `.ply` of just your subject, ready for any Gaussian-Splatting viewer.

> Built as my Bachelor's thesis (Computer Engineering, University of Alicante). Along the way I ran a systematic study on **what actually makes a reconstruction good** — and the answer was not the one I expected. See [The finding](#-the-finding).

<p align="center">
  <img src="assets/results.gif" alt="Reconstructed 3D model rotating in real time" width="70%">
  <br><em>⚠️ placeholder — replace <code>assets/results.gif</code> with the real capture</em>
</p>

<p align="center">
  <img src="assets/comparison.png" alt="Original render vs 3DGS reconstruction" width="70%">
  <br><em>Left: original. Right: 3DGS reconstruction.</em>
</p>

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900)

---

## ✨ What it does

```bash
python reconstruct.py my_photos.zip
```

```
my_photos.zip
   │
   ▼  COLMAP            camera poses + sparse point cloud
   ▼  rembg             segment the subject → masks/ (background removal)
   ▼  LichtFeld Studio  3D Gaussian Splatting, trained ignoring the background
   ▼  crop              remove residual floaters from the trained model
   ▼
model.ply               your subject in 3D — no background
```

- **One entry point.** A `.zip` (or folder) of photos in, a clean `model.ply` out.
- **Background removed by default.** The subject is segmented and the background is left out of training, so the model is *just the subject* — and every Gaussian is spent on it, which raises detail.
- **Fully automated**, no manual steps between stages.
- Also runnable by **double-clicking** `reconstruct.bat` (Windows) or dropping a zip onto it.

![Pipeline](docs/pipeline.png)

---

## 🎯 Background removal (the key step)

Cleaning the sparse point cloud is *not enough*: 3D Gaussian Splatting learns from the **images**, so if the background is in the photos, the optimizer rebuilds it — no matter how you pre-clean the point cloud. This pipeline removes the background where it is actually decided, in **two stages**:

1. **Subject masks (during training).** [`rembg`](https://github.com/danielgatis/rembg) segments the subject in every photo into a `masks/` folder (white = subject). LichtFeld trains with `--mask-mode ignore`, so the background is **left out of the loss** and no background Gaussians ever form.
2. **Crop (after training).** [`crop_splat.py`](crop_splat.py) removes leftover **far floaters** from the trained `.ply` — Gaussians well outside the subject region, located from the camera geometry. It is a *spatial* crop only: it never thins the subject, so the surface stays smooth.

The result is a model of the subject alone. Both stages are on by default; disable them with `--no-mask` / `--no-crop`.

> Masking does the heavy lifting: on a clean capture it already gives a spotless model, and the crop just sweeps up any distant floaters. Opacity-based pruning is available (`--crop-opacity`) but **off by default** — removing low-opacity Gaussians makes the surface look chunky.

---

## 🔍 The finding

The thesis studied how capture decisions affect reconstruction quality, using synthetic images as a **controlled lab** (exact camera poses, every variable under control). Two results stand out:

1. **More cameras is not always better.** 80 cameras registered a *lower* rate (80.0 %) than 60 cameras (86.7 %) — the geometric distribution of the views matters as much as their number.
2. **Framing beats camera count.** Making sure the whole object is in frame (wider lens, more distance) pushed COLMAP registration to **100 %** *without adding a single camera or any extra compute*.

Full write-up in the [thesis](#-thesis).

---

## 🧰 Requirements

| Tool | Why | Link |
|---|---|---|
| **Python 3.9+** | runs the orchestrator | [python.org](https://www.python.org/) |
| **COLMAP** (CUDA build) | camera poses + sparse cloud | [colmap.github.io](https://colmap.github.io/) |
| **LichtFeld Studio** | 3D Gaussian Splatting training | [github.com/MrNeRF/LichtFeld-Studio](https://github.com/MrNeRF/LichtFeld-Studio) |
| **NVIDIA GPU** (CUDA) | required by COLMAP-CUDA and 3DGS | — |

The orchestrator itself is stdlib-only. **Background removal** adds a few Python packages:

```bash
pip install "rembg[cpu]" numpy scipy pillow
```

> Without them the pipeline still runs — masking and cropping are skipped with a note, and you get the plain (background-included) model. `rembg` runs fine on CPU (~1–2 s/photo); the GPU is reserved for training.

> Tested on an NVIDIA RTX 4060 Laptop (8 GB). The default cap of 500 000 Gaussians keeps training within 8 GB of VRAM.

---

## 🚀 Installation

```bash
git clone https://github.com/emr81-ua/3d-gaussian-splatting-reconstruction.git
cd 3d-gaussian-splatting-reconstruction
pip install "rembg[cpu]" numpy scipy pillow      # for background removal
```

Then point the script at COLMAP and LichtFeld Studio in **any** of these ways:

- add them to your `PATH`, or
- set environment variables `COLMAP_EXE` and `LICHTFELD_EXE`, or
- drop them under a local `tools/` folder, or
- pass `--colmap-exe` / `--lichtfeld-exe` on the command line.

---

## ▶️ Usage

### Quick start (Windows, no commands)

1. Install the Python deps: `pip install -r requirements.txt`
2. Copy `herramientas.local.bat.example` to **`herramientas.local.bat`** and set your COLMAP and LichtFeld paths there (this file stays local, it is git-ignored).
3. Drop your batch of photos into the **`entrada/`** folder.
4. Double-click **`RECONSTRUIR.bat`**. It asks for the number of iterations (Enter = 15000) and runs the whole pipeline — COLMAP, masks, masked training and crop.
5. The result appears in **`salida/<date_time>/`** with `images/`, `dense/`, `dense/masks/` and the final `model.ply`.

### Command line

```bash
# from a zip of photos (background removed by default)
python reconstruct.py my_photos.zip --iter 15000

# from a folder of photos
python reconstruct.py path/to/photos/ --iter 15000

# quick preview (fewer iterations)
python reconstruct.py my_photos.zip --iter 7000

# keep the background (no masking, no crop)
python reconstruct.py my_photos.zip --no-mask --no-crop

# generic object instead of a person
python reconstruct.py my_photos.zip --mask-model u2net

# COLMAP only (sparse cloud, no training)
python reconstruct.py my_photos.zip --skip-training
```

| Option | Description |
|---|---|
| `--iter N` | training iterations (more = better, slower). Default `15000` |
| `--max-gaussians N` | cap on Gaussians. Lower it if you run out of VRAM. Default `500000` |
| `--no-mask` | keep the background (skip subject segmentation) |
| `--mask-model NAME` | `u2net_human_seg` for people (default), `u2net` for objects |
| `--mask-mode MODE` | `ignore` (default), `segment`, or `alpha_consistent` |
| `--no-crop` | skip the post-training far-floater crop |
| `--crop-factor F` | keep Gaussians within `F ×` the camera-ring radius (default `0.8`) |
| `--crop-opacity O` | drop Gaussians below opacity `O` (default `0` = off; leave it off) |
| `--output DIR` | output folder (default `output/<name>`) |
| `--skip-training` | run COLMAP only |
| `--colmap-exe` / `--lichtfeld-exe` | explicit tool paths |

**Output** lands in `output/<name>/` (or the folder you pass to `--output`; `RECONSTRUIR.bat` uses `salida/<date_time>/`):
- `model.ply` — the final model (masked + cropped, background removed)
- `dense/` — the COLMAP reconstruction (`masks/` included when masking is on)
- `dense/output/` — the raw training splat, before the crop
- `images/` — the input photos

**View the result** with any Gaussian-Splatting viewer, e.g. LichtFeld Studio:
```bash
LichtFeld-Studio --view output/my_photos/model.ply
```

---

## 📸 Tips for good photos

The single most important thing (it was the main finding of the thesis) is **framing**: keep the **whole subject in frame with margin** in every shot — never crop the top or bottom.

- **Whole subject in frame**, with air around it. Step back and use a wider lens rather than zooming in.
- **Go all the way around (360°)**, with **overlap** between consecutive shots, at **2–3 heights** (low looking slightly up, mid, high looking slightly down).
- **40–60 photos** is plenty — distribution matters more than sheer count.
- **Keep the subject still** and shoot quickly; motion between shots breaks pose estimation.
- **Plain, uniform background** and **even lighting** give the cleanest masks and the best result.
- Avoid motion blur and fully smooth / reflective surfaces.

There is a tiny example set under [`examples/`](examples/) so you can try the pipeline right away.

---

## ⚙️ How it works

| Stage | Tool | What happens |
|---|---|---|
| 1. Poses | **COLMAP** | SIFT features → matching → Structure-from-Motion → sparse cloud, undistorted into the 3DGS format |
| 2. Masks | **rembg** (U²-Net) | segments the subject in every image → `masks/` (white = subject) |
| 3. Training | **LichtFeld Studio** | trains a 3D Gaussian Splatting model, ignoring the masked-out background |
| 4. Crop | `crop_splat.py` | removes far background floaters from the trained `.ply` (spatial only; subject untouched) |
| 5. Output | — | the cropped model is written to `model.ply` |

COLMAP parameters are tuned for low-texture / synthetic images (lower SIFT `peak_threshold`, relaxed mapper thresholds) and work well on real photos too.

An optional extra step, `clean_pointcloud.py` (`--clean`), tidies the sparse cloud between COLMAP and training. It only affects the *initialization* — masking is what actually removes the background — so it is off by default.

---

## 🌐 Web viewer

The [`viewer/`](viewer/) folder is a **self-contained WebGL2 viewer** that renders a Gaussian-Splatting model in the browser — real anisotropic splats with depth sorting in a Web Worker, so it stays consistent while you orbit. No install, no GPU driver, works on a phone.

- **Orbit** to rotate, **wheel** to zoom, **right-drag** to pan.
- **Drag & drop** your own `model.ply` (or a `.splat`) onto the page, or use the *Open* button.
- It ships with a small example model (`viewer/example.splat`) so it shows something out of the box.

Run it locally (it needs to be served over HTTP, not opened as a file):

```bash
python -m http.server 8777          # from the repo root
# then open http://localhost:8777/viewer/
```

Convert any trained model to the compact `.splat` format (32 bytes/Gaussian) the viewer loads fastest with:

```bash
python ply_to_splat.py output/my_photos/model.ply --out viewer/example.splat --max 300000
```

> Publish `viewer/` on **GitHub Pages** and anyone can open your 3D model in their browser — the easiest way to let people try the result without the full COLMAP/LichtFeld toolchain.

---

## 🔬 Research scripts

The [`research/`](research/) folder contains the scripts used for the thesis experiments — procedural camera generation in Blender, batch COLMAP/LichtFeld runs and the evaluation code. They are provided as-is for transparency; some paths may need adjusting to your machine.

---

## 📄 Thesis

*Realistic 3D model reconstruction using Gaussian Splatting* — Eric Muñoz Rouillion, University of Alicante, 2026. *(link to the thesis PDF here)*

---

## 🙏 Acknowledgements

- [COLMAP](https://colmap.github.io/) — Schönberger & Frahm
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) — 3D Gaussian Splatting training
- [rembg](https://github.com/danielgatis/rembg) — subject segmentation (U²-Net)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al.

## 📜 License

[MIT](LICENSE) © 2026 Eric Muñoz Rouillion
