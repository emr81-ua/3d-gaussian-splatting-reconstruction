<!-- Language: English | [Español](README.es.md) -->

# Photo → 3D · Gaussian Splatting pipeline

**Turn a handful of photos into a 3D model you can explore in real time.** One command takes a `.zip` of images and runs the whole pipeline — camera-pose estimation with COLMAP and 3D Gaussian Splatting training with LichtFeld Studio — and hands you a `.ply` you can open in any Gaussian-Splatting viewer.

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

## 🌐 Try it in your browser — no install

Upload a `.zip` of photos and get a 3D model, running on Google's **free** GPU. Nothing to install.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emr81-ua/3d-gaussian-splatting-reconstruction/blob/main/colab/reconstruccion_3d.ipynb)

*(The Colab uses [nerfstudio](https://docs.nerf.studio/)'s `splatfacto` trainer so it can run for free in the cloud; the desktop pipeline below uses LichtFeld Studio.)*

---

## ✨ What it does

```bash
python reconstruct.py my_photos.zip
```

```
my_photos.zip  ──▶  COLMAP  ──▶  LichtFeld Studio  ──▶  model.ply
   (photos)         (poses +        (3D Gaussian         (real-time
                   sparse cloud)     Splatting)           3D model)
```

- **One entry point.** A `.zip` (or folder) of photos in, a `model.ply` out.
- **Fully automated**, no manual steps between stages.
- **Zero Python dependencies** — the orchestrator uses only the standard library.
- Also runnable by **double-clicking** `reconstruct.bat` (Windows) or dropping a zip onto it.

![Pipeline](docs/pipeline.png)

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

> Tested on an NVIDIA RTX 4060 Laptop (8 GB). The default cap of 500 000 Gaussians keeps training within 8 GB of VRAM.

---

## 🚀 Installation

```bash
git clone https://github.com/emr81-ua/3d-gaussian-splatting-reconstruction.git
cd 3d-gaussian-splatting-reconstruction
```

Then point the script at COLMAP and LichtFeld Studio in **any** of these ways:

- add them to your `PATH`, or
- set environment variables `COLMAP_EXE` and `LICHTFELD_EXE`, or
- drop them under a local `tools/` folder, or
- pass `--colmap-exe` / `--lichtfeld-exe` on the command line.

---

## ▶️ Usage

```bash
# from a zip of photos
python reconstruct.py my_photos.zip --iter 15000

# from a folder of photos
python reconstruct.py path/to/photos/ --iter 15000

# quick preview (fewer iterations)
python reconstruct.py my_photos.zip --iter 2000

# COLMAP only (sparse cloud, no training)
python reconstruct.py my_photos.zip --skip-training
```

| Option | Description |
|---|---|
| `--iter N` | training iterations (more = better, slower). Default `15000` |
| `--max-gaussians N` | cap on Gaussians. Lower it if you run out of VRAM. Default `500000` |
| `--output DIR` | output folder (default `output/<name>`) |
| `--skip-training` | run COLMAP only |
| `--colmap-exe` / `--lichtfeld-exe` | explicit tool paths |

**Output** lands in `output/<name>/`:
- `model.ply` — the final 3D Gaussian Splatting model
- `dense/` — the COLMAP reconstruction
- `images/` — the input photos

**View the result** with any Gaussian-Splatting viewer, e.g. LichtFeld Studio:
```bash
LichtFeld-Studio --view output/my_photos/model.ply
```

---

## 📸 Tips for good photos

- **30–60 photos** going all the way around the object, with **overlap** between consecutive shots.
- Keep the object **still**, with even lighting and a background that has some texture.
- Avoid motion blur and fully smooth / reflective surfaces.

There is a tiny example set under [`examples/`](examples/) so you can try the pipeline right away.

---

## ⚙️ How it works

| Stage | Tool | What happens |
|---|---|---|
| 1. Poses | **COLMAP** | SIFT features → matching → Structure-from-Motion → sparse point cloud, undistorted into the 3DGS format |
| 2. Training | **LichtFeld Studio** | trains a 3D Gaussian Splatting model on the posed images |
| 3. Output | — | the latest splat is copied to `model.ply` |

COLMAP parameters are tuned for low-texture / synthetic images (lower SIFT `peak_threshold`, relaxed mapper thresholds) and work well on real photos too.

---

## 🔬 Research scripts

The [`research/`](research/) folder contains the scripts used for the thesis experiments — procedural camera generation in Blender, batch COLMAP/LichtFeld runs and the evaluation code. They are provided as-is for transparency; some paths may need adjusting to your machine.

---

## 📄 Thesis

*Realistic 3D model reconstruction using Gaussian Splatting* — Eric Muñoz Rouillion, University of Alicante, 2026. *(link to the memoir / PDF here)*

---

## 🙏 Acknowledgements

- [COLMAP](https://colmap.github.io/) — Schönberger & Frahm
- [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) — 3D Gaussian Splatting training
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) — Kerbl et al.

## 📜 License

[MIT](LICENSE) © 2026 Eric Muñoz Rouillion
