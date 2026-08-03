# Research scripts (thesis experiments)

These are the scripts used to run the experiments behind the thesis. They are
included **as-is, for transparency** — they are not needed to run the main
`reconstruct.py` pipeline, and some paths are hard-coded to the original machine
(Blender install, Python interpreter, dataset locations), so expect to adjust
them before running.

| Folder | What it does |
|---|---|
| `blender/` | Procedural camera generation and headless rendering of the synthetic dataset. Each experiment is described by a JSON in `blender/experimentos/` (number of cameras, rows, radius, lens…). |
| `colmap/` | Batch COLMAP processing of the experiment datasets. `run_colmap_pipeline.py` is the single-dataset engine that the main `reconstruct.py` is based on. |
| `lichtfield/` | Batch 3D Gaussian Splatting training with LichtFeld Studio. |
| `evaluation/` | Reconstruction-quality evaluation used for the experiments. |

## The two experimental phases

- **Phase 1 — number of views:** 40, 60, 80 and 120 cameras in a circular layout.
  Finding: registration is *not* monotonic in the number of cameras (80 registered
  worse than 60); 60 gave the best quality/cost trade-off.
- **Phase 2 — framing:** with the camera count fixed, widening the lens and moving
  back so the whole subject stays in frame pushed registration to 100 %.

See the thesis for the full methodology and results.
