# Example dataset

A tiny set of 12 rendered views of a 3D character (Mixamo model, royalty-free),
downscaled so you can try the pipeline right away:

```bash
python reconstruct.py examples/example_head --iter 2000
```

`--iter 2000` gives a quick preview. Use `--iter 15000` for a proper result.

The output will be written to `output/example_head/model.ply`.
