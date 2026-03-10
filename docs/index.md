# motionflow

**motionflow** computes and visualises optical flow from video files. It ships as both a command-line tool and a Python library, so it works equally well as a standalone script, in a CI pipeline, or inside a Jupyter notebook.

## Quick start

```bash
# Install (OpenCV-only — classical methods)
pip install motionflow

# Install with deep learning support
pip install motionflow[torch]

# Run on a video
motionflow input.mp4
```

This produces `input_flow.mp4` in the same directory using the Farneback method with HSV colour-wheel visualisation.

## Features

- Six optical flow methods: Farneback, TV-L1, Lucas-Kanade, RAFT, SEA-RAFT, NeuFlow v2
- Four visualisation styles: HSV colour wheel, magnitude heatmap, quiver overlay, blend
- Streaming frame-by-frame processing — handles multi-hour videos without loading into memory
- GPU acceleration for deep methods via PyTorch (CUDA, MPS, CPU)
- Python API mirrors the CLI exactly — usable in notebooks or custom pipelines
- Available on PyPI and conda-forge
