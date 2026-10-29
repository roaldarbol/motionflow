# motionflow

**motionflow** computes and visualises optical flow from video files. It ships as both a command-line tool and a Python library, so it works equally well as a standalone script, in a CI pipeline, or inside a Jupyter notebook.

## Quick start

```bash
# Install (OpenCV-only — classical methods)
pip install motionflow

# Install with deep learning support
pip install "motionflow[torch]"

# Run on a video
motionflow input.mp4
```

This produces `input_flow.mp4` in the same directory using the DIS method with HSV colour-wheel visualisation.

## Features

- Optical flow methods: DIS, Farneback, TV-L1 and Lucas-Kanade on the CPU; RAFT, SEA-RAFT and NeuFlow v2 with PyTorch
- Chromatic motion (BehaveAI's encoding): recent movement blue, older green and red
- Event camera simulation, with the event stream saved as HDF5, NumPy or text
- Visualisation styles: HSV colour wheel, speed heatmap, quiver arrows; any of them can be drawn over the original frame, with a colour key
- One colour scale for the whole video by default, so the same colour means the same speed in every frame
- Streaming frame-by-frame processing — handles multi-hour videos without loading into memory
- GPU acceleration for deep methods via PyTorch (CUDA, MPS, CPU)
- Python API with the same options as the CLI, usable in notebooks or custom pipelines
- Available on PyPI (conda-forge planned)
