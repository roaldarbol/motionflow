# Installation

## PyPI (pip / uv)

The base install includes OpenCV with its contrib modules and covers all classical methods (DIS, Farneback, TV-L1, Lucas-Kanade):

```bash
pip install motionflow
# or, as a command-line tool in its own environment
uv tool install motionflow
```

!!! warning "One OpenCV package only"
    motionflow depends on `opencv-contrib-python-headless`. Don't install another OpenCV wheel (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`) into the same environment: they all install into the same `cv2` folder and break each other. If `import cv2` fails or TV-L1 reports that it isn't available, uninstall every OpenCV wheel and reinstall `opencv-contrib-python-headless`.

To also enable the deep learning methods, install with the `torch` extra. The quotes stop shells like zsh from treating the brackets as a pattern:

```bash
pip install "motionflow[torch]"
```

Which PyTorch build this gets depends on your platform: on Linux, PyTorch from PyPI includes CUDA (a large download); on Windows it is CPU-only; on macOS it supports Apple-silicon GPUs (MPS). To choose a specific build, such as CPU-only on Linux or a particular CUDA version, install PyTorch first using the [official instructions](https://pytorch.org/get-started/locally/), then motionflow:

```bash
# Example: CPU-only PyTorch on Linux
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install motionflow
```

!!! note "Deep methods in this release"
    RAFT works. SEA-RAFT and NeuFlow v2 don't work in this release; they are being reworked ([#16](https://github.com/roaldarbol/motionflow/issues/16)).

## conda-forge (conda / pixi / mamba)

A conda-forge package is planned but not available yet. Until then, install from PyPI.

## Verifying the install

```bash
motionflow --version
python -c "import motionflow; print(motionflow.__version__)"
```

## Development install

See [CONTRIBUTING.md](https://github.com/roaldarbol/motionflow/blob/main/CONTRIBUTING.md): development uses [pixi](https://pixi.sh).
