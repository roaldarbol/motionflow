# Installation

## PyPI (pip / uv)

The base install includes OpenCV and covers all classical methods (Farneback, TV-L1, Lucas-Kanade):

```bash
pip install motionflow
# or
uv tool install motionflow
```

To also enable deep learning methods (RAFT, SEA-RAFT, NeuFlow v2), install with the `torch` extra. This adds `torch` and `torchvision` as CPU builds by default:

```bash
pip install motionflow[torch]
```

For GPU-accelerated deep methods, install PyTorch separately first using the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version, then install motionflow:

```bash
# Example: CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install motionflow
```

### TV-L1 method

TV-L1 requires the OpenCV contrib modules, which are not included in `opencv-python-headless`. Install `opencv-contrib-python-headless` instead:

```bash
pip install opencv-contrib-python-headless motionflow
```

On conda-forge, `opencv` includes contrib by default so no additional step is needed.

## conda-forge (conda / pixi / mamba)

The conda-forge package includes all dependencies including PyTorch. The solver automatically picks the correct GPU or CPU build based on your system:

```bash
conda install -c conda-forge motionflow
# or
pixi add motionflow
```

### Global tool install with Pixi

```bash
pixi global install motionflow
```

Pixi detects whether CUDA is available and installs the appropriate PyTorch variant automatically.

## SEA-RAFT and NeuFlow v2: one-time setup

These two methods use research model architectures that are not on PyPI. After installing motionflow, run this once to download the architecture source files from their upstream GitHub repositories:

```bash
motionflow setup-vendors
```

This only needs to be done once per installation. The files are placed inside the motionflow package directory. Model weights are then downloaded separately on first use (with a prompt). To force a re-download of the architecture files:

```bash
motionflow setup-vendors --force
```

## Verifying the install

```bash
motionflow --version
python -c "import motionflow; print(motionflow.__version__)"
```
