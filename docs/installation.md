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

## Model weights

The deep methods download their weights the first time they're used, with a progress bar, and check each file against a checksum:

| Method | Size | From |
|---|---|---|
| `raft` | 4 MB (small), 21 MB (large) | torchvision, into PyTorch's cache (`~/.cache/torch`) |
| `sea-raft` | 36 MB (small), 79 MB (medium, large) | GitHub ([ptlflow](https://github.com/hmorimitsu/ptlflow)'s mirror of the authors' weights) |
| `neuflow` | 36 MB | GitHub ([the authors' repository](https://github.com/neufieldrobotics/NeuFlow_v2)) |

SEA-RAFT's and NeuFlow v2's weights go to `~/.cache/motionflow/weights`; set `MOTIONFLOW_CACHE_DIR` to put the cache somewhere else. To download everything ahead of time, for example on a cluster's login node before a batch job on nodes without internet:

```bash
motionflow download              # all deep methods
motionflow download sea-raft     # just one
```

`motionflow methods` shows which methods can run and whether their weights are cached.

The model code of SEA-RAFT (BSD 3-Clause) and NeuFlow v2 (Apache 2.0) is included in motionflow; see their licences in `motionflow/methods/_vendor/`.

Writing event streams as HDF5 (`motionflow events --save-events h5`) needs h5py:

```bash
pip install "motionflow[events]"
```

## conda-forge (conda / pixi / mamba)

A conda-forge package is planned but not available yet. Until then, install from PyPI.

## Verifying the install

```bash
motionflow --version
python -c "import motionflow; print(motionflow.__version__)"
```

## Development install

See [CONTRIBUTING.md](https://github.com/roaldarbol/motionflow/blob/main/CONTRIBUTING.md): development uses [pixi](https://pixi.sh).
