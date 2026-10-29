# Methods

## Classical methods

These run on CPU using OpenCV and require no additional dependencies beyond the base install.

### DIS (`dis`)

The default method. Dense Inverse Search matches small patches between frames, coarse to fine, then smooths the result with a variational refinement. Three presets trade speed for accuracy: `ultrafast`, `fast` and `medium` (the default); in Python, `DISFlow(preset="fast")`.

On test sequences with known motion (translation, rotation with zoom, and an object moving over a still background), the `medium` preset was 3–4× faster than Farneback and more accurate:

| 1920×1080, 8 CPU threads | Time per frame | Error, rotation + zoom | Error, moving object |
|---|---|---|---|
| `dis` (medium) | 101 ms | 0.04 px | 0.02 px |
| `dis`, `fast` preset | 20 ms | 0.11 px | 0.05 px |
| `farneback` | 305 ms | 3.18 px | 0.03 px |

Farneback's large error under rotation and zoom comes from motions of up to ~40 px near the corners, more than it can follow with its default settings. The numbers come from [`benchmarks/flow_methods.py`](https://github.com/roaldarbol/motionflow/blob/main/benchmarks/flow_methods.py); times vary with the machine.

**Best for:** almost everything without a GPU.

**Reference:** Kroeger, Timofte, Dai & Van Gool, "Fast Optical Flow using Dense Inverse Search", ECCV 2016. [arXiv:1603.03590](https://arxiv.org/abs/1603.03590)

### Farneback (`farneback`)

Estimates dense flow via polynomial expansion of each frame neighbourhood. Produces smooth results, but is slower than DIS and loses track of large motions. The default before motionflow 0.2.

**Best for:** reproducing results made with motionflow 0.1.

**Reference:** Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion", SCIA 2003.

### TV-L1 (`tvl1`)

A variational method that minimises a total-variation energy functional. Produces sharp motion boundaries but is considerably slower than DIS.

**Best for:** scenes with hard motion edges, e.g. cut-out animation or text on screen.

Uses OpenCV's contrib modules, which motionflow installs.

### Lucas-Kanade (`lucas-kanade`)

Sparse tracker: tracks points from one frame to the next with pyramidal Lucas-Kanade. By default the points are Shi-Tomasi corners; in Python, `LucasKanadeFlow(points="grid", grid_step=16)` tracks a regular grid instead. The flow is zero except at the tracked points, which are drawn as dots (`hsv`, `heatmap`) or arrows (`quiver`).

**Best for:** feature point tracking visualisation, sparse motion analysis.

## Deep learning methods

These require `torch` and `torchvision`. Install with `pip install "motionflow[torch]"`; see [Installation](installation.md) for choosing a PyTorch build.

### RAFT (`raft`)

Recurrent All-Pairs Field Transforms. The model that reset state-of-the-art benchmarks in 2020. Uses the torchvision implementation with pretrained weights downloaded on first use.

Available sizes: `small` (faster) and `large` (more accurate, default).

!!! warning
    `--model-size small` is currently ignored on the command line, so RAFT always uses the large model there ([#11](https://github.com/roaldarbol/motionflow/issues/11)). The Python API's `model_size="small"` works.

**Best for:** high-quality flow on scenes with complex or large motion.

**Reference:** Teed & Deng, ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)

### SEA-RAFT (`sea-raft`)

Simple, Efficient, Accurate RAFT — ECCV 2024 Best Paper candidate. At least 2.3× faster than prior methods at competitive accuracy. Weights loaded from HuggingFace.

!!! warning "Not working in this release"
    SEA-RAFT can't be loaded yet. It is being reworked, likely on top of [ptlflow](https://github.com/hmorimitsu/ptlflow) ([#16](https://github.com/roaldarbol/motionflow/issues/16)).

**Reference:** Wang et al., ECCV 2024. [arXiv:2405.14793](https://arxiv.org/abs/2405.14793)

### NeuFlow v2 (`neuflow`)

Edge-optimised architecture running 5–110× faster than heavyweight models at comparable accuracy. Designed for deployment on resource-constrained hardware.

!!! warning "Not working in this release"
    NeuFlow v2 can't be loaded yet. It is being reworked, likely on top of [ptlflow](https://github.com/hmorimitsu/ptlflow) ([#16](https://github.com/roaldarbol/motionflow/issues/16)).

**Reference:** Zhang et al., 2024. [arXiv:2408.10161](https://arxiv.org/abs/2408.10161)

## Choosing a method

| Method | Speed | Quality | Dense | GPU |
|--------|-------|---------|-------|-----|
| `dis` | Fast | Very good | Yes | No |
| `farneback` | Medium | Good; misses large motion | Yes | No |
| `tvl1` | Slow | Better edges | Yes | No |
| `lucas-kanade` | Fast | Sparse | No | No |
| `raft` | Medium | Excellent | Yes | Yes |
| `sea-raft` | Fast | Excellent | Yes | Yes |
| `neuflow` | Very fast | Good | Yes | Yes |
