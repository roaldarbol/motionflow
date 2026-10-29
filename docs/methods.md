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

These need PyTorch: install with `pip install "motionflow[torch]"`, and see [Installation](installation.md) for choosing a PyTorch build. They run on a CPU, slowly (seconds per 720p frame), and much faster on an NVIDIA GPU (`--device cuda`) or Apple silicon (`--device mps`). Their weights download on first use; see [Model weights](installation.md#model-weights).

On a known (+3, −2) px shift at 330×250, all three recover the motion to within a fraction of a pixel (mean endpoint error, from the test suite): SEA-RAFT 0.01 px, NeuFlow v2 0.08 px, RAFT large 0.08 px, RAFT small 0.2 px.

### RAFT (`raft`)

Recurrent All-Pairs Field Transforms, the model that reset optical flow benchmarks in 2020. torchvision's implementation and weights.

Sizes: `small` (faster) and `large` (default). In Python, `RAFTFlow(iters=12)` sets the number of refinement iterations.

**Reference:** Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)

### SEA-RAFT (`sea-raft`)

Simple, Efficient, Accurate RAFT: a simplified RAFT that is faster and more accurate. The most accurate method in our tests.

Sizes: `small`, `medium` (4 refinement iterations), and `large` (the medium model with 12 iterations, as in the paper). The command line's default size, `large`, is the most accurate; `--model-size small` is about 1.7× faster. The weights were trained on several datasets including Spring, a high-resolution benchmark of natural scenes.

**Reference:** Wang, Lipson & Deng, "SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow", ECCV 2024. [arXiv:2405.14793](https://arxiv.org/abs/2405.14793)

### NeuFlow v2 (`neuflow`)

A compact model designed for speed on small devices: the fastest deep method here. On an NVIDIA GPU it runs in half precision (float16), as its authors do.

**Reference:** Zhang, Gu, Taylor & Wang, "NeuFlow v2: High-Efficiency Optical Flow Estimation on Edge Devices", 2024. [arXiv:2408.10161](https://arxiv.org/abs/2408.10161)

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
