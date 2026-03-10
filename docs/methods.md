# Methods

## Classical methods

These run on CPU using OpenCV and require no additional dependencies beyond the base install.

### Farneback (`farneback`)

The default method. Estimates dense flow via polynomial expansion of each frame neighbourhood. Fast, robust, and produces smooth results. Suitable for most use cases.

**Best for:** general-purpose flow, real-time-adjacent processing, long videos where GPU is unavailable.

### TV-L1 (`tvl1`)

A variational method that minimises a total-variation energy functional. Produces sharper motion boundaries than Farneback but is considerably slower.

**Best for:** scenes with hard motion edges, e.g. cut-out animation or text on screen.

!!! warning
    Requires `opencv-contrib-python-headless` on PyPI. On conda-forge, the standard `opencv` package includes contrib.

### Lucas-Kanade (`lucas-kanade`)

Sparse tracker — identifies Shi-Tomasi corners, tracks them across frames with pyramidal Lucas-Kanade, and returns a sparse flow field (zero everywhere except tracked points). Works well with `--viz quiver`.

**Best for:** feature point tracking visualisation, sparse motion analysis.

## Deep learning methods

These require `torch` and `torchvision`. Install with `pip install motionflow[torch]` or use the conda-forge package, which includes PyTorch.

**SEA-RAFT and NeuFlow v2 also require a one-time architecture setup** — the model source code lives in research repos not on PyPI. Run once after install:

```bash
motionflow setup-vendors
```

### RAFT (`raft`)

Recurrent All-Pairs Field Transforms. The model that reset state-of-the-art benchmarks in 2020. Uses the torchvision implementation with pretrained weights downloaded on first use.

Available sizes: `small` (faster) and `large` (more accurate, default).

**Best for:** high-quality flow on scenes with complex or large motion.

**Reference:** Teed & Deng, ECCV 2020. [arXiv:2003.12039](https://arxiv.org/abs/2003.12039)

### SEA-RAFT (`sea-raft`)

Simple, Efficient, Accurate RAFT — ECCV 2024 Best Paper candidate. At least 2.3× faster than prior methods at competitive accuracy. Weights loaded from HuggingFace.

!!! note
    SEA-RAFT model loading is not yet implemented. See [GitHub](https://github.com/princeton-vl/SEA-RAFT) for the architecture — contributions welcome.

**Reference:** Wang et al., ECCV 2024. [arXiv:2405.14793](https://arxiv.org/abs/2405.14793)

### NeuFlow v2 (`neuflow`)

Edge-optimised architecture running 5–110× faster than heavyweight models at comparable accuracy. Designed for deployment on resource-constrained hardware.

!!! note
    NeuFlow v2 model loading is not yet implemented. See [GitHub](https://github.com/neufieldrobotics/NeuFlow_v2) — contributions welcome.

**Reference:** Zhang et al., 2024. [arXiv:2408.10161](https://arxiv.org/abs/2408.10161)

## Choosing a method

| Method | Speed | Quality | Dense | GPU |
|--------|-------|---------|-------|-----|
| `farneback` | Fast | Good | Yes | No |
| `tvl1` | Slow | Better edges | Yes | No |
| `lucas-kanade` | Fast | Sparse | No | No |
| `raft` | Medium | Excellent | Yes | Yes |
| `sea-raft` | Fast | Excellent | Yes | Yes |
| `neuflow` | Very fast | Good | Yes | Yes |
