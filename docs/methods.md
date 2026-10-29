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

## Chromatic motion (`motionflow chromatic`)

The motion encoding of [BehaveAI](https://github.com/troscianko/BehaveAI). Each frame is compared with three references of increasing age, and the differences become colour channels drawn over a dimmed greyscale frame:

- **blue**: change since the previous frame (recent motion);
- **green**: change against a medium-term running average;
- **red**: change against a long-term running average (older motion).

A moving animal leaves a trail that fades from blue through green to red, while still parts of the scene stay grey. It isn't optical flow: it's frame differencing, so it's fast (tens of milliseconds per 720p frame) and doesn't need texture to track, but it shows *where* things changed rather than *which way* they moved.

Options:

- `--strategy exponential` (default) uses running averages that fade at rates `--exp-a` (green, default 0.5) and `--exp-b` (red, 0.8); `--strategy sequential` uses the last three frames instead.
- `--gain` amplifies the differences (one value, or `recent,medium,older`); `--threshold` subtracts a constant to hide noise.
- `--motion-only` draws the motion on black; `--lum-weight` sets the brightness of the greyscale frame otherwise.
- `--tail-only` keeps in each colour only the motion the more recent one doesn't show, separating the bands.
- `--frame-skip N` uses every (N+1)-th frame, as BehaveAI does: in slow-motion or high-frame-rate video, the motion between the frames used is larger. The output then has fewer frames.

motionflow's implementation is written from the method's description (BehaveAI is AGPL-licensed; motionflow is MIT). It keeps the running averages in floating point, so it can differ slightly from BehaveAI's CPU output, which rounds them every frame.

**Best for:** seeing where and when an animal moves, behaviour classification inputs, very fast processing.

**Reference:** Troscianko, O'Shea-Wheller, Galloway & Gaston, "BehaveAI enables rapid detection and classification of objects and behavior from motion", *PLOS Biology* 24(2): e3003632 (2026). [doi:10.1371/journal.pbio.3003632](https://doi.org/10.1371/journal.pbio.3003632)

## Event camera simulation (`motionflow events`)

An event camera (dynamic vision sensor, DVS) doesn't record frames: each pixel independently reports an *event* whenever its brightness changes by more than a contrast threshold, an ON event when it gets brighter and an OFF event when it gets darker. `motionflow events` simulates that from ordinary video, so you can see what an event camera would record, or produce event data for event-based methods.

Each pixel keeps a reference log brightness; when the frame's log brightness differs from it by more than the threshold `--threshold` (default 0.2, about a 22% change), the pixel emits one event per threshold crossed and its reference moves by that many thresholds. When a pixel fires several events between two frames, they get timestamps spread linearly over the interval.

The output video draws the events: ON blue, OFF red, on black (`--render polarity`), or the number of events per pixel as a heatmap (`--render count`). `--window N` draws the events of the last N frames together, as event cameras are usually shown; `--overlay` draws them over the greyscale video. `--save-events h5|npz|txt` also writes the event stream as `(t, x, y, p)` records, `p` being 1 for ON and 0 for OFF:

- `h5`: HDF5, datasets `events/t` (microseconds), `events/x`, `events/y`, `events/p`; needs h5py (`pip install "motionflow[events]"`).
- `npz`: NumPy arrays with the same names (held in memory, so for short clips).
- `txt`: one event per line, `t x y p`, with `t` in seconds, as in the event-camera datasets of Mueggler et al. (2017).

This is the ideal-sensor core model: no noise, every pixel identical, no bandwidth limit or refractory period. High-frame-rate video works best, since the simulator can't know what happened between frames. More realistic sensor effects are planned ([#23](https://github.com/roaldarbol/motionflow/issues/23)).

Event streams are large: one second of 240 fps, 2.7K footage of a hammer throw gave about 10 million events.

**References:** Lichtsteiner, Posch & Delbruck, "A 128×128 120 dB 15 µs latency asynchronous temporal contrast vision sensor", *IEEE Journal of Solid-State Circuits* 2008. Rebecq, Gehrig & Scaramuzza, "ESIM: an open event camera simulator", CoRL 2018. Hu, Liu & Delbruck, "v2e: From video frames to realistic DVS events", CVPR Workshops 2021.

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
