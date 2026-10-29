# Contributing

## Development setup

motionflow is developed with [pixi](https://pixi.sh), which installs Python and every dependency from conda-forge into `.pixi/` in the repository.

```bash
pixi install                 # the default environment (Python 3.13, dev tools)
pixi run lefthook install    # git hooks: format and lint on commit, tests on push
```

## Common tasks

```bash
pixi run test                # run the tests
pixi run lint                # ruff check
pixi run format              # ruff format
pixi run -e docs docs-serve  # preview the docs site
pixi run -e build build      # build the sdist and wheel into dist/
pixi run motionflow input.mp4
```

## Environments

| Environment | What it's for |
|---|---|
| `default` | Development on Python 3.13 |
| `py310` … `py314` | Testing on each supported Python |
| `torch` | Deep methods, CPU builds of PyTorch |
| `cuda` | Deep methods on an NVIDIA GPU (Linux, Windows); needs a driver supporting CUDA 12.9 |
| `docs` | Building the documentation |
| `build` | Building the release packages |

Run a task in a specific environment with `-e`, e.g. `pixi run -e py310 test`.

## Dependencies

Runtime dependencies are declared twice: in `pyproject.toml` for PyPI users, and in `pixi.toml` for development. Keep them in step.

OpenCV is the exception. PyPI users get `opencv-contrib-python-headless`; in pixi it comes from conda-forge's `opencv`, which includes contrib. `pixi.toml` maps conda-forge's OpenCV to the PyPI OpenCV names, so pixi doesn't also install the wheel. Two OpenCV packages in one environment overwrite each other's `cv2` files.

`pixi.lock` is refreshed by the *Relock* workflow (Actions → Relock → Run workflow), since Dependabot can't update it.
