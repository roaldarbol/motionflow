from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from motionflow.methods.base import FlowEstimator
from motionflow.weights import require_weights, weights_path

_VENDOR_DIR = Path(__file__).parent / "_vendor"


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the best available device string."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Deep learning methods require torch and torchvision. Install with:\n"
            "  pip install motionflow[torch]\n"
            "On conda-forge, pytorch and torchvision are included by default."
        ) from None


def _require_vendor(name: str, sentinel: str) -> None:
    """Ensure vendor architecture files are present, prompting the user if not."""
    sentinel_path = _VENDOR_DIR / name / sentinel
    if sentinel_path.exists():
        return

    import sys

    _UPSTREAM = {
        "sea_raft": "https://github.com/princeton-vl/SEA-RAFT",
        "neuflow_v2": "https://github.com/neufieldrobotics/NeuFlow_v2",
    }
    source = _UPSTREAM.get(name, "upstream repository")

    print(
        f"\n  Architecture files for {name} are not installed.\n"
        f"  Source : {source}\n"
        f"  Destination : {_VENDOR_DIR / name}\n",
        file=sys.stderr,
    )
    while True:
        answer = input("  Download now? [y/N] ").strip().lower()
        if answer in {"y", "yes"}:
            break
        if answer in {"", "n", "no"}:
            raise RuntimeError(
                f"Architecture files for {name} are required but were not downloaded. "
                f"Re-run and accept the prompt, or see {source}."
            )
        print("  Please enter y or n.", file=sys.stderr)

    from motionflow.methods._vendor_setup import setup_neuflow_v2, setup_sea_raft

    if name == "sea_raft":
        setup_sea_raft()
    elif name == "neuflow_v2":
        setup_neuflow_v2()
    else:
        raise RuntimeError(f"No vendor setup defined for {name!r}.")


class RAFTFlow(FlowEstimator):
    """Dense optical flow via RAFT (Recurrent All-Pairs Field Transforms).

    Uses the torchvision implementation with pretrained weights. Weights are
    downloaded automatically on first use and cached by torchvision — no manual
    step required.

    Reference: Teed & Deng, ECCV 2020. https://arxiv.org/abs/2003.12039

    Args:
        size: Model variant — ``"large"`` for best accuracy or ``"small"`` for
            speed.
        device: Compute device. ``"auto"`` selects CUDA > MPS > CPU.
        iters: Number of refinement iterations (higher = more accurate, slower).
    """

    def __init__(
        self,
        size: str = "large",
        device: str = "auto",
        iters: int = 20,
    ) -> None:
        _require_torch()
        import torch

        self._torch = torch
        self._device = _resolve_device(device)
        self._iters = iters

        if size == "small":
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        else:
            from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)

        self._model = self._model.eval().to(self._device)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        h, w = prev.shape[:2]
        prev_t, (pad_h, pad_w) = self._preprocess(prev)
        curr_t, _ = self._preprocess(curr)

        with self._torch.no_grad():
            flows = self._model(prev_t, curr_t, num_flow_updates=self._iters)

        flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()

        if pad_h > 0 or pad_w > 0:
            flow = flow[:h, :w]

        return flow

    def _preprocess(self, frame: np.ndarray) -> tuple[object, tuple[int, int]]:
        h, w = frame.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            self._torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self._device)
        )

        if pad_h > 0 or pad_w > 0:
            import torch.nn.functional as F

            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))

        return tensor, (pad_h, pad_w)


class SEARaftFlow(FlowEstimator):
    """Dense optical flow via SEA-RAFT (Simple, Efficient, Accurate RAFT).

    SEA-RAFT is at least 2.3x faster than prior methods at comparable accuracy.
    Architecture files are vendored locally; weights are downloaded from
    HuggingFace and cached in ``~/.cache/motionflow/weights/``.

    **First-time setup required** — run once before using this method:

    .. code-block:: bash

        motionflow setup-vendors

    If weights are not cached locally, motionflow will prompt before downloading.

    Reference: Wang et al., ECCV 2024 (Best Paper candidate).
    https://arxiv.org/abs/2405.14793

    Args:
        size: Model variant — ``"large"`` or ``"medium"``.
        device: Compute device. ``"auto"`` selects CUDA > MPS > CPU.
    """

    _HF_REPOS: dict[str, str] = {
        "large": "MemorySlices/Tartan-C-T-TSKH-spring540x960-L",
        "medium": "MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
    }
    _FILENAMES: dict[str, str] = {
        "large": "sea_raft_large.pth",
        "medium": "sea_raft_medium.pth",
    }

    def __init__(self, size: str = "large", device: str = "auto") -> None:
        _require_torch()
        _require_vendor("sea_raft", "sea_raft.py")
        self._device = _resolve_device(device)
        self._size = size
        self._model = self._load_model(size)

    def _load_model(self, size: str) -> object:
        import sys

        import torch

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise RuntimeError(
                "SEA-RAFT requires huggingface-hub. Install with:\n  pip install huggingface-hub"
            ) from None

        if size not in self._HF_REPOS:
            raise ValueError(f"Unknown SEA-RAFT size: {size!r}. Choose 'large' or 'medium'.")

        hf_repo = self._HF_REPOS[size]
        filename = self._FILENAMES[size]

        def _download() -> None:
            hf_path = hf_hub_download(repo_id=hf_repo, filename="model.pth")
            shutil.copy2(hf_path, weights_path(filename))

        weights = require_weights(
            method=f"SEA-RAFT ({size})",
            source=f"https://huggingface.co/{hf_repo}",
            filename=filename,
            downloader=_download,
        )

        # Import vendored architecture
        vendor_dir = str(_VENDOR_DIR / "sea_raft")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from sea_raft import RAFT as SEARaft  # type: ignore[import]

        model = SEARaft()
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint)
        return model.eval().to(self._device)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        import torch

        h, w = prev.shape[:2]
        prev_t = self._to_tensor(prev)
        curr_t = self._to_tensor(curr)

        with torch.no_grad():
            output = self._model(prev_t, curr_t)

        # SEA-RAFT returns a dict with a "flow" key
        flow = output["flow"][0].permute(1, 2, 0).cpu().numpy()
        return flow[:h, :w]

    def _to_tensor(self, frame: np.ndarray) -> object:
        import torch

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self._device)


class NeuFlowV2Flow(FlowEstimator):
    """Dense optical flow via NeuFlow v2.

    NeuFlow v2 is optimised for efficiency, running 5–110x faster than
    heavyweight methods at comparable accuracy, originally targeting edge devices.

    **First-time setup required** — run once before using this method:

    .. code-block:: bash

        motionflow setup-vendors

    Weights are downloaded from GitHub releases and cached in
    ``~/.cache/motionflow/weights/``. If not cached, motionflow will prompt
    before downloading.

    Reference: Zhang et al., 2024. https://arxiv.org/abs/2408.10161

    Args:
        device: Compute device. ``"auto"`` selects CUDA > MPS > CPU.
    """

    _WEIGHT_URL = (
        "https://github.com/neufieldrobotics/NeuFlow_v2/releases/download/v1.0/neuflow_v2.pth"
    )
    _FILENAME = "neuflow_v2.pth"

    def __init__(self, device: str = "auto") -> None:
        _require_torch()
        _require_vendor("neuflow_v2", "neuflow.py")
        self._device = _resolve_device(device)
        self._model = self._load_model()

    def _load_model(self) -> object:
        import sys

        import torch

        def _download() -> None:
            import urllib.request

            dest = weights_path(self._FILENAME)
            print(
                f"  Downloading NeuFlow v2 weights from {self._WEIGHT_URL} ...",
                flush=True,
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                urllib.request.urlretrieve(self._WEIGHT_URL, tmp.name)
                shutil.copy2(tmp.name, dest)
                Path(tmp.name).unlink(missing_ok=True)

        weights = require_weights(
            method="NeuFlow v2",
            source=self._WEIGHT_URL,
            filename=self._FILENAME,
            downloader=_download,
        )

        # Import vendored architecture
        vendor_dir = str(_VENDOR_DIR / "neuflow_v2")
        if vendor_dir not in sys.path:
            sys.path.insert(0, vendor_dir)
        from neuflow import NeuFlow  # type: ignore[import]

        model = NeuFlow()
        checkpoint = torch.load(weights, map_location="cpu")
        model.load_state_dict(checkpoint)
        return model.eval().to(self._device)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        import torch

        h, w = prev.shape[:2]
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        prev_t = self._to_tensor(prev, pad_h, pad_w)
        curr_t = self._to_tensor(curr, pad_h, pad_w)

        with torch.no_grad():
            flow = self._model(prev_t, curr_t)

        flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
        return flow_np[:h, :w]

    def _to_tensor(self, frame: np.ndarray, pad_h: int, pad_w: int) -> object:
        import torch
        import torch.nn.functional as F

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self._device)
        )
        if pad_h > 0 or pad_w > 0:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
        return tensor
