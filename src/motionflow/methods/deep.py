"""Deep-learning optical flow (PyTorch): RAFT, SEA-RAFT and NeuFlow v2.

Each estimator takes BGR ``uint8`` frames and returns ``(H, W, 2)`` float32
flow in pixels, like the classical ones. Weights download on first use; see
:mod:`motionflow.weights`. SEA-RAFT's and NeuFlow v2's model code is
vendored in ``motionflow/methods/_vendor`` at pinned commits.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from motionflow import weights
from motionflow.methods.base import FlowEstimator


def _require_torch() -> Any:
    try:
        import torch
        import torchvision  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Deep learning methods need PyTorch. Install it with:\n"
            '  pip install "motionflow[torch]"\n'
            "or follow https://pytorch.org/get-started/locally/ for a GPU build."
        ) from None
    return torch


def resolve_device(device: str = "auto") -> Any:
    """Turn ``"auto"``, ``"cuda"``, ``"cuda:1"``, ``"mps"`` or ``"cpu"`` into a torch device.

    ``"auto"`` picks CUDA, then Apple's MPS, then the CPU.

    Raises:
        ValueError: For an unknown device name.
        RuntimeError: If the requested device isn't available here.
    """
    torch = _require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    kind = device.split(":")[0]
    if kind not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {device!r}: use auto, cuda, mps or cpu.")
    if kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU is available to this PyTorch (no NVIDIA driver, or a CPU-only "
            "PyTorch build). Use --device cpu, or install a CUDA build of PyTorch."
        )
    if kind == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple's MPS isn't available here. Use --device cpu.")
    return torch.device(device)


def _check_size(method: str, size: str, sizes: tuple[str, ...]) -> str:
    size = getattr(size, "value", size)
    if size not in sizes:
        raise ValueError(f"{method} sizes are {', '.join(sizes)}; got {size!r}.")
    return size


# Smallest frame side the models get; smaller frames are padded up to it.
_MIN_SIZE = 128


class _TorchFlow(FlowEstimator):
    """Shared plumbing: device, tensors, padding."""

    def __init__(self, device: str) -> None:
        self._torch = _require_torch()
        self.device = resolve_device(device)

    def _tensor(self, frame: np.ndarray, rgb: bool) -> Any:
        """(H, W, 3) uint8 -> (1, 3, H, W) float32 on the device, 0-255."""
        image = frame[..., ::-1] if rgb else frame
        tensor = self._torch.from_numpy(np.ascontiguousarray(image))
        return tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self._torch.float32)

    def _pad(self, *tensors: Any, multiple: int) -> tuple[list[Any], tuple[int, int]]:
        """Pad bottom/right by replicating edges, to a multiple of ``multiple``.

        Frames smaller than ``_MIN_SIZE`` are padded up to it: the models'
        correlation pyramids need a few pixels at their coarsest level.
        """
        h, w = tensors[0].shape[-2:]
        pad_h = max(_MIN_SIZE - h, 0) + -max(h, _MIN_SIZE) % multiple
        pad_w = max(_MIN_SIZE - w, 0) + -max(w, _MIN_SIZE) % multiple
        if pad_h or pad_w:
            pad = self._torch.nn.functional.pad
            tensors = tuple(pad(t, (0, pad_w, 0, pad_h), mode="replicate") for t in tensors)
        return list(tensors), (h, w)

    @staticmethod
    def _to_numpy(flow: Any, size: tuple[int, int]) -> np.ndarray:
        """(2, H', W') tensor -> (H, W, 2) float32 array, cropped to ``size``."""
        h, w = size
        return flow[:, :h, :w].permute(1, 2, 0).float().cpu().numpy()


class RAFTFlow(_TorchFlow):
    """RAFT (Teed & Deng, ECCV 2020), torchvision's implementation and weights.

    Args:
        size: ``"small"`` (faster) or ``"large"``.
        device: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
        iters: Refinement iterations; more is slower and slightly better.

    Reference: Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for
    Optical Flow", ECCV 2020. https://arxiv.org/abs/2003.12039
    """

    SIZES = ("small", "large")

    def __init__(self, size: str = "large", device: str = "auto", iters: int = 12) -> None:
        super().__init__(device)
        self.size = _check_size("RAFT", size, self.SIZES)
        if iters < 1:
            raise ValueError(f"iters must be at least 1, got {iters}")
        self.iters = iters
        from torchvision.models import optical_flow as of

        if self.size == "small":
            weights_enum = of.Raft_Small_Weights.DEFAULT
            self._model = of.raft_small(weights=weights_enum, progress=True)
        else:
            weights_enum = of.Raft_Large_Weights.DEFAULT
            self._model = of.raft_large(weights=weights_enum, progress=True)
        self._transforms = weights_enum.transforms()  # uint8 -> float, normalised to [-1, 1]
        self._model = self._model.eval().to(self.device)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            a = torch.from_numpy(np.ascontiguousarray(prev[..., ::-1])).permute(2, 0, 1)[None]
            b = torch.from_numpy(np.ascontiguousarray(curr[..., ::-1])).permute(2, 0, 1)[None]
            a, b = self._transforms(a, b)
            (a, b), size = self._pad(a.to(self.device), b.to(self.device), multiple=8)
            flows = self._model(a, b, num_flow_updates=self.iters)
            return self._to_numpy(flows[-1][0], size)


# SEA-RAFT eval configs (config/eval/spring-{S,M,L}.json upstream): the
# model is the same for M and L, which only runs more iterations.
_SEA_RAFT_CONFIG = {
    "use_var": True,
    "var_min": 0,
    "var_max": 10,
    "initial_dim": 64,
    "block_dims": [64, 128, 256],
    "radius": 4,
    "dim": 128,
    "num_blocks": 2,
}
_SEA_RAFT_SIZES = {
    # size: (backbone, default iterations, weights)
    "small": ("resnet18", 4, "sea-raft-small"),
    "medium": ("resnet34", 4, "sea-raft-medium"),
    "large": ("resnet34", 12, "sea-raft-medium"),
}


class SEARaftFlow(_TorchFlow):
    """SEA-RAFT (Wang, Lipson & Deng, ECCV 2024): simpler, faster RAFT.

    Args:
        size: ``"small"``, ``"medium"`` or ``"large"``. Large is the medium
            model with 12 refinement iterations instead of 4, as upstream.
        device: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
        iters: Refinement iterations; overrides the size's default.

    Reference: Wang, Lipson & Deng, "SEA-RAFT: Simple, Efficient, Accurate
    RAFT for Optical Flow", ECCV 2024. https://arxiv.org/abs/2405.14793
    """

    SIZES = tuple(_SEA_RAFT_SIZES)

    def __init__(
        self, size: str = "medium", device: str = "auto", iters: int | None = None
    ) -> None:
        super().__init__(device)
        self.size = _check_size("SEA-RAFT", size, self.SIZES)
        backbone, default_iters, weight_name = _SEA_RAFT_SIZES[self.size]
        self.iters = iters if iters is not None else default_iters
        if self.iters < 1:
            raise ValueError(f"iters must be at least 1, got {self.iters}")

        from motionflow.methods._vendor.sea_raft import RAFT

        args = SimpleNamespace(**_SEA_RAFT_CONFIG, pretrain=backbone, iters=self.iters)
        args.block_dims = list(args.block_dims)  # the model edits it in place
        model = RAFT(args)
        checkpoint = self._torch.load(
            weights.fetch(weight_name), map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])
        self._model = model.eval().to(self.device)

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            a, b = self._tensor(prev, rgb=True), self._tensor(curr, rgb=True)
            # The model normalises itself and pads to a multiple of 8; we only
            # pad tiny frames up to the minimum size.
            (a, b), size = self._pad(a, b, multiple=8)
            out = self._model(a, b, iters=self.iters, test_mode=True)
            return self._to_numpy(out["final"][0], size)


class NeuFlowV2Flow(_TorchFlow):
    """NeuFlow v2 (Zhang et al., 2024): fast optical flow for edge devices.

    Args:
        device: ``"auto"``, ``"cuda"``, ``"mps"`` or ``"cpu"``.
        half: Run in half precision (float16), as upstream does on GPUs.
            Defaults to ``True`` on CUDA and ``False`` elsewhere.

    Reference: Zhang, Gu, Taylor & Wang, "NeuFlow v2: High-Efficiency
    Optical Flow Estimation on Edge Devices", 2024.
    https://arxiv.org/abs/2408.10161
    """

    def __init__(self, device: str = "auto", half: bool | None = None) -> None:
        super().__init__(device)
        self.half = self.device.type == "cuda" if half is None else half
        from motionflow.methods._vendor.neuflow_v2 import NeuFlow

        model = NeuFlow()
        checkpoint = self._torch.load(
            weights.fetch("neuflow-v2"), map_location="cpu", weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        model = model.eval().to(self.device)
        self._model = model.half() if self.half else model
        self._shape: tuple[int, int] | None = None

    def estimate(self, prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            # Upstream feeds BGR in 0-255; the model divides by 255 itself.
            a, b = self._tensor(prev, rgb=False), self._tensor(curr, rgb=False)
            (a, b), size = self._pad(a, b, multiple=16)
            if self.half:
                a, b = a.half(), b.half()
            shape = tuple(a.shape[-2:])
            if shape != self._shape:
                # The model keeps size-dependent buffers; set them up per size.
                self._model.init_bhwd(1, shape[0], shape[1], self.device, amp=self.half)
                self._shape = shape
            flows = self._model(a, b)
            return self._to_numpy(flows[-1][0], size)
