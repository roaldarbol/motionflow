"""Speed and accuracy of the classical flow methods on motion with known ground truth.

Run with ``pixi run python benchmarks/flow_methods.py``. The numbers in
docs/methods.md come from this script.

Three cases per frame size, each a smooth random texture:
- translation by (3, -2) px;
- rotation by 1.5 degrees with 2% zoom about the centre (motion grows
  towards the corners, ~40 px at 1080p);
- an object moving (5, 3) px over a still background.

Error is the mean endpoint error (distance between estimated and true flow
vectors) in pixels, excluding a 32 px border. Times are the best of three.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from motionflow.methods import DISFlow, FarnebackFlow

rng = np.random.default_rng(0)


def texture(h: int, w: int) -> np.ndarray:
    base = rng.integers(0, 256, (h // 4, w // 4), dtype=np.uint8)
    base = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.GaussianBlur(base, (5, 5), 1.5), cv2.COLOR_GRAY2BGR)


def cases(h: int, w: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    out = {}

    t = texture(h, w)
    M = np.float32([[1, 0, 3], [0, 1, -2]])
    gt = np.dstack([np.full((h, w), 3, np.float32), np.full((h, w), -2, np.float32)])
    out["translation"] = (t, cv2.warpAffine(t, M, (w, h), borderMode=cv2.BORDER_REFLECT), gt)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.02).astype(np.float32)
    gt = np.dstack(
        [M[0, 0] * xs + M[0, 1] * ys + M[0, 2] - xs, M[1, 0] * xs + M[1, 1] * ys + M[1, 2] - ys]
    )
    out["rotation+zoom"] = (t, cv2.warpAffine(t, M, (w, h), borderMode=cv2.BORDER_REFLECT), gt)

    bg, obj = texture(h, w), texture(h // 3, w // 3)
    y0, x0 = h // 3, w // 3
    oh, ow = obj.shape[:2]
    f1, f2 = bg.copy(), bg.copy()
    f1[y0 : y0 + oh, x0 : x0 + ow] = obj
    f2[y0 + 3 : y0 + 3 + oh, x0 + 5 : x0 + 5 + ow] = obj
    gt = np.zeros((h, w, 2), np.float32)
    gt[y0 : y0 + oh, x0 : x0 + ow] = (5, 3)
    out["object"] = (f1, f2, gt)
    return out


def main() -> None:
    methods = {
        "dis (medium)": DISFlow("medium"),
        "dis (fast)": DISFlow("fast"),
        "dis (ultrafast)": DISFlow("ultrafast"),
        "farneback": FarnebackFlow(),
    }
    for h, w in [(720, 1280), (1080, 1920)]:
        print(f"\n{w}x{h}, {cv2.getNumThreads()} OpenCV threads")
        data = cases(h, w)
        print(f"  {'method':16s} {'ms/frame':>9s}  " + "  ".join(f"{c:>13s}" for c in data))
        for name, estimator in methods.items():
            times, errors = [], []
            for a, b, gt in data.values():
                best = float("inf")
                for _ in range(3):
                    start = time.perf_counter()
                    flow = estimator(a, b)
                    best = min(best, time.perf_counter() - start)
                times.append(best)
                errors.append(np.linalg.norm((flow - gt)[32:-32, 32:-32], axis=2).mean())
            print(
                f"  {name:16s} {1000 * np.mean(times):9.1f}  "
                + "  ".join(f"{e:11.3f}px" for e in errors)
            )


if __name__ == "__main__":
    main()
