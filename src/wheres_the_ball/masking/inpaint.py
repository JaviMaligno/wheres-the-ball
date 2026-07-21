"""Remove the ball from a frame by inpainting, using its bbox as the mask.

Rationale (see docs/fase-0-viabilidad.md): a blur/patch leaks the ball position via a
visible artifact. Inpainting reconstructs a coherent background so the model must infer
the ball from the players, not spot an edit. A leak-control step (elsewhere) still
verifies no residual cue remains.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BBox:
    """Normalized bounding box (YOLO-style center + size), all in [0, 1]."""

    cx: float
    cy: float
    w: float
    h: float

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = int(round((self.cx - self.w / 2) * width))
        y1 = int(round((self.cy - self.h / 2) * height))
        x2 = int(round((self.cx + self.w / 2) * width))
        y2 = int(round((self.cy + self.h / 2) * height))
        return x1, y1, x2, y2


def build_mask(
    height: int, width: int, bbox: BBox, dilate_frac: float = 0.6,
    extra_pad_px: float = 0.0,
) -> np.ndarray:
    """White (255) over the ball region, dilated to also cover shadow/halo.

    dilate_frac is a fraction of the ball's larger side (small pixel floor so tiny
    balls still get a workable margin). extra_pad_px adds absolute padding — used to
    cover motion blur on fast balls, whose visible streak exceeds the tight GT bbox.
    """
    x1, y1, x2, y2 = bbox.to_pixels(width, height)
    ball_px = max(x2 - x1, y2 - y1, 1)
    pad = max(int(round(ball_px * dilate_frac)), 4) + int(round(extra_pad_px))
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    x2, y2 = min(x2 + pad, width), min(y2 + pad, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def write_mask(path: str, height: int, width: int, bbox: BBox,
               dilate_frac: float = 0.6, extra_pad_px: float = 0.0) -> None:
    """Save a white-on-black mask PNG (white = ball region) for external inpainters
    like LaMa/iopaint, which expect image + mask pairs."""
    cv2.imwrite(path, build_mask(height, width, bbox, dilate_frac=dilate_frac,
                                 extra_pad_px=extra_pad_px))


def inpaint_ball(
    image_bgr: np.ndarray, bbox: BBox, dilate_frac: float = 0.6, radius: int = 3,
    extra_pad_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (inpainted_image, mask). Uses Telea fast marching inpainting.

    The ball is tiny over low-frequency background (grass/stands), so classical
    inpainting is usually seamless and instant. Fall back to LaMa only if not.
    """
    h, w = image_bgr.shape[:2]
    mask = build_mask(h, w, bbox, dilate_frac=dilate_frac, extra_pad_px=extra_pad_px)
    out = cv2.inpaint(image_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    return out, mask
