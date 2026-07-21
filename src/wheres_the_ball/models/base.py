"""Shared types for VLM ball-localization predictions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator


class BallPrediction(BaseModel):
    """A model's guess for the hidden ball, in normalized image coordinates."""

    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x (0=left, 1=right)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y (0=top, 1=bottom)")
    uncertainty_radius: float = Field(
        0.0, ge=0.0, le=1.5, description="Declared radius of uncertainty (normalized)"
    )
    confidence: float = Field(0.0, ge=0.0, le=100.0, description="Declared confidence 0-100")
    rationale: str = Field("", description="Short justification")

    @field_validator("x", "y", mode="before")
    @classmethod
    def _clip_unit(cls, v: float) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return v
        return min(1.0, max(0.0, v))


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_prediction(raw: str) -> BallPrediction:
    """Parse a model's text answer into a BallPrediction.

    Tolerates markdown code fences and surrounding prose by extracting the first
    JSON object found. Raises ValueError if nothing parseable is present.
    """
    if raw is None:
        raise ValueError("empty model response")
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise ValueError(f"no JSON object in response: {raw[:200]!r}")
        data = json.loads(m.group(0))
    return BallPrediction.model_validate(data)


@dataclass
class Item:
    """One evaluation item: a masked frame plus its ground-truth ball location."""

    item_id: str
    image_path: str  # masked image shown to the model
    original_path: str  # unedited frame (kept for audit / viz)
    gt_x: float  # normalized ground-truth ball center
    gt_y: float
    width: int
    height: int
    leak_flag: bool = False  # set by the leak-control detector
    leak_note: str = ""
