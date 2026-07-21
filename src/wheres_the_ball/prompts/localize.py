"""Versioned prompts for the ball-localization task.

Keep prompt text here (versioned) so results are reproducible and diffable.
"""
from __future__ import annotations

PROMPT_VERSION = "v0-neutral"

# Neutral prompt: does NOT name the sport or give tactical heuristics.
# (RQ3 will contrast this with an "informed" prompt in the full experiment.)
LOCALIZE_NEUTRAL = """You are shown a single frame from a team sport. The object the \
players are contesting (the ball) has been removed from the image and is NOT visible.

Your task: infer where that object most likely is, using only the positions, body \
orientation, and apparent motion of the players.

Answer with a STRICT JSON object and nothing else:
{
  "x": <float 0..1, normalized horizontal position, 0=left edge, 1=right edge>,
  "y": <float 0..1, normalized vertical position, 0=top edge, 1=bottom edge>,
  "uncertainty_radius": <float 0..1, how far off you might be, as a fraction of image width>,
  "confidence": <int 0..100>,
  "rationale": "<one short sentence: what in the players' configuration led you there>"
}"""

# Informed prompt: names the sport and gives spectator heuristics (for RQ3 later).
LOCALIZE_INFORMED = """You are shown a single frame from a football (soccer) match. The \
ball has been removed from the image and is NOT visible.

Infer where the ball most likely is. Useful cues a spectator uses: players orient their \
bodies and heads toward the ball; several players converge on or run toward it; the \
player in possession is usually slightly ahead of a cluster; defenders position between \
the ball and their goal.

Answer with a STRICT JSON object and nothing else:
{
  "x": <float 0..1, 0=left, 1=right>,
  "y": <float 0..1, 0=top, 1=bottom>,
  "uncertainty_radius": <float 0..1, fraction of image width>,
  "confidence": <int 0..100>,
  "rationale": "<one short sentence>"
}"""

# Temporal condition (RQ2): a short sequence of consecutive frames (oldest→newest),
# the ball removed from ALL of them. The model infers the ball in the LAST frame.
LOCALIZE_TEMPORAL_NEUTRAL = """You are shown a short sequence of consecutive frames \
from a team sport (about one second apart, earliest first, last frame most recent). \
The object the players are contesting (the ball) has been removed from EVERY frame and \
is NOT visible in any of them.

Use how the players move and reorient across the frames to infer where that object is \
IN THE LAST (most recent) frame.

Answer with a STRICT JSON object and nothing else, with coordinates relative to the \
LAST frame:
{
  "x": <float 0..1, 0=left edge, 1=right edge>,
  "y": <float 0..1, 0=top edge, 1=bottom edge>,
  "uncertainty_radius": <float 0..1, fraction of image width>,
  "confidence": <int 0..100>,
  "rationale": "<one short sentence: what motion cue led you there>"
}"""

PROMPTS = {
    "neutral": LOCALIZE_NEUTRAL,
    "informed": LOCALIZE_INFORMED,
    "temporal": LOCALIZE_TEMPORAL_NEUTRAL,
}

# Leak control: run on the MASKED image to check the edit left no usable cue.
LEAK_CHECK = """This image may have been digitally edited to remove a small ball. Look \
carefully. Answer with STRICT JSON and nothing else:
{
  "ball_visible": <true if you can actually see a ball, else false>,
  "artifact_visible": <true if you see a blur, smear, patch or edit that reveals where \
something was removed, else false>,
  "where": "<if either is true, roughly where (e.g. 'lower left'); else empty>"
}"""
