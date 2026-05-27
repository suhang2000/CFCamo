"""Parse a Qwen3-VL response under the CFCamo schema.

CFCamo schema (set by CFCAMO_SYSTEM_PROMPT):
  Detect: <think>...</think> <bbox>[x1,y1,x2,y2]</bbox>  (or a nested array)
  Abstain: <think>...</think> <no_camouflage/>

bbox coords are in 0~1000 (Qwen-native canonical, not 0~1).

format_valid: whether the output matches the CFCamo schema (used for the SFT
format-pass rate and the RL schema bonus):
  - detect valid: <think> + >=1 bbox, all coords in [0,1000] with x2>x1, y2>y1
  - abstain valid: <think> + abstain marker, no bbox
  - otherwise format_valid=False
(points/labels are parsed for compatibility with baseline outputs but are not
required by the schema.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ============ regexes ============
# DOTALL: think body may span lines
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
# bbox: accepts <bbox>[x1,y1,x2,y2]</bbox> (flat single bbox) and
# <bbox>[[x1,y1,x2,y2], [x3,y3,x4,y4], ...]</bbox> (the model occasionally
# outputs a nested list). _BBOX_BLOCK_RE grabs <bbox>...</bbox>; _BBOX_INNER_RE
# grabs any 4-tuple inside.
_BBOX_BLOCK_RE = re.compile(r"<bbox>(.*?)</bbox>", re.DOTALL)
_BBOX_INNER_RE = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
# points / labels: parsed only for compatibility with baseline (e.g. Seg-R1)
# outputs; not part of the CFCamo schema.
_POINTS_BLOCK_RE = re.compile(r"<points>\s*\[(.*?)\]\s*</points>", re.DOTALL)
_PT_RE = re.compile(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_LABELS_BLOCK_RE = re.compile(r"<labels>\s*\[(.*?)\]\s*</labels>", re.DOTALL)
_LABEL_RE = re.compile(r"-?\d+")
# abstain marker: <no_camouflage/> or <no_camouflage> (LLMs often drop the slash)
_NOCAM_RE = re.compile(r"<no_camouflage\s*/?\s*>")

# Qwen-native canonical coordinate range
_COORD_MIN = 0
_COORD_MAX = 1000


@dataclass
class ParsedResponse:
    kind: str                    # "detect" | "refuse" | "invalid"
    format_valid: bool = False   # schema valid: <think> + bbox, or <think> + <no_camouflage/>
    thinking: str = ""           # <think> body (without the tags)
    bboxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    points: list[tuple[int, int]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    refuse: bool = False         # abstain marker detected
    conflict: bool = False       # bbox and abstain marker both present
    bbox_out_of_range: bool = False  # any coord outside [0, 1000]
    bbox_inverted: bool = False  # any bbox with x2<x1 or y2<y1


def _parse_bboxes(text: str) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for blk in _BBOX_BLOCK_RE.finditer(text):
        for m in _BBOX_INNER_RE.finditer(blk.group(1)):
            out.append(tuple(int(g) for g in m.groups()))
    return out


def _parse_points(text: str) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for blk in _POINTS_BLOCK_RE.finditer(text):
        for m in _PT_RE.finditer(blk.group(1)):
            pts.append((int(m.group(1)), int(m.group(2))))
    return pts


def _parse_labels(text: str) -> list[int]:
    labs: list[int] = []
    for blk in _LABELS_BLOCK_RE.finditer(text):
        for m in _LABEL_RE.finditer(blk.group(1)):
            labs.append(int(m.group(0)))
    return labs


def parse_response(text: str) -> ParsedResponse:
    if not text:
        return ParsedResponse(kind="invalid", format_valid=False, thinking="")

    # think
    tm = _THINK_RE.search(text)
    thinking = tm.group(1).strip() if tm else ""

    # components
    bboxes = _parse_bboxes(text)
    points = _parse_points(text)
    labels = _parse_labels(text)
    has_refuse = bool(_NOCAM_RE.search(text))

    # validity sub-checks
    out_of_range = any(
        not (_COORD_MIN <= c <= _COORD_MAX) for bb in bboxes for c in bb
    )
    inverted = any(x2 <= x1 or y2 <= y1 for x1, y1, x2, y2 in bboxes)

    has_bbox = len(bboxes) > 0
    has_think = bool(thinking)

    # conflict: bbox and abstain marker both present
    conflict = has_bbox and has_refuse

    if conflict:
        # self-contradictory: neither detect nor abstain
        kind = "invalid"
        format_valid = False
    elif has_refuse:
        kind = "refuse"
        # abstain valid: think + abstain marker only, no bbox/points/labels
        format_valid = has_think and not has_bbox and not points and not labels
    elif has_bbox:
        kind = "detect"
        # detect valid: think + bbox with valid coords
        format_valid = (
            has_think and has_bbox
            and not out_of_range and not inverted
        )
    else:
        # neither bbox nor abstain marker = invalid
        kind = "invalid"
        format_valid = False

    return ParsedResponse(
        kind=kind,
        format_valid=format_valid,
        thinking=thinking,
        bboxes=bboxes,
        points=points,
        labels=labels,
        refuse=has_refuse,
        conflict=conflict,
        bbox_out_of_range=out_of_range,
        bbox_inverted=inverted,
    )
