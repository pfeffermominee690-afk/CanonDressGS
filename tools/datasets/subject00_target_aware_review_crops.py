"""Target-aware display-only crop boxes for Subject00 review assets."""

from __future__ import annotations


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def review_crop_boxes(
    person_bbox: tuple[int, int, int, int], width: int, height: int
) -> dict[str, tuple[int, int, int, int]]:
    """Derive review boxes from the detected/mask-derived person bbox.

    This replaces the invalid fixed full-frame face crop. Back views use the
    same head-region evidence box and must be labeled ``face not visible``.
    """
    x0, y0, x1, y1 = person_bbox
    pw, ph = x1 - x0, y1 - y0
    pad_x = max(16, round(pw * 0.22))
    return {
        "face_head": clamp_box(
            (x0 - pad_x, y0 - round(ph * 0.10), x1 + pad_x, y0 + round(ph * 0.30)),
            width,
            height,
        ),
        "neck_shoulder": clamp_box(
            (x0 - pad_x, y0 + round(ph * 0.12), x1 + pad_x, y0 + round(ph * 0.43)),
            width,
            height,
        ),
        "garment_boundary": clamp_box(
            (x0 - pad_x, y0 + round(ph * 0.24), x1 + pad_x, y0 + round(ph * 0.78)),
            width,
            height,
        ),
        "hands_feet": clamp_box(
            (x0 - round(pw * 0.35), y0 + round(ph * 0.45), x1 + round(pw * 0.35), y1 + round(ph * 0.05)),
            width,
            height,
        ),
    }
