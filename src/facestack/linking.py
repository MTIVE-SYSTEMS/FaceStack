"""Pure-geometry face<->body association.

When we see faces and bodies in the same frame we need to know which face
belongs to which body so a recognized face can lend its identity to the body
(for later body-only recognition). This is deliberately model-free: it operates
on plain (x1, y1, x2, y2) bboxes so it is trivially unit-testable and carries no
heavy dependencies. A face belongs to the body whose box best *contains* the
face and whose upper region the face sits in (faces ride near the top of a body).
"""

from __future__ import annotations

import logging

log = logging.getLogger("facestack.linking")

Bbox = tuple[float, float, float, float]  # (x1, y1, x2, y2)

# A face should sit near the top of the body it belongs to: we require its
# center within the top 40% of the body's height (some slack below "the head"
# so a slightly-low or large-head crop still links).
_UPPER_REGION_FRAC = 0.4


def _area(box: Bbox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: Bbox, b: Bbox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _center(box: Bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _containment(face: Bbox, body: Bbox) -> float:
    """Fraction of the face box that lies inside the body box, in [0, 1]."""
    fa = _area(face)
    if fa <= 0:
        return 0.0
    return _intersection_area(face, body) / fa


def link_faces_to_bodies(
    faces: list[Bbox],
    bodies: list[Bbox],
    min_containment: float,
) -> list[int | None]:
    """Map each face to the index of its best-matching body, or None.

    A face is eligible for a body when:
      * the face bbox center falls inside the body bbox, AND
      * the face sits in the body's upper region (center y in the top fraction
        of the body's height), AND
      * the body contains at least `min_containment` of the face's area.

    Among eligible bodies we pick the one with the highest containment of the
    face; ties break toward the smallest body area (the tighter, more specific
    box). Returns a list parallel to `faces`.
    """
    result: list[int | None] = []
    for face in faces:
        fcx, fcy = _center(face)
        best_idx: int | None = None
        best_containment = -1.0
        best_area = float("inf")
        for bi, body in enumerate(bodies):
            bx1, by1, bx2, by2 = body
            # Center must be inside the body box.
            if not (bx1 <= fcx <= bx2 and by1 <= fcy <= by2):
                continue
            # Face center must be in the body's upper region.
            bh = by2 - by1
            if bh <= 0:
                continue
            if fcy > by1 + _UPPER_REGION_FRAC * bh:
                continue
            cont = _containment(face, body)
            if cont < min_containment:
                continue
            area = _area(body)
            # Higher containment wins; ties -> smaller body.
            if cont > best_containment or (cont == best_containment and area < best_area):
                best_containment = cont
                best_area = area
                best_idx = bi
        result.append(best_idx)
    return result
