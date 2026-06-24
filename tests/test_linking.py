"""Tests for pure-geometry face<->body linking — no models required."""

from __future__ import annotations

from facestack.linking import link_faces_to_bodies

# Default containment threshold used across most cases.
MIN_CONT = 0.5


def test_face_contained_in_single_body():
    # One body spanning a tall person; the face sits near the top, fully inside.
    body = (0.0, 0.0, 100.0, 300.0)
    face = (40.0, 10.0, 60.0, 50.0)  # center (50, 30), in top 40% (<120)
    result = link_faces_to_bodies([face], [body], MIN_CONT)
    assert result == [0]


def test_face_in_upper_region_links_but_lower_does_not():
    body = (0.0, 0.0, 100.0, 300.0)
    # Upper region cutoff is by1 + 0.4*height = 120.
    upper_face = (40.0, 90.0, 60.0, 130.0)  # center y = 110 < 120 -> linked
    lower_face = (40.0, 200.0, 60.0, 240.0)  # center y = 220 > 120 -> rejected
    result = link_faces_to_bodies([upper_face, lower_face], [body], MIN_CONT)
    assert result == [0, None]


def test_nested_bodies_picks_best_containment_then_smallest_area():
    # A large loose box and a tight box both contain the face, both upper region.
    # Tight box contains the whole face (containment 1.0) -> wins on containment.
    face = (45.0, 10.0, 55.0, 30.0)  # center (50, 20)
    big = (0.0, 0.0, 100.0, 300.0)        # contains face fully too
    tight = (40.0, 0.0, 60.0, 80.0)       # contains face fully, smaller area
    result = link_faces_to_bodies([face], [big, tight], MIN_CONT)
    assert result == [1]  # tie on containment(1.0) -> smaller area wins


def test_equal_containment_smallest_area_tiebreak():
    # Two bodies that each fully contain the face; smaller one must win.
    face = (45.0, 10.0, 55.0, 30.0)  # center (50, 20)
    larger = (0.0, 0.0, 200.0, 400.0)
    smaller = (0.0, 0.0, 100.0, 200.0)
    # Both fully contain the face (containment 1.0) and center is in upper region.
    result = link_faces_to_bodies([face], [larger, smaller], MIN_CONT)
    assert result == [1]


def test_face_with_no_containing_body_returns_none():
    body = (0.0, 0.0, 100.0, 300.0)
    far_face = (500.0, 500.0, 520.0, 540.0)  # center way outside the body
    result = link_faces_to_bodies([far_face], [body], MIN_CONT)
    assert result == [None]


def test_low_containment_rejected():
    # Face center inside body upper region, but most of the face spills outside,
    # so containment falls below the threshold.
    body = (0.0, 0.0, 100.0, 300.0)
    # Face from x=90..130 -> only 10/40 width inside -> containment 0.25 < 0.5.
    face = (90.0, 10.0, 130.0, 50.0)  # center (110, 30): cx=110 > 100 -> outside
    assert link_faces_to_bodies([face], [body], MIN_CONT) == [None]
    # Shift so center is inside but containment still low.
    face2 = (75.0, 10.0, 115.0, 50.0)  # center x = 95 inside; inside width 25/40
    assert link_faces_to_bodies([face2], [body], 0.7) == [None]


def test_multiple_faces_multiple_bodies():
    body_left = (0.0, 0.0, 100.0, 300.0)
    body_right = (200.0, 0.0, 300.0, 300.0)
    face_left = (40.0, 10.0, 60.0, 50.0)    # center (50, 30) -> body_left (idx 0)
    face_right = (240.0, 10.0, 260.0, 50.0)  # center (250, 30) -> body_right (1)
    face_none = (400.0, 400.0, 420.0, 440.0)  # outside both -> None
    result = link_faces_to_bodies(
        [face_left, face_right, face_none],
        [body_left, body_right],
        MIN_CONT,
    )
    assert result == [0, 1, None]
