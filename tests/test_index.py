"""Tests for the recognition gallery (FaceIndex) — no models required."""

from __future__ import annotations

import numpy as np

from facestack import FaceIndex


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_enroll_and_recognize_same_person():
    idx = FaceIndex(dim=512, threshold=0.4)
    base = _vec(1)
    idx.add("alice", base)

    # a near-identical embedding should match alice
    noisy = base + 0.01 * _vec(99)
    m = idx.recognize(noisy)
    assert m is not None
    assert m.person_id == "alice"
    assert m.matched
    assert m.similarity > 0.4


def test_unknown_face_does_not_match():
    idx = FaceIndex(dim=512, threshold=0.4)
    idx.add("alice", _vec(1))
    stranger = _vec(424242)  # orthogonal-ish random vector
    m = idx.recognize(stranger)
    assert m is not None
    assert not m.matched  # below threshold => treated as unknown


def test_multiple_people_picks_closest():
    idx = FaceIndex(dim=512, threshold=0.3)
    a, b = _vec(1), _vec(2)
    idx.add("alice", a)
    idx.add("bob", b)
    assert idx.recognize(a + 0.01 * _vec(7)).person_id == "alice"
    assert idx.recognize(b + 0.01 * _vec(8)).person_id == "bob"


def test_empty_gallery_returns_none():
    idx = FaceIndex(dim=512)
    assert idx.recognize(_vec(1)) is None


def test_remove_person():
    idx = FaceIndex(dim=512, threshold=0.4)
    idx.add("alice", _vec(1))
    assert "alice" in idx.people
    removed = idx.remove_person("alice")
    assert removed == 1
    assert "alice" not in idx.people


def test_save_and_load(tmp_path):
    idx = FaceIndex(dim=512, threshold=0.4)
    a = _vec(1)
    idx.add("alice", a)
    idx.add("bob", _vec(2))

    ip = str(tmp_path / "faces.bin")
    mp = str(tmp_path / "faces.meta.json")
    idx.save(ip, mp)

    loaded = FaceIndex.load(ip, mp)
    assert sorted(loaded.people) == ["alice", "bob"]
    assert loaded.recognize(a + 0.01 * _vec(7)).person_id == "alice"
