"""Tests for the TTL-aware body gallery (BodyIndex) — no models required."""

from __future__ import annotations

import numpy as np

from facestack.bodyindex import BodyIndex


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# A fixed reference "now" and a one-hour TTL keep timestamp math explicit.
NOW = 1_700_000_000.0
TTL = 3600.0


def test_add_and_recognize_within_ttl():
    idx = BodyIndex(dim=512, threshold=0.4)
    base = _vec(1)
    idx.add("alice", base, ts=NOW)

    noisy = base + 0.01 * _vec(99)
    m = idx.recognize(noisy, now=NOW + 60.0, ttl=TTL)
    assert m is not None
    assert m.person_id == "alice"
    assert m.matched
    assert m.similarity > 0.4


def test_ttl_expiry_ignores_old_embedding():
    idx = BodyIndex(dim=512, threshold=0.4)
    base = _vec(1)
    # Enrolled two hours ago; with a one-hour TTL it is expired.
    idx.add("alice", base, ts=NOW - 2 * TTL)

    noisy = base + 0.01 * _vec(99)
    m = idx.recognize(noisy, now=NOW, ttl=TTL)
    assert m is None


def test_recognize_at_exact_cutoff_is_kept():
    # search filters with `ts < now - ttl`, so ts == cutoff survives.
    idx = BodyIndex(dim=512, threshold=0.4)
    base = _vec(1)
    idx.add("alice", base, ts=NOW - TTL)
    m = idx.recognize(base, now=NOW, ttl=TTL)
    assert m is not None
    assert m.person_id == "alice"


def test_unknown_body_does_not_match():
    idx = BodyIndex(dim=512, threshold=0.4)
    idx.add("alice", _vec(1), ts=NOW)
    stranger = _vec(424242)
    m = idx.recognize(stranger, now=NOW, ttl=TTL)
    assert m is not None
    assert not m.matched  # below threshold => treated as unknown


def test_multiple_people_picks_closest():
    idx = BodyIndex(dim=512, threshold=0.3)
    a, b = _vec(1), _vec(2)
    idx.add("alice", a, ts=NOW)
    idx.add("bob", b, ts=NOW)
    assert idx.recognize(a + 0.01 * _vec(7), now=NOW, ttl=TTL).person_id == "alice"
    assert idx.recognize(b + 0.01 * _vec(8), now=NOW, ttl=TTL).person_id == "bob"


def test_purge_removes_expired_only():
    idx = BodyIndex(dim=512, threshold=0.4)
    fresh = _vec(1)
    stale = _vec(2)
    idx.add("alice", fresh, ts=NOW)
    idx.add("bob", stale, ts=NOW - 2 * TTL)
    assert len(idx) == 2

    purged = idx.purge(now=NOW, ttl=TTL)
    assert purged == 1
    assert len(idx) == 1
    assert idx.people == ["alice"]
    # bob (the expired one) is gone; only alice remains tracked.
    assert "bob" not in idx.people
    # purge is idempotent: re-purging finds nothing new to remove.
    assert idx.purge(now=NOW, ttl=TTL) == 0
    assert len(idx) == 1
    # Searching right after a purge must NOT crash: search clamps its over-fetch
    # to the live element count, so the surviving person is still recognised even
    # though _next_label still counts the purged label.
    m = idx.recognize(fresh, now=NOW, ttl=TTL)
    assert m is not None and m.person_id == "alice"


def test_remove_person():
    idx = BodyIndex(dim=512, threshold=0.4)
    idx.add("alice", _vec(1), ts=NOW)
    idx.add("alice", _vec(2), ts=NOW)  # multiple embeddings per person
    idx.add("bob", _vec(3), ts=NOW)
    assert "alice" in idx.people

    removed = idx.remove_person("alice")
    assert removed == 2
    assert "alice" not in idx.people
    assert idx.people == ["bob"]


def test_save_and_load_preserves_timestamps(tmp_path):
    idx = BodyIndex(dim=512, threshold=0.4)
    a = _vec(1)
    b = _vec(2)
    idx.add("alice", a, ts=NOW - TTL)  # right at the cutoff boundary
    idx.add("bob", b, ts=NOW)

    ip = str(tmp_path / "bodies.bin")
    mp = str(tmp_path / "bodies.meta.json")
    idx.save(ip, mp)

    loaded = BodyIndex.load(ip, mp)
    assert sorted(loaded.people) == ["alice", "bob"]
    # Timestamps survived the round-trip: alice (ts == cutoff) is kept,
    # but if her ts had not been preserved as NOW - TTL she would expire.
    assert loaded._label_to_ts == {0: NOW - TTL, 1: NOW}
    assert loaded.recognize(a, now=NOW, ttl=TTL).person_id == "alice"
    assert loaded.recognize(b + 0.01 * _vec(7), now=NOW, ttl=TTL).person_id == "bob"

    # And TTL is still enforced after load: advance now past alice's expiry.
    # alice's embedding is now ignored; the only surviving entry is bob, so the
    # query for alice's vector must NOT come back as alice (recognize returns the
    # nearest *surviving* person regardless of the match threshold).
    m = loaded.recognize(a, now=NOW + 1.0, ttl=TTL)
    assert m is None or m.person_id != "alice"


def test_empty_gallery_returns_none():
    idx = BodyIndex(dim=512)
    assert idx.recognize(_vec(1), now=NOW, ttl=TTL) is None
    assert idx.search(_vec(1), now=NOW, ttl=TTL, k=3) == []
    assert len(idx) == 0
    assert idx.people == []
