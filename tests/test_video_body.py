"""Video body-track identity persistence — no models required.

The headline behaviour: once a body track's face is confidently matched, the
identity must SURVIVE the person turning away (face gone), and the new back/side
view must be enrolled so the body gallery learns them from behind too.
"""

from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from facestack.bodyengine import DetectedBody
from facestack.bodyindex import BodyIndex
from facestack.config import Config
from facestack.index import Match
from facestack.video import VideoRecognizer

BODY_BBOX = (100.0, 100.0, 200.0, 400.0)  # a standing person
FACE_BBOX = (130.0, 110.0, 170.0, 160.0)  # top-centre, inside the body's upper region


def _vec(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _fake_rec(face_visible_emb, body_emb):
    """A Recognizer stub: faces matched only while face_visible_emb is set."""
    state = {"face_emb": face_visible_emb, "body_emb": body_emb}
    body_index = BodyIndex(dim=512, threshold=0.5)

    def detect(_img):
        # (bbox, det_score, kps); kps must be non-None to trigger face embed.
        if state["face_emb"] is None:
            return []
        return [(FACE_BBOX, 0.95, np.zeros((5, 2), dtype=np.float32))]

    def embed_aligned(_img, _kps):
        return state["face_emb"]

    def index_recognize(_emb):
        return Match(person_id="alice", similarity=0.9, matched=True)

    def detect_and_embed(_img):
        return [DetectedBody(bbox=BODY_BBOX, det_score=0.88, embedding=state["body_emb"])]

    rec = SimpleNamespace(
        config=Config(enable_body=True, reid_interval=1),
        engine=SimpleNamespace(detect=detect, embed_aligned=embed_aligned),
        index=SimpleNamespace(recognize=index_recognize),
        body_engine=SimpleNamespace(detect_and_embed=detect_and_embed),
        body_index=body_index,
        _body_enabled=True,
        _already_enrolled=lambda pid, emb, now: False,  # always enrol in this test
    )
    return rec, state, body_index


def test_identity_persists_through_turn_and_back_view_enrolled():
    front, back = _vec(1), _vec(2)
    rec, state, body_index = _fake_rec(front, front)
    vr = VideoRecognizer(rec, rec.config)
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: face visible & matched -> body track confirmed as alice (front view).
    persons = vr.process_frame_persons(img, now=1000.0)
    alice = [p for p in persons if p.person_id == "alice"]
    assert alice, "face-visible frame should identify alice"
    assert body_index.recognize(front, now=1000.0, ttl=rec.config.body_ttl_seconds).person_id == "alice"

    # Frame 2: person turned away -> NO face, same body box (same track), new view.
    state["face_emb"] = None       # face no longer detectable
    state["body_emb"] = back       # back-of-body looks different
    persons = vr.process_frame_persons(img, now=1001.0)

    # Identity must persist via the track even though the face is gone.
    body_persons = [p for p in persons if p.source == "body"]
    assert body_persons, "should still emit a person for the now-faceless body"
    assert body_persons[0].person_id == "alice"
    assert body_persons[0].matched is True

    # And the back view must now be enrolled under alice (gallery learned it).
    assert body_index.recognize(back, now=1001.0, ttl=rec.config.body_ttl_seconds).person_id == "alice"


def test_unconfirmed_track_does_not_invent_identity():
    # A body that was never face-confirmed and matches nothing stays unknown.
    rec, state, _ = _fake_rec(None, _vec(7))  # face never visible
    vr = VideoRecognizer(rec, rec.config)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    persons = vr.process_frame_persons(img, now=2000.0)
    body = [p for p in persons if p.source == "body"]
    assert body and body[0].person_id is None and body[0].matched is False
