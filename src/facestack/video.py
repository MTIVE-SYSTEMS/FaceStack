"""Live-video recognition.

Recognising every face on every frame wastes the GPU: a person's identity does
not change while they stay in frame. So we run lightweight IoU tracking and only
recompute the embedding + gallery match once per track (and refresh every
reid_interval frames). Detection still runs each frame to keep boxes tight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .linking import link_faces_to_bodies
from .recognizer import RecognizedFace, Recognizer

log = logging.getLogger("facestack.video")


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    person_id: str | None = None
    similarity: float = 0.0
    matched: bool = False
    age: int = 0  # frames since last seen
    frames_since_reid: int = 1_000_000  # force re-id on first sight
    det_score: float = 0.0
    # --- body linkage (only used when config.enable_body) ---
    source: str = "face"  # 'face' | 'body'
    body_bbox: tuple[float, float, float, float] | None = None
    # Throttles body auto-enroll / body recognize, reusing reid_interval.
    frames_since_body_reid: int = 1_000_000
    # Sticky once this body track's face was confidently matched: identity then
    # persists through the turn (face hidden) instead of being re-queried and lost.
    face_confirmed: bool = False


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    bbox: tuple[float, float, float, float]
    person_id: str | None
    similarity: float
    matched: bool


@dataclass(slots=True)
class TrackedPerson:
    """Unified track record (face track or body-only track), used when body on."""

    track_id: int
    bbox: tuple[float, float, float, float]  # face bbox if present else body bbox
    person_id: str | None
    similarity: float
    matched: bool
    source: str  # 'face' | 'body'
    body_bbox: tuple[float, float, float, float] | None = None


class VideoRecognizer:
    """Stateful per-stream frame processor. One instance per camera/stream."""

    def __init__(self, recognizer: Recognizer, config: Config | None = None):
        self.rec = recognizer
        self.config = config or recognizer.config
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 0
        # Body state — only ever touched when the recognizer has body enabled.
        self._body_enabled = getattr(self.rec, "_body_enabled", False)
        self._body_tracks: dict[int, Track] = {}
        self._next_body_track_id = 1_000_000  # disjoint id range from face tracks

    def process_frame(self, img_bgr: np.ndarray, now: float | None = None) -> list[TrackedFace]:
        """Face-track this frame and return TrackedFace records (legacy contract).

        Return type and behaviour are unchanged from before; `now` is an
        optional injectable timestamp (defaults to time.time()).
        """
        if now is None:
            import time

            now = time.time()
        results, _ = self._process_faces(img_bgr, now)
        self._evict()
        return results

    def _process_faces(
        self, img_bgr: np.ndarray, now: float
    ) -> tuple[list[TrackedFace], list[Track]]:
        """Run the (unchanged) face-tracking pipeline.

        Returns the TrackedFace list and the aligned list of touched face Track
        objects (so the body pipeline can link faces to bodies this frame).
        """
        # Detection only — runs every frame. Embedding is deferred to re-id below.
        detections = self.rec.engine.detect(img_bgr)

        # age all tracks; survivors get reset when matched below
        for t in self._tracks.values():
            t.age += 1
            t.frames_since_reid += 1

        assigned: set[int] = set()
        results: list[TrackedFace] = []
        touched: list[Track] = []

        for bbox, det_score, kps in detections:
            tid = self._assign(bbox, assigned)
            if tid is None:
                tid = self._spawn(bbox, det_score)
            assigned.add(tid)
            track = self._tracks[tid]
            track.bbox = bbox
            track.det_score = det_score
            track.age = 0

            # Embed + match ONLY on first sight or every reid_interval frames.
            # This is the optimization: identity is cached per track, so the
            # heavy ArcFace embedding does not run on every frame.
            if track.frames_since_reid >= self.config.reid_interval and kps is not None:
                emb = self.rec.engine.embed_aligned(img_bgr, kps)
                m = self.rec.index.recognize(emb)
                track.person_id = m.person_id if (m and m.matched) else None
                track.similarity = m.similarity if m else 0.0
                track.matched = bool(m and m.matched)
                track.frames_since_reid = 0

            results.append(
                TrackedFace(
                    track_id=tid,
                    bbox=track.bbox,
                    person_id=track.person_id,
                    similarity=track.similarity,
                    matched=track.matched,
                )
            )
            touched.append(track)

        return results, touched

    def process_frame_persons(
        self, img_bgr: np.ndarray, now: float | None = None
    ) -> list[TrackedPerson]:
        """Unified per-frame pipeline: face tracks + body-only tracks.

        Runs the same face tracking as process_frame, then (when body enabled)
        detects/tracks bodies, auto-enrolls a confidently-matched face's body
        embedding (throttled by reid_interval), and recognises faceless bodies
        via the body gallery. Returns TrackedPerson records.
        """
        if now is None:
            import time

            now = time.time()

        face_results, face_tracks = self._process_faces(img_bgr, now)

        if not self._body_enabled:
            # No body work: surface the face tracks as persons unchanged.
            self._evict()
            return [
                TrackedPerson(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    person_id=t.person_id,
                    similarity=t.similarity,
                    matched=t.matched,
                    source="face",
                    body_bbox=None,
                )
                for t in face_results
            ]

        # Detect + embed bodies this frame.
        bodies = self.rec.body_engine.detect_and_embed(img_bgr)

        # Age all body tracks.
        for bt in self._body_tracks.values():
            bt.age += 1
            bt.frames_since_body_reid += 1

        # IoU-track bodies against existing body tracks.
        assigned: set[int] = set()
        body_track_for_idx: list[int] = []  # body index -> body track id
        for body in bodies:
            tid = self._assign_body(body.bbox, assigned)
            if tid is None:
                tid = self._spawn_body(body.bbox, body.det_score)
            assigned.add(tid)
            bt = self._body_tracks[tid]
            bt.bbox = body.bbox
            bt.body_bbox = body.bbox
            bt.det_score = body.det_score
            bt.age = 0
            body_track_for_idx.append(tid)

        # Link current-frame face tracks to current-frame bodies.
        links = link_faces_to_bodies(
            [t.bbox for t in face_tracks],
            [b.bbox for b in bodies],
            self.config.body_link_containment,
        )
        body_to_face: dict[int, int] = {}
        for fi, bj in enumerate(links):
            if bj is not None and bj not in body_to_face:
                body_to_face[bj] = fi

        ran_body_reid = False

        def _enrol(pid: str) -> None:
            # Add the current-view embedding for pid, throttled + deduped. As a
            # tracked person rotates this captures front -> side -> back views.
            nonlocal ran_body_reid
            if bt.frames_since_body_reid >= self.config.reid_interval:
                if not self.rec._already_enrolled(pid, body.embedding, now):
                    self.rec.body_index.add(pid, body.embedding, ts=now)
                bt.frames_since_body_reid = 0
                ran_body_reid = True

        for bj, body in enumerate(bodies):
            bt = self._body_tracks[body_track_for_idx[bj]]
            ftrack = face_tracks[body_to_face[bj]] if bj in body_to_face else None
            face_match = ftrack is not None and ftrack.matched and ftrack.person_id is not None

            if face_match:
                # Face visible & matched: (re)confirm identity, enrol the front view.
                bt.person_id = ftrack.person_id
                bt.matched = True
                bt.similarity = ftrack.similarity
                bt.source = "face"
                bt.face_confirmed = True
                _enrol(ftrack.person_id)
            elif bt.face_confirmed and bt.person_id is not None:
                # Same continuous track, face now hidden (turned away): KEEP the
                # identity from when the face was visible — this is the whole point
                # of body recognition — and keep enrolling the new (side/back) view
                # so the gallery learns this person from behind too.
                bt.matched = True
                bt.source = "body"
                _enrol(bt.person_id)
            else:
                # Never identified on this track (e.g. entered back-first): try the
                # body gallery — works if they were seen face-first elsewhere/earlier.
                if bt.frames_since_body_reid >= self.config.reid_interval:
                    m = self.rec.body_index.recognize(
                        body.embedding, now=now, ttl=self.config.body_ttl_seconds
                    )
                    bt.person_id = m.person_id if (m and m.matched) else None
                    bt.similarity = m.similarity if m else 0.0
                    bt.matched = bool(m and m.matched)
                    bt.source = "body"
                    bt.frames_since_body_reid = 0
                    ran_body_reid = True

        # Periodic housekeeping — only when we actually ran a body re-id, to
        # avoid churn (search filters TTL itself, so correctness is unaffected).
        if ran_body_reid:
            self.rec.body_index.purge(now, self.config.body_ttl_seconds)

        self._evict()
        self._evict_body()

        # Emit faces first, then body-only tracks (those with no linked face).
        persons: list[TrackedPerson] = []
        linked_body_track_ids = {body_track_for_idx[bj] for bj in body_to_face}
        for fi, t in enumerate(face_results):
            bj = links[fi]
            # Only attach the body to the face that owns it (body_to_face winner),
            # so two faces linking to one body don't both claim its region.
            own_body = bj is not None and body_to_face.get(bj) == fi
            persons.append(
                TrackedPerson(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    person_id=t.person_id,
                    similarity=t.similarity,
                    matched=t.matched,
                    source="face",
                    body_bbox=bodies[bj].bbox if own_body else None,
                )
            )
        for bj, body in enumerate(bodies):
            tid = body_track_for_idx[bj]
            if tid in linked_body_track_ids:
                continue
            bt = self._body_tracks[tid]
            persons.append(
                TrackedPerson(
                    track_id=bt.track_id,
                    bbox=bt.bbox,
                    person_id=bt.person_id,
                    similarity=bt.similarity,
                    matched=bt.matched,
                    source="body",
                    body_bbox=bt.body_bbox,
                )
            )
        return persons

    def _assign(self, bbox, assigned: set[int]) -> int | None:
        best_id, best_iou = None, self.config.track_iou_threshold
        for tid, t in self._tracks.items():
            if tid in assigned:
                continue
            score = _iou(bbox, t.bbox)
            if score >= best_iou:
                best_id, best_iou = tid, score
        return best_id

    def _spawn(self, bbox, det_score: float) -> int:
        tid = self._next_track_id
        self._next_track_id += 1
        self._tracks[tid] = Track(track_id=tid, bbox=bbox, det_score=det_score)
        return tid

    def _evict(self) -> None:
        dead = [tid for tid, t in self._tracks.items() if t.age > self.config.track_max_age]
        for tid in dead:
            del self._tracks[tid]

    def _assign_body(self, bbox, assigned: set[int]) -> int | None:
        best_id, best_iou = None, self.config.track_iou_threshold
        for tid, t in self._body_tracks.items():
            if tid in assigned:
                continue
            score = _iou(bbox, t.bbox)
            if score >= best_iou:
                best_id, best_iou = tid, score
        return best_id

    def _spawn_body(self, bbox, det_score: float) -> int:
        tid = self._next_body_track_id
        self._next_body_track_id += 1
        self._body_tracks[tid] = Track(
            track_id=tid, bbox=bbox, det_score=det_score, source="body", body_bbox=bbox
        )
        return tid

    def _evict_body(self) -> None:
        dead = [tid for tid, t in self._body_tracks.items() if t.age > self.config.track_max_age]
        for tid in dead:
            del self._body_tracks[tid]
