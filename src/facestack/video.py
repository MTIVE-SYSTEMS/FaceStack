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


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    bbox: tuple[float, float, float, float]
    person_id: str | None
    similarity: float
    matched: bool


class VideoRecognizer:
    """Stateful per-stream frame processor. One instance per camera/stream."""

    def __init__(self, recognizer: Recognizer, config: Config | None = None):
        self.rec = recognizer
        self.config = config or recognizer.config
        self._tracks: dict[int, Track] = {}
        self._next_track_id = 0

    def process_frame(self, img_bgr: np.ndarray) -> list[TrackedFace]:
        faces = self.rec.engine.embed_frame(img_bgr)

        # age all tracks; survivors get reset when matched below
        for t in self._tracks.values():
            t.age += 1
            t.frames_since_reid += 1

        assigned: set[int] = set()
        results: list[TrackedFace] = []

        for face in faces:
            tid = self._assign(face.bbox, assigned)
            if tid is None:
                tid = self._spawn(face.bbox, face.det_score)
            assigned.add(tid)
            track = self._tracks[tid]
            track.bbox = face.bbox
            track.det_score = face.det_score
            track.age = 0

            # Re-identify only on first sight or every reid_interval frames.
            if track.frames_since_reid >= self.config.reid_interval:
                m = self.rec.index.recognize(face.embedding)
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

        self._evict()
        return results

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
