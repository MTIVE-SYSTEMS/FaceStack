"""Recognizer — the headline API: enroll saved faces, then recognise them.

Composes a FaceEngine (pixels -> embeddings) with a FaceIndex (the saved-face
gallery + 1:N matcher). This is what other projects import / the service wraps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .engine import DetectedFace, FaceEngine
from .index import FaceIndex, Match

log = logging.getLogger("facestack.recognizer")


@dataclass(slots=True)
class RecognizedFace:
    bbox: tuple[float, float, float, float]
    det_score: float
    person_id: str | None  # None when no enrolled face is close enough
    similarity: float
    matched: bool


class Recognizer:
    def __init__(self, config: Config | None = None, index: FaceIndex | None = None):
        self.config = config or Config()
        self.engine = FaceEngine(self.config)
        self.index = index or FaceIndex(
            dim=self.config.embedding_dim,
            capacity=self.config.index_capacity,
            threshold=self.config.match_threshold,
        )

    # --- enrollment (saving faces) ---
    def enroll_frame(self, person_id: str, img_bgr: np.ndarray) -> int:
        """Enroll every face found in a full image under person_id. Returns count."""
        faces = self.engine.embed_frame(img_bgr)
        for f in faces:
            self.index.add(person_id, f.embedding)
        return len(faces)

    def enroll_crop(self, person_id: str, img_bgr: np.ndarray) -> bool:
        """Enroll a single cropped face. Returns False if none could be embedded."""
        face = self.engine.embed_crop(img_bgr)
        if face is None:
            return False
        self.index.add(person_id, face.embedding)
        return True

    # --- recognition ---
    def recognize_frame(self, img_bgr: np.ndarray) -> list[RecognizedFace]:
        """Locate every face in a frame and match each against saved faces."""
        return [self._match(f) for f in self.engine.embed_frame(img_bgr)]

    def recognize_crop(self, img_bgr: np.ndarray) -> RecognizedFace | None:
        """Match a single cropped face against saved faces."""
        face = self.engine.embed_crop(img_bgr)
        return self._match(face) if face is not None else None

    def _match(self, face: DetectedFace) -> RecognizedFace:
        m: Match | None = self.index.recognize(face.embedding)
        return RecognizedFace(
            bbox=face.bbox,
            det_score=face.det_score,
            person_id=m.person_id if (m and m.matched) else None,
            similarity=m.similarity if m else 0.0,
            matched=bool(m and m.matched),
        )

    # --- persistence passthrough ---
    def save(self) -> None:
        self.index.save(self.config.index_path, self.config.meta_path)

    def load(self) -> None:
        self.index = FaceIndex.load(self.config.index_path, self.config.meta_path)
