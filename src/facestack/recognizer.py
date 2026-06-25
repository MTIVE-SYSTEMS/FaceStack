"""Recognizer — the headline API: enroll saved faces, then recognise them.

Composes a FaceEngine (pixels -> embeddings) with a FaceIndex (the saved-face
gallery + 1:N matcher). This is what other projects import / the service wraps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from .config import Config
from .engine import DetectedFace, FaceEngine
from .index import FaceIndex, Match
from .linking import link_faces_to_bodies  # pure-geometry; no heavy deps

log = logging.getLogger("facestack.recognizer")

# Skip auto-enrolling a body embedding when the same person already has a
# near-identical one live in the gallery: it adds no recognition signal and only
# inflates the index. High enough that genuinely different poses/angles/cameras
# still enrol (those DO help cross-camera ReID).
_ENROLL_DEDUP_SIM = 0.92


@dataclass(slots=True)
class RecognizedFace:
    bbox: tuple[float, float, float, float]
    det_score: float
    person_id: str | None  # None when no enrolled face is close enough
    similarity: float
    matched: bool


@dataclass(slots=True)
class RecognizedBody:
    bbox: tuple[float, float, float, float]
    det_score: float
    similarity: float
    matched: bool


@dataclass(slots=True)
class PersonResult:
    """A unified identity from a scene: either face-driven or body-driven."""

    person_id: str | None
    matched: bool
    similarity: float
    source: str  # 'face' | 'body'
    face: RecognizedFace | None = None
    body: RecognizedBody | None = None


class Recognizer:
    def __init__(self, config: Config | None = None, index: FaceIndex | None = None):
        self.config = config or Config()
        self.engine = FaceEngine(self.config)
        self.index = index or FaceIndex(
            dim=self.config.embedding_dim,
            capacity=self.config.index_capacity,
            threshold=self.config.match_threshold,
        )

        # Body recognition is fully opt-in. With it OFF (the default) we never
        # construct BodyEngine/BodyIndex — so the package needs no body models
        # and existing behaviour is byte-for-byte unchanged. Heavy/body imports
        # stay local to this branch.
        self._body_enabled = self.config.enable_body
        self.body_engine = None
        self.body_index = None
        if self._body_enabled:
            from .bodyengine import BodyEngine
            from .bodyindex import BodyIndex

            self.body_engine = BodyEngine(self.config)
            self.body_index = BodyIndex(
                dim=self.config.body_embedding_dim,
                capacity=self.config.index_capacity,
                threshold=self.config.body_match_threshold,
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

    def enroll_images(
        self, person_id: str, images: list[np.ndarray], cropped: bool = False
    ) -> list[int]:
        """Enroll several photos of one person at once (varied angles/lighting).

        A few embeddings per person make recognition far more robust than a single
        shot. Returns the per-image face count (0 where no face was usable), so the
        caller can report which photos contributed.
        """
        counts: list[int] = []
        for img in images:
            if cropped:
                counts.append(1 if self.enroll_crop(person_id, img) else 0)
            else:
                counts.append(self.enroll_frame(person_id, img))
        return counts

    # --- body enrollment (permanent, mirrors face enrollment) ---
    def _require_body(self) -> None:
        if not self._body_enabled or self.body_engine is None or self.body_index is None:
            raise RuntimeError("body recognition is disabled (set FACESTACK_ENABLE_BODY=1)")

    def enroll_body_frame(self, person_id: str, img_bgr: np.ndarray) -> int:
        """Permanently enroll every body found in a full image. Returns count.

        Unlike auto-enrolled (day-scoped) bodies, these never expire — enroll a
        few angles (front/side/back) per person, just like multi-shot faces.
        """
        self._require_body()
        import time

        now = time.time()
        bodies = self.body_engine.detect_and_embed(img_bgr)
        for b in bodies:
            self.body_index.add(person_id, b.embedding, ts=now, permanent=True)
        return len(bodies)

    def enroll_body_crop(self, person_id: str, img_bgr: np.ndarray) -> bool:
        """Permanently enroll one already-cropped body. False if it can't embed."""
        self._require_body()
        import time

        if img_bgr is None or img_bgr.size == 0:
            return False
        h, w = img_bgr.shape[:2]
        emb = self.body_engine.embed_body(img_bgr, (0.0, 0.0, float(w), float(h)))
        self.body_index.add(person_id, emb, ts=time.time(), permanent=True)
        return True

    def enroll_body_images(
        self, person_id: str, images: list[np.ndarray], cropped: bool = False
    ) -> list[int]:
        """Permanently enroll several body photos at once. Per-image body count."""
        self._require_body()
        counts: list[int] = []
        for img in images:
            if cropped:
                counts.append(1 if self.enroll_body_crop(person_id, img) else 0)
            else:
                counts.append(self.enroll_body_frame(person_id, img))
        return counts

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

    # --- unified scene recognition (faces + bodies) ---
    def recognize_scene(self, img_bgr: np.ndarray, now: float | None = None) -> list[PersonResult]:
        """Recognise a whole scene: faces, and (when enabled) bodies linked to them.

        When a body's linked face is a CONFIDENT match its embedding is auto-
        enrolled into the body gallery under that person_id (the only way body
        embeddings are created — there is no manual body enroll). Bodies with no
        visible/matched face are recognised against the body gallery. `now` is
        injectable for tests/persistence parity and flows to BodyIndex.
        """
        if now is None:
            import time

            now = time.time()

        faces = self.recognize_frame(img_bgr)

        # Body-disabled fast path: emit face-only PersonResults, no body model
        # needed. Keeps recognize_scene callable even without body support.
        if not self._body_enabled:
            return [
                PersonResult(
                    person_id=f.person_id,
                    matched=f.matched,
                    similarity=f.similarity,
                    source="face",
                    face=f,
                    body=None,
                )
                for f in faces
            ]

        # Bound gallery growth: drop everything past its TTL before we enrol more.
        # The stills/HTTP path has no per-track throttle (unlike video), so this
        # purge plus the near-duplicate skip below are what keep it from growing
        # without bound under repeated requests for the same person.
        self.body_index.purge(now, self.config.body_ttl_seconds)

        bodies = self.body_engine.detect_and_embed(img_bgr)

        # Link each face to its best body, then build a body->face reverse map
        # (first face to claim a body wins; faces are usually 1:1 with bodies).
        face_boxes = [f.bbox for f in faces]
        body_boxes = [b.bbox for b in bodies]
        links = link_faces_to_bodies(face_boxes, body_boxes, self.config.body_link_containment)
        body_to_face: dict[int, int] = {}
        for i, j in enumerate(links):
            if j is not None and j not in body_to_face:
                body_to_face[j] = i

        # Auto-enroll body embeddings ONLY for bodies whose linked face is a
        # confident match (f.matched True == similarity >= threshold + person_id).
        for j, body in enumerate(bodies):
            if j in body_to_face:
                f = faces[body_to_face[j]]
                if f.matched and f.person_id is not None:
                    if not self._already_enrolled(f.person_id, body.embedding, now):
                        self.body_index.add(f.person_id, body.embedding, ts=now)

        results: list[PersonResult] = []

        # Faces first, in detection order; attach the linked body region (its
        # identity is the FACE-derived one propagated to the body box).
        for i, f in enumerate(faces):
            j = links[i]
            body_result = None
            # Only attach the body to the face that actually owns it: if two faces
            # link to the same body, body_to_face picked one winner.
            if j is not None and body_to_face.get(j) == i:
                b = bodies[j]
                body_result = RecognizedBody(
                    bbox=b.bbox,
                    det_score=b.det_score,
                    similarity=f.similarity,
                    matched=f.matched,
                )
            results.append(
                PersonResult(
                    person_id=f.person_id,
                    matched=f.matched,
                    similarity=f.similarity,
                    source="face",
                    face=f,
                    body=body_result,
                )
            )

        # Bodies with no linked face: recognise via the body gallery.
        for j, body in enumerate(bodies):
            if j in body_to_face:
                continue
            m = self.body_index.recognize(
                body.embedding, now=now, ttl=self.config.body_ttl_seconds
            )
            person_id = m.person_id if (m and m.matched) else None
            similarity = m.similarity if m else 0.0
            matched = bool(m and m.matched)
            results.append(
                PersonResult(
                    person_id=person_id,
                    matched=matched,
                    similarity=similarity,
                    source="body",
                    face=None,
                    body=RecognizedBody(
                        bbox=body.bbox,
                        det_score=body.det_score,
                        similarity=similarity,
                        matched=matched,
                    ),
                )
            )

        return results

    def _already_enrolled(self, person_id: str, embedding: np.ndarray, now: float) -> bool:
        """True if person_id already has a live near-duplicate of this embedding."""
        m = self.body_index.recognize(embedding, now=now, ttl=self.config.body_ttl_seconds)
        return bool(m and m.person_id == person_id and m.similarity >= _ENROLL_DEDUP_SIM)

    # --- persistence passthrough ---
    def save(self) -> None:
        self.index.save(self.config.index_path, self.config.meta_path)
        if self._body_enabled and self.body_index is not None:
            ip = os.path.expanduser(self.config.body_index_path)
            mp = os.path.expanduser(self.config.body_meta_path)
            self.body_index.save(ip, mp)

    def load(self) -> None:
        self.index = FaceIndex.load(self.config.index_path, self.config.meta_path)
        if self._body_enabled:
            from .bodyindex import BodyIndex

            ip = os.path.expanduser(self.config.body_index_path)
            mp = os.path.expanduser(self.config.body_meta_path)
            if os.path.exists(ip) and os.path.exists(mp):
                self.body_index = BodyIndex.load(ip, mp)
            # else: keep the empty body_index from __init__ (nothing saved yet)
