"""FaceEngine — turns pixels into ArcFace embeddings.

Two entry points reflect the two input kinds:
  * embed_frame(img) — full scene/video frame: locate faces, then embed each one.
  * embed_crop(img)  — an already-cropped face: embed directly (detection is
    still attempted for proper alignment, with a resize-and-embed fallback).

Detection here is *plumbing* for recognition, not the product. The headline
job — matching against saved faces — lives in FaceIndex / Recognizer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import Config
from .runtime import ctx_id_for, select_providers, using_gpu

log = logging.getLogger("facestack.engine")


@dataclass(slots=True)
class DetectedFace:
    """One face found (or assumed) in an image, with its embedding."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    det_score: float  # detector confidence (0.0 when detection was skipped)
    embedding: np.ndarray  # L2-normalized 512-d ArcFace vector
    kps: np.ndarray | None = None  # 5x2 landmarks, when available


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class FaceEngine:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.providers = select_providers(self.config.force_provider)

        # Imported lazily so the rest of the package (index, schemas) is usable
        # without the heavy insightface/onnxruntime stack installed.
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(name=self.config.model_pack, providers=self.providers)
        self.app.prepare(
            ctx_id=ctx_id_for(self.providers),
            det_thresh=self.config.det_thresh,
            det_size=(self.config.det_size, self.config.det_size),
        )
        # ArcFace recognition model, reused directly for the cropped-face fallback.
        self.rec = self.app.models["recognition"]

        # Truth, not intent: a requested GPU provider can silently fail to load
        # (missing/ABI-mismatched ROCm libs) and ONNX Runtime falls back to CPU.
        # Read back what the real session actually applied.
        self.active_providers = list(self.app.models["detection"].session.get_providers())
        if not using_gpu(self.active_providers) and using_gpu(self.providers):
            log.warning(
                "Requested GPU (%s) but session applied %s — running on CPU. "
                "Check ROCm libs / LD_LIBRARY_PATH.",
                self.providers,
                self.active_providers,
            )
        log.info(
            "FaceEngine ready: pack=%s requested=%s active=%s gpu=%s",
            self.config.model_pack,
            self.providers,
            self.active_providers,
            using_gpu(self.active_providers),
        )

    @property
    def on_gpu(self) -> bool:
        """True only if a GPU provider was actually loaded by the session."""
        return using_gpu(self.active_providers)

    # --- full frame: locate then embed ---
    def embed_frame(self, img_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect, align and embed every face in a full scene/video frame."""
        faces = self.app.get(img_bgr)
        return [
            DetectedFace(
                bbox=tuple(float(x) for x in f.bbox),
                det_score=float(f.det_score),
                embedding=_l2norm(np.asarray(f.normed_embedding, dtype=np.float32)),
                kps=np.asarray(f.kps) if f.kps is not None else None,
            )
            for f in faces
        ]

    # --- cropped face: embed directly ---
    def embed_crop(self, img_bgr: np.ndarray) -> DetectedFace | None:
        """Embed an already-cropped face.

        Detection is still attempted (proper 5-point alignment is what makes
        ArcFace accurate). If the crop is too tight for the detector, we fall
        back to a plain 112x112 resize — lower accuracy, but never fails.
        """
        faces = self.app.get(img_bgr)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return DetectedFace(
                bbox=tuple(float(x) for x in f.bbox),
                det_score=float(f.det_score),
                embedding=_l2norm(np.asarray(f.normed_embedding, dtype=np.float32)),
                kps=np.asarray(f.kps) if f.kps is not None else None,
            )

        emb = self._embed_unaligned(img_bgr)
        if emb is None:
            return None
        h, w = img_bgr.shape[:2]
        return DetectedFace(bbox=(0.0, 0.0, float(w), float(h)), det_score=0.0, embedding=emb)

    def _embed_unaligned(self, img_bgr: np.ndarray) -> np.ndarray | None:
        import cv2

        if img_bgr is None or img_bgr.size == 0:
            return None
        blob = cv2.resize(img_bgr, (112, 112))
        feat = self.rec.get_feat(blob)  # (1, 512), not normalized
        return _l2norm(np.asarray(feat, dtype=np.float32).flatten())
