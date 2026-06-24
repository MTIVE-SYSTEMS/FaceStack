"""BodyEngine — turns pixels into person boxes and OSNet appearance embeddings.

The body half of FaceStack: a YOLOv8 person detector and an OSNet ReID model,
both run as raw ``onnxruntime.InferenceSession`` (NO torch / ultralytics /
torchreid at runtime), mirroring how ``FaceEngine`` drives insightface.

Three entry points reflect how callers compose detection and embedding:
  * detect(img)            — locate people only (cheap, every frame in video).
  * embed_body(img, bbox)  — OSNet embedding for one already-located body.
  * detect_and_embed(img)  — convenience: detect, then embed each, for stills.

Everything heavy (onnxruntime, cv2) is imported lazily inside ``__init__`` /
methods so the package still imports and existing tests pass when body models
and ORT are absent and ``config.enable_body`` is False (the default).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .runtime import select_providers, using_gpu

log = logging.getLogger("facestack.bodyengine")


@dataclass(slots=True)
class DetectedBody:
    """One person found in an image, with their appearance embedding."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in original pixels
    det_score: float  # YOLOv8 person-class confidence
    embedding: np.ndarray  # L2-normalized 512-d OSNet vector


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ImageNet stats, RGB order, shaped (3,1,1) to broadcast over a CHW array.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# --------------------------------------------------------------------------- #
# YOLOv8 numpy helpers — letterbox + NMS + decode. Module-level, pure numpy
# (cv2 imported lazily inside letterbox), so they are trivially unit-testable.
# --------------------------------------------------------------------------- #
def letterbox(
    img_bgr: np.ndarray,
    new_shape: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio, pad to a square. Returns (img, ratio, pad_w, pad_h).

    A single ratio (no separate w/h scale) keeps the geometry invertible and
    exact — ratio/pad are exactly what ``decode_persons`` needs to map boxes in
    640-letterboxed space back to original pixels.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    ratio = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w = (new_shape - nw) / 2.0  # left/right pad (float; halves)
    pad_h = (new_shape - nh) / 2.0  # top/bottom pad
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return out, ratio, float(left), float(top)


def preprocess_yolo(
    img_bgr: np.ndarray, size: int = 640
) -> tuple[np.ndarray, float, float, float]:
    """BGR image -> (blob (1,3,size,size), ratio, pad_w, pad_h) for YOLOv8."""
    lb, ratio, pad_w, pad_h = letterbox(img_bgr, size)
    rgb = lb[:, :, ::-1]  # BGR -> RGB
    blob = rgb.astype(np.float32) / 255.0
    blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])  # (1,3,H,W)
    return blob, ratio, pad_w, pad_h


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list[int]:
    """Greedy NMS. boxes: (N,4) x1y1x2y2. Returns kept indices, score-desc."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode_persons(
    output: np.ndarray,
    ratio: float,
    pad_w: float,
    pad_h: float,
    orig_w: int,
    orig_h: int,
    conf_thr: float = 0.5,
    iou_thr: float = 0.45,
    person_class: int = 0,
) -> list[tuple[tuple[float, float, float, float], float]]:
    """YOLOv8 raw (1,84,8400) output -> [((x1,y1,x2,y2), score), ...] for `person`.

    84 = 4 box (cx,cy,w,h in 640 letterbox space) + 80 class scores; YOLOv8 has
    NO separate objectness, the class score *is* the confidence.

    Exports vary in layout: the canonical Ultralytics ONNX is (1,84,8400) but
    transformers.js / some re-exports ship (1,8400,84). We orient by the axes
    (attrs ~84 << anchors ~8400) instead of assuming, so a transposed export
    produces correct boxes rather than silently reading garbage.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"unexpected YOLOv8 output shape {np.asarray(output).shape}")
    # Put anchors on axis 0, attributes on axis 1: attrs is the smaller axis.
    pred = arr.T if arr.shape[0] < arr.shape[1] else arr
    nattr = pred.shape[1]
    if nattr < 5 + person_class:
        raise ValueError(
            f"YOLOv8 output has {nattr} attrs; need >= {5 + person_class} "
            f"(4 box + class {person_class}). Wrong model export?"
        )
    boxes_cxcywh = pred[:, :4]
    scores = pred[:, 4 + person_class]  # class-0 column == person confidence

    m = scores >= conf_thr
    if not np.any(m):
        return []
    b = boxes_cxcywh[m]
    s = scores[m]

    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xyxy = np.empty_like(b)
    xyxy[:, 0] = cx - bw / 2
    xyxy[:, 1] = cy - bh / 2  # in 640 letterbox space
    xyxy[:, 2] = cx + bw / 2
    xyxy[:, 3] = cy + bh / 2

    keep = nms(xyxy, s, iou_thr)
    out: list[tuple[tuple[float, float, float, float], float]] = []
    for i in keep:
        x1 = (xyxy[i, 0] - pad_w) / ratio  # undo pad, then undo scale
        y1 = (xyxy[i, 1] - pad_h) / ratio
        x2 = (xyxy[i, 2] - pad_w) / ratio
        y2 = (xyxy[i, 3] - pad_h) / ratio
        x1 = float(np.clip(x1, 0, orig_w - 1))
        y1 = float(np.clip(y1, 0, orig_h - 1))
        x2 = float(np.clip(x2, 0, orig_w - 1))
        y2 = float(np.clip(y2, 0, orig_h - 1))
        out.append(((x1, y1, x2, y2), float(s[i])))
    return out


# --------------------------------------------------------------------------- #
# OSNet numpy preprocessing — plain resize (NO letterbox), ImageNet-normalized.
# --------------------------------------------------------------------------- #
def preprocess_osnet(
    img_bgr: np.ndarray, size_hw: tuple[int, int] = (256, 128)
) -> np.ndarray:
    """BGR crop -> normalized NCHW float32 blob for OSNet (1,3,H,W).

    OSNet uses a plain (aspect-distorting) resize — that is how it was trained,
    unlike YOLO's letterbox. ``size_hw`` is (H, W) but cv2 wants (W, H): that
    swap is the single most common bug in this path.
    """
    import cv2

    h, w = size_hw
    resized = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0  # HWC->CHW, [0,1]
    chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD  # ImageNet normalize
    return np.ascontiguousarray(chw[np.newaxis, ...])  # (1,3,H,W)


class BodyEngine:
    """YOLOv8 person detector + OSNet ReID, both raw onnxruntime sessions.

    Both ONNX sessions are created in ``__init__`` from the configured paths. A
    missing model raises a clear RuntimeError naming the fetch script — the
    package itself stays importable because every heavy import is local.
    """

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.providers = select_providers(self.config.force_provider)

        # Fail early and helpfully if either model is missing.
        det_path = Path(self.config.body_detector_path).expanduser()
        reid_path = Path(self.config.body_reid_path).expanduser()
        if not det_path.is_file():
            raise RuntimeError(
                f"body detector ONNX not found at {det_path}; "
                "run scripts/fetch_body_models.py"
            )
        if not reid_path.is_file():
            raise RuntimeError(
                f"body ReID ONNX not found at {reid_path}; "
                "run scripts/fetch_body_models.py"
            )

        # Imported lazily so the rest of the package is usable without ORT.
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.det_session = ort.InferenceSession(
            str(det_path),
            sess_options=so,
            providers=self.providers,  # requested; CPU already last
        )
        self.reid_session = ort.InferenceSession(
            str(reid_path),
            sess_options=so,
            providers=self.providers,
        )

        # Read I/O names off the graph rather than hardcoding "images"/"output0";
        # exports vary (images/input/input.1, output0/features).
        self._det_input = self.det_session.get_inputs()[0].name
        self._det_output = self.det_session.get_outputs()[0].name
        self._reid_input = self.reid_session.get_inputs()[0].name
        self._reid_output = self.reid_session.get_outputs()[0].name

        # Some public OSNet exports fix the batch dimension (e.g. 16) instead of
        # leaving it dynamic. We feed one crop at a time, so detect a fixed batch
        # here and tile the single crop up to it in embed_body, slicing row 0
        # back out. A dynamic dim (str/None) or 1 means no tiling.
        reid_batch = self.reid_session.get_inputs()[0].shape[0]
        self._reid_batch = reid_batch if isinstance(reid_batch, int) and reid_batch > 1 else 1

        # Truth, not intent: a requested GPU provider can silently fail to load
        # (missing/ABI-mismatched ROCm libs) and ORT falls back to CPU per-node.
        # Read back what the sessions actually applied. Both use the same
        # provider list, so the detector's readback represents the engine.
        self.active_providers = list(self.det_session.get_providers())
        if not using_gpu(self.active_providers) and using_gpu(self.providers):
            log.warning(
                "Requested GPU (%s) but session applied %s — running on CPU. "
                "Check ROCm libs / LD_LIBRARY_PATH.",
                self.providers,
                self.active_providers,
            )
        log.info(
            "BodyEngine ready: detector=%s reid=%s requested=%s active=%s gpu=%s",
            det_path.name,
            reid_path.name,
            self.providers,
            self.active_providers,
            using_gpu(self.active_providers),
        )

    @property
    def on_gpu(self) -> bool:
        """True only if a GPU provider was actually loaded by the session."""
        return using_gpu(self.active_providers)

    # --- detection only (no embedding) — the per-frame cost in video ---
    def detect(
        self, img_bgr: np.ndarray
    ) -> list[tuple[tuple[float, float, float, float], float]]:
        """Locate people WITHOUT embedding them. Returns (bbox, det_score).

        Splitting detection from embedding mirrors FaceEngine: the detector runs
        every frame, the heavier OSNet embedder only when a track needs ReID.
        """
        h, w = img_bgr.shape[:2]
        blob, ratio, pad_w, pad_h = preprocess_yolo(img_bgr, self.config.body_det_size)
        out = self.det_session.run([self._det_output], {self._det_input: blob})[0]
        dets = decode_persons(
            np.asarray(out),
            ratio,
            pad_w,
            pad_h,
            w,
            h,
            self.config.body_det_thresh,
            self.config.body_nms_iou,
        )
        # Sanity guard: the MIGraphX phantom-detection failure (the SCRFD bug) is
        # numerical, not a crash — implausibly many survivors on MIGraphX means
        # garbage output, not a crowd.
        if len(dets) > 300 and any("MIGraphX" in p for p in self.active_providers):
            log.warning(
                "Implausible person count (%d) on MIGraphX — likely the "
                "numerically-wrong-output bug; prefer ROCMExecutionProvider.",
                len(dets),
            )
        return dets

    # --- embed a single already-located body via OSNet ---
    def embed_body(
        self, img_bgr: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> np.ndarray:
        """L2-normalized 512-d OSNet appearance embedding for one body crop."""
        x1, y1, x2, y2 = (int(round(c)) for c in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1], x2), min(img_bgr.shape[0], y2)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            raise ValueError(f"empty body crop for bbox {bbox}")
        blob = preprocess_osnet(crop, self.config.body_reid_size)
        if self._reid_batch > 1:  # fixed-batch export: tile, embed, keep row 0
            blob = np.tile(blob, (self._reid_batch, 1, 1, 1))
        feat = self.reid_session.run([self._reid_output], {self._reid_input: blob})[0]
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 2:  # (batch, 512) -> first row is our single crop
            feat = feat[0]
        return _l2norm(feat.reshape(-1))

    # --- full frame: locate then embed (stills / convenience) ---
    def detect_and_embed(self, img_bgr: np.ndarray) -> list[DetectedBody]:
        """Detect every person and embed each — for single still images."""
        out: list[DetectedBody] = []
        for bbox, score in self.detect(img_bgr):
            try:
                emb = self.embed_body(img_bgr, bbox)
            except ValueError:
                continue  # degenerate crop at the frame edge — skip it
            out.append(DetectedBody(bbox=bbox, det_score=score, embedding=emb))
        return out
