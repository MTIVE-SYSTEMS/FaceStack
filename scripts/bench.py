"""Throughput benchmark. Compares the per-stage cost profiles, with warmup so we
measure steady-state (not MIOpen first-compile) latency.

    LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/bench.py

Tune via env: FACESTACK_DET_SIZE=320, FACESTACK_MODEL_PACK=buffalo_s, etc.
With FACESTACK_ENABLE_BODY=1 it also benchmarks the body (person ReID) path:
detector, detector+ReID, the unified recognize_scene, and the video persons path.
"""

from __future__ import annotations

import glob
import time

from insightface.data import get_image as get_image

from facestack import Config, Recognizer
from facestack.video import VideoRecognizer


def bench(fn, n: int = 100, warmup: int = 15) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    t = time.time()
    for _ in range(n):
        fn()
    dt = time.time() - t
    return n / dt, dt / n * 1000.0


def _load_person_image(max_side: int = 1280):
    """A realistic full-person frame for the body path: prefer a dataset photo,
    fall back to the bundled insightface image. Resized so the longer side is
    `max_side` (representative of a live stream, not a 12 MP phone photo)."""
    import cv2

    img = None
    for p in sorted(glob.glob("dataset/*/*.jpg")) + sorted(glob.glob("dataset/*/*.png")):
        img = cv2.imread(p)
        if img is not None:
            break
    if img is None:
        img = get_image("t1")
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))))
    return img


def main() -> int:
    cfg = Config()
    rec = Recognizer(cfg)
    print(
        f"pack={cfg.model_pack} det_size={cfg.det_size} reid_interval={cfg.reid_interval}\n"
        f"active_providers={rec.engine.active_providers} gpu={rec.engine.on_gpu}\n"
    )

    img = get_image("t1")
    faces = rec.engine.embed_frame(img)
    rec.index.add("p0", faces[0].embedding)
    n_faces = len(faces)

    fps, ms = bench(lambda: rec.engine.detect(img))
    print(f"detect only        : {fps:7.1f} FPS  {ms:6.2f} ms/frame")

    fps, ms = bench(lambda: rec.engine.embed_frame(img))
    print(f"detect + embed all : {fps:7.1f} FPS  {ms:6.2f} ms/frame  ({n_faces} faces/frame)")

    vr = VideoRecognizer(rec, cfg)
    fps, ms = bench(lambda: vr.process_frame(img))
    print(f"video (steady)     : {fps:7.1f} FPS  {ms:6.2f} ms/frame  (identity cached per track)")

    # --- body (person ReID) path, only when enabled ---
    if cfg.enable_body and rec.body_engine is not None:
        bimg = _load_person_image()
        bh, bw = bimg.shape[:2]
        bodies = rec.body_engine.detect_and_embed(bimg)
        n_bodies = len(bodies)
        # Seed the body gallery so recognize_scene exercises a real search.
        for i, b in enumerate(bodies):
            rec.body_index.add(f"b{i}", b.embedding, ts=time.time())
        print(
            f"\nbody: reid_batch={rec.body_engine._reid_batch} gpu={rec.body_engine.on_gpu} "
            f"frame={bw}x{bh} bodies/frame={n_bodies}\n"
        )

        fps, ms = bench(lambda: rec.body_engine.detect(bimg))
        print(f"body detect only   : {fps:7.1f} FPS  {ms:6.2f} ms/frame")

        fps, ms = bench(lambda: rec.body_engine.detect_and_embed(bimg))
        print(f"body detect + ReID : {fps:7.1f} FPS  {ms:6.2f} ms/frame  ({n_bodies} bodies/frame)")

        fps, ms = bench(lambda: rec.recognize_scene(bimg))
        print(f"recognize_scene    : {fps:7.1f} FPS  {ms:6.2f} ms/frame  (faces+bodies+link+match)")

        vrb = VideoRecognizer(rec, cfg)
        fps, ms = bench(lambda: vrb.process_frame_persons(bimg))
        print(f"video persons      : {fps:7.1f} FPS  {ms:6.2f} ms/frame  (face+body tracks, cached)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
