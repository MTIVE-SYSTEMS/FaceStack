"""Throughput benchmark. Compares the three cost profiles, with warmup so we
measure steady-state (not MIOpen first-compile) latency.

    LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/bench.py

Tune via env: FACESTACK_DET_SIZE=320, FACESTACK_MODEL_PACK=buffalo_s, etc.
"""

from __future__ import annotations

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
