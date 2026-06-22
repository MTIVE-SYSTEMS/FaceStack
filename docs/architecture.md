# Architecture

FaceStack recognises **saved faces** — given a face, it answers *who is this?*
against an enrolled gallery (1:N identification). Detection (locating a face in
a frame) is a supporting step, not the product.

```
full frame  ──► detect ──► align ──► ArcFace ──► 512-d embedding ─┐
cropped face ─► (detect skipped; embed directly, align fallback)   │
                                                                   ▼
                                              FaceIndex (hnswlib cosine gallery)
                                              nearest enrolled embedding, 1:N
                                                                   │
                                                  similarity ≥ threshold ?
                                                   ├─ yes → person_id
                                                   └─ no  → unknown
```

## Components

| Module | Responsibility |
|---|---|
| `facestack.engine.FaceEngine` | Pixels → embeddings. Wraps InsightFace `buffalo_l` (SCRFD detector + ArcFace r100). Selects the ONNX Runtime provider. |
| `facestack.index.FaceIndex` | The saved-face gallery. In-memory hnswlib cosine index, 1:N search, persistence. **The heart of the product.** |
| `facestack.recognizer.Recognizer` | Facade composing engine + index: `enroll_*` / `recognize_*`. |
| `facestack.video.VideoRecognizer` | Stateful per-stream processor. IoU tracking + per-track identity caching for live video. |
| `facestack.runtime` | ONNX Runtime provider selection (ROCm > CUDA > CPU; MIGraphX excluded). |
| `facestack.service.app` | FastAPI REST + WebSocket service. |

## Key design decisions

- **No training, no fine-tuning.** ArcFace (`buffalo_l`) already generalises at
  >99% on standard benchmarks. We calibrate the decision *threshold* per
  deployment instead of touching the model. See [calibration.md](calibration.md).
- **Embeddings, not classification.** Adding/removing a person is just adding/
  removing a vector — no retraining. Scales to the <10K-identity target with a
  plain in-memory index.
- **Detection split from embedding for video.** `FaceEngine.detect()` runs every
  frame (cheap); the heavier ArcFace embedding runs only when a track needs
  (re)identification. Identity is cached per track. This is the ~12–20× live-video
  speed-up — see [../README.md#performance](../README.md).
- **Provider honesty.** A requested GPU provider can silently fail to load and
  fall back to CPU. `FaceEngine.active_providers` / `on_gpu` report what the
  session *actually* loaded, surfaced in `GET /healthz`.

## Two input paths

- **Full frame** (`embed_frame` / `recognize_frame`): detect + align + embed every
  face. Use for scenes, group photos, video frames.
- **Cropped face** (`embed_crop` / `recognize_crop`): detection is attempted for
  proper alignment; if the crop is too tight, it falls back to a plain 112×112
  resize. Use when an upstream system already cropped the face.
