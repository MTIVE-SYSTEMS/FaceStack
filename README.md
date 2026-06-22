# FaceStack

Fast, efficient, accurate **face recognition** engine. The job is to recognise
**saved faces** (1:N identification against an enrolled gallery) — *not* to
locate faces. Detection is just the unavoidable first step when a full frame
comes in; for already-cropped faces it is skipped.

Reusable as a Python library or as a REST/WebSocket service. Targets AMD ROCm
(RX 7900 XT on `motis`) with a transparent CPU fallback.

## Architecture

```
                        ┌─────────── FaceEngine ───────────┐
full frame  ──► detect ─┤ (SCRFD)        align    ArcFace   ├─► 512-d embedding ─┐
cropped face ──► (skip detect, embed directly, align fallback)                    │
                        └──────────────────────────────────┘                     ▼
                                                              FaceIndex (hnswlib cosine)
                                                              the saved-face gallery, 1:N match
                                                                                  │
                                                                                  ▼
                                                              person_id + similarity
```

- **Models:** InsightFace `buffalo_l` — SCRFD detector + ArcFace r100 (512-d). No
  training, no fine-tuning: ArcFace already generalises at >99% on standard
  benchmarks. Calibrate `match_threshold` per deployment instead.
- **Gallery:** in-memory `hnswlib` cosine index (<10K identities, several
  embeddings per person recommended), persisted to disk.
- **Video:** IoU tracking + per-track identity caching — embedding/match runs
  once per track, refreshed every `reid_interval` frames, not every frame.
- **Runtime:** ONNX Runtime; auto-selects ROCm > MIGraphX > CUDA > CPU.

## Install (dev / CPU)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt        # ships onnxruntime (CPU)
pip install -e .
python scripts/check_env.py
```

## Deploy on motis (AMD GPU / ROCm)

`motis` has an RX 7900 XT (RDNA3, `gfx1100`). Swap the CPU runtime for the ROCm one:

```bash
pip uninstall -y onnxruntime
pip install onnxruntime-rocm        # from AMD's ROCm wheel index for your ROCm version
python scripts/check_env.py         # expect ROCMExecutionProvider in the list
```

The code picks `ROCMExecutionProvider` automatically; no code change needed.
A `docker/Dockerfile.rocm` is provided for a reproducible GPU image.

## Library usage

```python
from facestack import Recognizer
import cv2

rec = Recognizer()

# enroll saved faces
rec.enroll_frame("alice", cv2.imread("alice1.jpg"))
rec.enroll_frame("alice", cv2.imread("alice2.jpg"))

# recognise
for face in rec.recognize_frame(cv2.imread("group_photo.jpg")):
    print(face.person_id, round(face.similarity, 3), face.matched)

rec.save()   # persist the gallery
```

## Service

```bash
uvicorn facestack.service.app:app --host 0.0.0.0 --port 8000
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/healthz` | liveness, provider, gallery size |
| POST | `/enroll` | save a face (`person_id`, `file`, `cropped`) |
| POST | `/recognize` | recognise faces in an image (`file`, `cropped`) |
| GET | `/identities` | list enrolled people |
| DELETE | `/identities/{id}` | remove a person |
| POST | `/index/save` · `/index/load` | persist / restore gallery |
| WS | `/stream/recognize` | per-frame recognition for live video |

## Config

All settings are `FACESTACK_*` env vars (see `src/facestack/config.py`), e.g.
`FACESTACK_MATCH_THRESHOLD`, `FACESTACK_MODEL_PACK`, `FACESTACK_REID_INTERVAL`,
`FACESTACK_FORCE_PROVIDER`.

## Tests

```bash
pytest          # gallery logic runs without models; engine tests need insightface
```
