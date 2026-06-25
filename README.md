# FaceStack

Fast, efficient, accurate **face recognition** engine. The job is to recognise
**saved faces** (1:N identification against an enrolled gallery) — *not* to
locate faces. Detection is just the unavoidable first step when a full frame
comes in; for already-cropped faces it is skipped.

Reusable as a Python library or as a REST/WebSocket service. Targets AMD ROCm
(RX 7900 XT) with a transparent CPU fallback.

## Documentation

📖 **Docs site (developers & agents):** <https://docs.mtive.tech/>
— the full integration spec, rendered. Agents can also fetch the raw Markdown
directly: `https://raw.githubusercontent.com/MTIVE-SYSTEMS/FaceStack/main/docs/integration.md`,
or a running server's `GET /openapi.json` (Swagger at `/docs`).

| Doc | What's in it |
|---|---|
| [docs/integration.md](docs/integration.md) | **Self-contained integration spec — hand this to another agent/dev** |
| [docs/architecture.md](docs/architecture.md) | Components, pipeline, design decisions |
| [docs/usage.md](docs/usage.md) | Library API + live-video usage |
| [docs/api.md](docs/api.md) | REST + WebSocket reference with examples |
| [docs/deployment.md](docs/deployment.md) | Dev/CPU, ROCm setup, systemd, env vars |
| [docs/calibration.md](docs/calibration.md) | Threshold calibration workflow |

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

## Deploy on a GPU server (AMD GPU / ROCm) — verified

Target: RX 7900 XT (RDNA3, `gfx1100`), **ROCm 7.2.4**. After the base install above
(which puts CPU `onnxruntime` in the venv), enable the GPU:

```bash
bash scripts/setup_rocm.sh                        # ROCm-EP wheel + compat symlink
LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/check_env.py   # expect ROCMExecutionProvider
bash scripts/serve.sh                                  # serves with LD_LIBRARY_PATH set
```

Validated end-to-end on the GPU: correct match + correct rejection, GPU
utilisation ~82% under load. `on_gpu`/`GET /healthz` report the provider the
session *actually* loaded, so a silent CPU fallback is visible, not hidden.

### The AMD ROCm 7.2.4 onnxruntime trap (why the script exists)

- **MIGraphX EP is unusable.** AMD's `rocm-rel-7.2.4` MIGraphX wheel returns
  numerically wrong SCRFD output (thousands of phantom detections); the PyPI
  wheel's MIGraphX lib links ROCm 6 and won't load. `runtime.py` excludes
  MIGraphX from auto-selection.
- **No plain ROCm-EP wheel ships for 7.2.x.** The ROCm-EP wheel built for **7.0**
  (`onnxruntime_rocm-1.22.1`) is ABI-compatible with 7.2.4 — except it needs
  `librocm_smi64.so.7` while 7.2.4 ships `.so.1`. The setup script symlinks it
  (rocm_smi is device-introspection only, not the compute path).
- The PyPI `onnxruntime-rocm` (ROCm 6) silently falls back to CPU here — avoid it.

A `docker/Dockerfile.rocm` is provided as a reproducible alternative.

## Library usage

```python
from facestack import Recognizer
import cv2

rec = Recognizer()

# enroll saved faces — several varied shots per person recognise far better
rec.enroll_images("alice", [cv2.imread(p) for p in ("alice1.jpg", "alice2.jpg", "alice3.jpg")])

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
| GET | `/healthz` | liveness, provider, gallery size (no auth) |
| POST | `/v1/enroll` | save a face (`person_id`, `file`, `cropped`) |
| POST | `/v1/enroll/batch` | save several photos of one person (`person_id`, `files`, `cropped`) |
| POST | `/v1/enroll/body` · `/v1/enroll/body/batch` | permanently enrol body photos (multi-angle; body mode) |
| POST | `/v1/recognize` | recognise faces in an image (`file`, `cropped`) |
| GET | `/v1/identities` | list enrolled people |
| DELETE | `/v1/identities/{id}` | remove a person |
| POST | `/v1/index/save` · `/v1/index/load` | persist / restore gallery |
| WS | `/v1/stream/recognize` | per-frame recognition for live video |

**Auth:** set `FACESTACK_API_KEYS` (comma-separated, in `.env`); every `/v1`
request then needs a matching `X-API-Key` header. Empty ⇒ open (dev only).
**Client:** drop-in Python SDK at [`client/facestack_client.py`](client/facestack_client.py)
(`requests`-only). See [docs/api.md](docs/api.md).
**Bulk enroll:** `python scripts/enroll_dataset.py dataset` loads a
`dataset/<person>/*.jpg` tree (one folder per person) via the API, then saves.

## Performance

Measured on the GPU server (RX 7900 XT, ROCm 7.2.4, ROCMExecutionProvider), warmed up,
6-face frame:

| pack / det_size | detect-only | detect+embed all (naïve) | **video (steady)** |
|---|---|---|---|
| buffalo_l / 640 | 267 FPS | 14.5 FPS | **184 FPS** |
| buffalo_l / 320 | 457 FPS | 15.1 FPS | **267 FPS** |
| buffalo_s / 640 | 247 FPS | 18.2 FPS | **217 FPS** |
| buffalo_s / 320 | 378 FPS | 18.6 FPS | **302 FPS** |

The live-video win comes from splitting detection (every frame) from embedding
(only on first sight / every `reid_interval` frames, identity cached per track) —
~12–20× over embedding every face every frame. Even the most accurate config
(`buffalo_l` / 640) clears real-time (~30 FPS) with large margin, so it stays the
default; drop `det_size` to 320 or switch to `buffalo_s` only if you need more
headroom and can accept slightly lower accuracy on small/distant faces.

> Benchmark feeds a static frame (very stable tracks); real footage re-embeds
> more often as faces enter/move, so expect throughput between the naïve and
> steady-state columns — still well above real-time. Run `scripts/bench.py`.

## Config

All settings are `FACESTACK_*` env vars (see `src/facestack/config.py`), e.g.
`FACESTACK_MATCH_THRESHOLD`, `FACESTACK_MODEL_PACK`, `FACESTACK_REID_INTERVAL`,
`FACESTACK_FORCE_PROVIDER`.

## Threshold calibration

`match_threshold` (default `0.40`) is the cosine cutoff for "same person". The
default is a literature value; calibrate it on data that looks like your
deployment. Lay out a labelled folder and run:

```
dataset/
  alice/  a1.jpg a2.jpg ...      # variety matters: angles, distance, light,
  bob/    b1.jpg ...             # glasses on/off — not near-identical frontals
  carol/  ...

LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/calibrate.py dataset/
```

It prints the same-person vs different-person similarity distributions (the two
should be cleanly separated), operating points (best-accuracy, EER, target-FAR),
and a recommended threshold. Apply via `FACESTACK_MATCH_THRESHOLD=0.xx`.

A 2–3 person set is a real-world *sanity check*, not a precise FAR measurement
(that needs thousands of cross-person pairs, e.g. a public set like LFW in the
same folder layout). The same script handles both.

## Tests

```bash
pytest          # gallery logic runs without models; engine tests need insightface
```

## License

**FaceStack code © MTIVE SYSTEMS — licensed under [AGPL-3.0-or-later](LICENSE).**

AGPL is network copyleft: anyone who uses, modifies, or runs FaceStack as a
service for others must make their corresponding source available under the same
license. Copyright stays with MTIVE SYSTEMS, who may also grant separate
commercial licenses.

> **The face-recognition models are NOT covered by this license.** FaceStack uses
> InsightFace `buffalo_l` (SCRFD + ArcFace). The InsightFace *library* is MIT, but
> its *pretrained models* are licensed for **non-commercial research only** —
> AGPL grants no commercial rights to them. Commercial use requires a separate
> license from InsightFace (`recognition-oss-pack@insightface.ai`) or swapping in
> a commercially-licensed model. FaceStack's intended scope is therefore
> **non-commercial / research** as shipped.

Other dependencies are permissive (onnxruntime MIT, hnswlib Apache-2.0, OpenCV
Apache-2.0, FastAPI MIT, NumPy BSD).
