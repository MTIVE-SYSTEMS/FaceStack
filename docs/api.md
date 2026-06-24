# Service API

Start the service:

```bash
bash scripts/serve.sh                 # uses LD_LIBRARY_PATH + FACESTACK_* env
# or: uvicorn facestack.service.app:app --host 0.0.0.0 --port 8000
```

On the GPU server it runs as a systemd service on port **8011** (see [deployment.md](deployment.md)).
Interactive docs: `/docs` (Swagger UI). All functional endpoints are under **`/v1`**.

## Authentication

Set one or more keys via `FACESTACK_API_KEYS` (comma-separated, keep them in
`.env`). Every `/v1` request must then send a matching `X-API-Key` header:

```bash
export FACESTACK_API_KEYS="proj-a-key,proj-b-key"   # in ~/FaceStack/.env on the server
curl -H "X-API-Key: proj-a-key" ...
```

- No keys configured ⇒ auth is **disabled** (open) — only for local dev.
- Missing/invalid key ⇒ `401`. `/healthz` is always open (liveness probes).

## Endpoints

### `GET /healthz`  (no auth)
```json
{ "status": "ok", "providers": ["ROCMExecutionProvider","CPUExecutionProvider"],
  "on_gpu": true, "gallery_size": 12, "people": 3,
  "body_enabled": false, "body_on_gpu": false, "body_gallery_size": 0 }
```
`body_*` fields reflect the optional body-recognition extension (see below).

### `POST /v1/enroll`
`multipart/form-data`: `person_id` (str), `file` (image), `cropped` (bool,
default false — true if `file` is already a cropped face). `422` if no face found.
```bash
curl -H "X-API-Key: $KEY" -F person_id=ahmet -F file=@ahmet.jpg http://<host>:8011/v1/enroll
# {"person_id":"ahmet","enrolled":1}
```

### `POST /v1/enroll/batch`
Enroll several photos of one person at once — a few varied angles/lighting
recognise far more reliably than a single shot. Form: `person_id` (str),
`files` (repeated), `cropped` (bool). `422` if no face is found in *any* image.
```bash
curl -H "X-API-Key: $KEY" -F person_id=ahmet \
     -F files=@a1.jpg -F files=@a2.jpg -F files=@a3.jpg http://<host>:8011/v1/enroll/batch
# {"person_id":"ahmet","images":3,"enrolled":4,"per_image":[2,1,1]}
```
`per_image[i]` is the face count from image `i` (0 = no usable face there).
Bulk-load a `dataset/<name>/*.jpg` tree: `python scripts/enroll_dataset.py dataset
--base-url http://<host>:8011 --api-key "$KEY"` (enrolls each folder, then saves).

### `POST /v1/recognize`
Form: `file` (image), `cropped` (bool, default false).
```bash
curl -H "X-API-Key: $KEY" -F file=@group.jpg http://<host>:8011/v1/recognize
```
```json
{ "faces": [
  { "bbox":[x1,y1,x2,y2], "det_score":0.92, "person_id":"ahmet", "similarity":0.71, "matched":true },
  { "bbox":[...], "det_score":0.88, "person_id":null, "similarity":0.13, "matched":false }
] }
```
`person_id` is `null` / `matched` false when nothing clears the threshold.

With body recognition enabled, the response also carries `persons` (unified
face+body identities) and `bodies` — see [Body recognition](#body-recognition-person-reid).

### `GET /v1/identities`
`{ "count": 3, "people": ["ahmet","aras","yigithan"] }`

### `DELETE /v1/identities/{person_id}`
`404` if unknown. `{ "ok": true, "detail": "Removed 2 embeddings" }`

### `POST /v1/index/save` · `POST /v1/index/load`
Persist / restore the gallery (configured paths). The service also auto-loads an
existing gallery on startup.

### `WS /v1/stream/recognize`
Live video. Send **binary** JPEG/PNG frames; receive one JSON message per frame.
Auth: `X-API-Key` header or `?api_key=` query param.
```json
{ "faces": [ { "track_id":4, "bbox":[...], "person_id":"ahmet", "similarity":0.69, "matched":true } ] }
```
With body on, each message also carries a `persons` array (adds body-only tracks
and a `body_bbox` per face track) — see below.

## Body recognition (person ReID)

Opt-in extension: set `FACESTACK_ENABLE_BODY=1` and fetch the models with
`python scripts/fetch_body_models.py` (YOLOv8 person detector + OSNet ReID, both
ONNX, no torch). Off by default — when off, the API is exactly as above.

When on, FaceStack also detects bodies and identifies a person from their body
when the face is not visible. **There is no body-enroll endpoint:** when a body's
face is recognised confidently, that body is auto-enrolled under the same
`person_id`; later, a faceless body is matched against that body gallery.

Body ReID is **appearance/clothing based and day-scoped** — embeddings expire
after `FACESTACK_BODY_TTL_SECONDS` (default 86400) and degrade across outfits/days.
Tune `FACESTACK_BODY_MATCH_THRESHOLD` (default 0.5) separately from the face one.
`POST /v1/index/save|load` persist the body gallery alongside the face gallery.

`POST /v1/recognize` then adds two arrays to the response:

```json
{ "faces": [ /* unchanged */ ],
  "persons": [
    { "person_id":"ahmet", "matched":true, "similarity":0.71, "source":"face",
      "face":{ "bbox":[...],"det_score":0.92,"person_id":"ahmet","similarity":0.71,"matched":true },
      "body":{ "bbox":[...],"det_score":0.86,"similarity":0.71,"matched":true } },
    { "person_id":"aras", "matched":true, "similarity":0.58, "source":"body",
      "face":null, "body":{ "bbox":[...],"det_score":0.84,"similarity":0.58,"matched":true } }
  ],
  "bodies": [ /* every detected body: {bbox,det_score,similarity,matched} */ ] }
```

`source` is `"face"` (identity from the face; `body` attached if linked) or
`"body"` (no usable face; identified via the body gallery). `BodyResult` has no
`person_id` — read it from the enclosing `persons[]` entry.

`WS /v1/stream/recognize` adds a `persons` array per frame (with `source` and
`body_bbox`); `faces` stays present for existing clients.

## Python client SDK

Drop `client/facestack_client.py` into your project (`pip install requests`):

```python
from facestack_client import FaceStackClient

fs = FaceStackClient("http://<host>:8011", api_key="proj-a-key")
fs.enroll("ahmet", "ahmet.jpg")            # path, bytes, or file-like
fs.enroll_batch("ahmet", ["a1.jpg", "a2.jpg", "a3.jpg"])   # several angles at once
for face in fs.recognize("group.jpg"):
    print(face["person_id"], round(face["similarity"], 3), face["matched"])
fs.identities(); fs.save()
```

It depends only on `requests` — no insightface/onnxruntime — so any project can
use it without the engine's heavy deps.

## Typical flow

1. `POST /v1/enroll/batch` each person (a few varied shots — angles/lighting).
2. `POST /v1/index/save` to persist.
3. `POST /v1/recognize` (images) or `WS /v1/stream/recognize` (video) to identify.
4. Tune `FACESTACK_MATCH_THRESHOLD` after [calibration](calibration.md).
