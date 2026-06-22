# Service API

Start the service:

```bash
bash scripts/serve.sh                 # uses LD_LIBRARY_PATH + FACESTACK_* env
# or: uvicorn facestack.service.app:app --host 0.0.0.0 --port 8000
```

On motis it runs as a systemd service on port **8011** (see [deployment.md](deployment.md)).
Interactive docs: `/docs` (Swagger UI). All functional endpoints are under **`/v1`**.

## Authentication

Set one or more keys via `FACESTACK_API_KEYS` (comma-separated, keep them in
`.env`). Every `/v1` request must then send a matching `X-API-Key` header:

```bash
export FACESTACK_API_KEYS="proj-a-key,proj-b-key"   # in motis ~/FaceStack/.env
curl -H "X-API-Key: proj-a-key" ...
```

- No keys configured ⇒ auth is **disabled** (open) — only for local dev.
- Missing/invalid key ⇒ `401`. `/healthz` is always open (liveness probes).

## Endpoints

### `GET /healthz`  (no auth)
```json
{ "status": "ok", "providers": ["ROCMExecutionProvider","CPUExecutionProvider"],
  "on_gpu": true, "gallery_size": 12, "people": 3 }
```

### `POST /v1/enroll`
`multipart/form-data`: `person_id` (str), `file` (image), `cropped` (bool,
default false — true if `file` is already a cropped face). `422` if no face found.
```bash
curl -H "X-API-Key: $KEY" -F person_id=ahmet -F file=@ahmet.jpg http://motis:8011/v1/enroll
# {"person_id":"ahmet","enrolled":1}
```

### `POST /v1/recognize`
Form: `file` (image), `cropped` (bool, default false).
```bash
curl -H "X-API-Key: $KEY" -F file=@group.jpg http://motis:8011/v1/recognize
```
```json
{ "faces": [
  { "bbox":[x1,y1,x2,y2], "det_score":0.92, "person_id":"ahmet", "similarity":0.71, "matched":true },
  { "bbox":[...], "det_score":0.88, "person_id":null, "similarity":0.13, "matched":false }
] }
```
`person_id` is `null` / `matched` false when nothing clears the threshold.

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

## Python client SDK

Drop `client/facestack_client.py` into your project (`pip install requests`):

```python
from facestack_client import FaceStackClient

fs = FaceStackClient("http://motis:8011", api_key="proj-a-key")
fs.enroll("ahmet", "ahmet.jpg")            # path, bytes, or file-like
for face in fs.recognize("group.jpg"):
    print(face["person_id"], round(face["similarity"], 3), face["matched"])
fs.identities(); fs.save()
```

It depends only on `requests` — no insightface/onnxruntime — so any project can
use it without the engine's heavy deps.

## Typical flow

1. `POST /v1/enroll` each person (a few varied shots).
2. `POST /v1/index/save` to persist.
3. `POST /v1/recognize` (images) or `WS /v1/stream/recognize` (video) to identify.
4. Tune `FACESTACK_MATCH_THRESHOLD` after [calibration](calibration.md).
