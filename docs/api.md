# Service API

Start the service:

```bash
bash scripts/serve.sh                 # uses LD_LIBRARY_PATH + FACESTACK_* env
# or directly:
uvicorn facestack.service.app:app --host 0.0.0.0 --port 8000
```

On motis it runs as a systemd service on port **8011** (see [deployment.md](deployment.md)).
Interactive docs are served at `/docs` (Swagger UI).

## Endpoints

### `GET /healthz`
Liveness + the provider actually loaded.
```json
{ "status": "ok", "providers": ["ROCMExecutionProvider","CPUExecutionProvider"],
  "on_gpu": true, "gallery_size": 12, "people": 3 }
```

### `POST /enroll`
Save a face. `multipart/form-data`: `person_id` (str), `file` (image),
`cropped` (bool, default false — set true if `file` is already a cropped face).
```bash
curl -F person_id=ahmet -F file=@ahmet.jpg http://motis:8011/enroll
# {"person_id":"ahmet","enrolled":1}
```
`422` if no face could be enrolled.

### `POST /recognize`
Recognise faces in an image. Form: `file` (image), `cropped` (bool, default false).
```bash
curl -F file=@group.jpg http://motis:8011/recognize
```
```json
{ "faces": [
  { "bbox": [x1,y1,x2,y2], "det_score": 0.92,
    "person_id": "ahmet", "similarity": 0.71, "matched": true },
  { "bbox": [...], "det_score": 0.88,
    "person_id": null, "similarity": 0.13, "matched": false }
] }
```
`person_id` is `null` and `matched` false when no enrolled face clears the threshold.

### `GET /identities`
```json
{ "count": 3, "people": ["ahmet","aras","yigithan"] }
```

### `DELETE /identities/{person_id}`
Remove a person from the gallery. `404` if unknown. `{ "ok": true, "detail": "Removed 2 embeddings" }`

### `POST /index/save` · `POST /index/load`
Persist / restore the gallery to the configured paths. The service also
auto-loads an existing gallery on startup.

### `WS /stream/recognize`
Live video. Client sends **binary** JPEG/PNG frames; server replies with one JSON
message per frame:
```json
{ "faces": [
  { "track_id": 4, "bbox": [x1,y1,x2,y2],
    "person_id": "ahmet", "similarity": 0.69, "matched": true }
] }
```

Python client:
```python
import asyncio, cv2, websockets, json

async def main():
    cap = cv2.VideoCapture(0)
    async with websockets.connect("ws://motis:8011/stream/recognize") as ws:
        while True:
            ok, frame = cap.read()
            if not ok: break
            _, buf = cv2.imencode(".jpg", frame)
            await ws.send(buf.tobytes())
            print(json.loads(await ws.recv())["faces"])

asyncio.run(main())
```

## Typical flow

1. `POST /enroll` each person (a few varied shots).
2. `POST /index/save` to persist.
3. `POST /recognize` (images) or `WS /stream/recognize` (video) to identify.
4. Tune `FACESTACK_MATCH_THRESHOLD` after [calibration](calibration.md).
