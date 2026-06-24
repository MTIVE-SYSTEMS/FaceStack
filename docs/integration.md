# FaceStack — Integration Spec (hand this to an agent/developer)

Self-contained reference for calling the FaceStack face-recognition service.
FaceStack recognises **enrolled faces** (1:N identification): you enroll people,
then ask "who is this?" for an image or video frame.

## Connection

```
BASE_URL = http://motis:8011        # or the motis host IP
API_KEY  = <ask the FaceStack owner; lives in motis ~/FaceStack/.env>
```

All `/v1/*` requests MUST send header `X-API-Key: <API_KEY>`.
`GET /healthz` needs no key. Missing/invalid key → HTTP `401`.

## Endpoints

| Method | Path | Body (multipart unless noted) | Returns |
|---|---|---|---|
| GET | `/healthz` | — | `{status, providers, on_gpu, gallery_size, people}` |
| POST | `/v1/enroll` | `person_id` (str), `file` (image), `cropped` (bool, default false) | `{person_id, enrolled}` |
| POST | `/v1/recognize` | `file` (image), `cropped` (bool, default false) | `{faces: [...]}` |
| GET | `/v1/identities` | — | `{count, people: [str]}` |
| DELETE | `/v1/identities/{person_id}` | — | `{ok, detail}` |
| POST | `/v1/index/save` | — | `{ok, detail}` — persist gallery to disk |
| POST | `/v1/index/load` | — | `{ok, detail}` — restore gallery |
| WS | `/v1/stream/recognize` | binary JPEG/PNG frames | per-frame `{faces:[{track_id,...}]}` |

`cropped=true` means `file` is already a tight single-face crop (skips detection).
Use `cropped=false` (default) for full photos/frames.

### `faces[]` object (from `/v1/recognize`)
```json
{ "bbox": [x1, y1, x2, y2],      // pixels
  "det_score": 0.92,             // detector confidence
  "person_id": "ahmet",          // null if no enrolled match
  "similarity": 0.71,            // cosine similarity to best match (0..1)
  "matched": true }              // similarity >= server threshold
```
`person_id: null` / `matched: false` ⇒ unknown face (no one enrolled is close enough).

## HTTP status codes

| Code | Meaning |
|---|---|
| 200 | OK |
| 401 | missing/invalid `X-API-Key` |
| 422 | `/v1/enroll`: no face found in the image |
| 400 | unreadable/!decodable image |
| 404 | `DELETE /v1/identities/{id}`: unknown person |

## curl

```bash
KEY=<API_KEY>; BASE=http://motis:8011
curl "$BASE/healthz"
curl -H "X-API-Key: $KEY" -F person_id=ahmet -F file=@ahmet.jpg  "$BASE/v1/enroll"
curl -H "X-API-Key: $KEY" -F file=@group.jpg                     "$BASE/v1/recognize"
curl -H "X-API-Key: $KEY" "$BASE/v1/identities"
curl -H "X-API-Key: $KEY" -X DELETE "$BASE/v1/identities/ahmet"
```

## Python (zero extra files — just `requests`)

```python
import requests

BASE = "http://motis:8011"
HEAD = {"X-API-Key": "<API_KEY>"}

def enroll(person_id, path):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/v1/enroll", headers=HEAD,
                          data={"person_id": person_id}, files={"file": f})
    r.raise_for_status(); return r.json()

def recognize(path):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/v1/recognize", headers=HEAD, files={"file": f})
    r.raise_for_status(); return r.json()["faces"]

# usage
enroll("ahmet", "ahmet1.jpg")
for face in recognize("group.jpg"):
    who = face["person_id"] if face["matched"] else "unknown"
    print(who, round(face["similarity"], 3), face["bbox"])
```

A fuller drop-in client is at `client/facestack_client.py` (in the repo).

## Live video (WebSocket)

```python
import asyncio, cv2, json, websockets

async def run():
    async with websockets.connect(
        "ws://motis:8011/v1/stream/recognize",
        additional_headers={"X-API-Key": "<API_KEY>"},   # or ?api_key=<key> in the URL
    ) as ws:
        cap = cv2.VideoCapture("rtsp://...")              # or 0 for a webcam
        while True:
            ok, frame = cap.read()
            if not ok: break
            await ws.send(cv2.imencode(".jpg", frame)[1].tobytes())
            for f in json.loads(await ws.recv())["faces"]:
                print(f["track_id"], f["person_id"], f["matched"])

asyncio.run(run())
```

## Task recipes

- **Recognise who's in a photo** → `POST /v1/recognize`; read `faces[].person_id`
  where `matched` is true.
- **Add a new person** → `POST /v1/enroll` a few times with varied photos of them
  (different angle/light), then `POST /v1/index/save` to persist.
- **List / remove people** → `GET /v1/identities`, `DELETE /v1/identities/{id}`.
- **Live camera** → `WS /v1/stream/recognize`, send frames, read per-frame faces.

## Gotchas

- One face per enrollment image. A multi-face photo enrolls every face under that
  `person_id` (or, with `cropped=true`, may embed the wrong face). Crop first.
- `matched` uses a server-side similarity threshold (default 0.40, tunable by the
  owner). Don't hardcode your own threshold — trust `matched`, or read `similarity`.
- Enrollment is in-memory until `POST /v1/index/save`; the service auto-loads the
  saved gallery on restart.

## License & scope

FaceStack code is AGPL-3.0-or-later (© MTIVE SYSTEMS). The bundled InsightFace
`buffalo_l` models are **non-commercial research only** — do not use this service
in a commercial product without resolving the model license. Intended scope as
shipped: non-commercial / research.
