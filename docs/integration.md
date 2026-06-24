# FaceStack — Integration Spec (hand this to an agent/developer)

Self-contained reference for calling the FaceStack face-recognition service.
FaceStack recognises **enrolled faces** (1:N identification): you enroll people,
then ask "who is this?" for an image or video frame.

It can **also recognise people from their body** when the face is not visible
(far away / back-turned), tying face and body to one `person_id`. This is an
opt-in extension (`FACESTACK_ENABLE_BODY=1` on the server) — see
[Body recognition](#body-recognition-person-reid). When it is off, the API is
exactly as described in the core sections and the body fields below are absent.

## Connection

```
BASE_URL = http://<host>:8011        # the GPU server's host/IP
API_KEY  = <ask the FaceStack owner; lives in ~/FaceStack/.env on the server>
```

All `/v1/*` requests MUST send header `X-API-Key: <API_KEY>`.
`GET /healthz` needs no key. Missing/invalid key → HTTP `401`.

## Endpoints

| Method | Path | Body (multipart unless noted) | Returns |
|---|---|---|---|
| GET | `/healthz` | — | `{status, providers, on_gpu, gallery_size, people, body_enabled, body_on_gpu, body_gallery_size}` |
| POST | `/v1/enroll` | `person_id` (str), `file` (image), `cropped` (bool, default false) | `{person_id, enrolled}` |
| POST | `/v1/recognize` | `file` (image), `cropped` (bool, default false) | `{faces: [...]}` (+ `persons`, `bodies` when body on) |
| GET | `/v1/identities` | — | `{count, people: [str]}` |
| DELETE | `/v1/identities/{person_id}` | — | `{ok, detail}` |
| POST | `/v1/index/save` | — | `{ok, detail}` — persist gallery (face + body) to disk |
| POST | `/v1/index/load` | — | `{ok, detail}` — restore gallery |
| WS | `/v1/stream/recognize` | binary JPEG/PNG frames | per-frame `{faces:[{track_id,...}]}` (+ `persons` when body on) |

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
KEY=<API_KEY>; BASE=http://<host>:8011
curl "$BASE/healthz"
curl -H "X-API-Key: $KEY" -F person_id=ahmet -F file=@ahmet.jpg  "$BASE/v1/enroll"
curl -H "X-API-Key: $KEY" -F file=@group.jpg                     "$BASE/v1/recognize"
curl -H "X-API-Key: $KEY" "$BASE/v1/identities"
curl -H "X-API-Key: $KEY" -X DELETE "$BASE/v1/identities/ahmet"
```

## Python (zero extra files — just `requests`)

```python
import requests

BASE = "http://<host>:8011"
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
        "ws://<host>:8011/v1/stream/recognize",
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
  saved gallery on restart. `save`/`load` cover the body gallery too when enabled.
- Body recognition (if enabled) is **day-scoped and clothing-based** — don't treat
  a `source:"body"` match as strongly as a face match, and expect it to lapse
  after the TTL or an outfit change. See [Body recognition](#body-recognition-person-reid).

## Body recognition (person ReID)

Optional, opt-in (`FACESTACK_ENABLE_BODY=1` on the server; off by default). When
on, FaceStack also detects each person's **body** and can identify them from it
when the face is missing — useful for someone far away or turned away.

**How identities link (automatic, no extra API):** there is *no* body-enroll
call. Whenever a body's face is recognised confidently, that body's appearance is
auto-saved under the same `person_id`. Later, a body with no visible/matched face
is looked up against that body gallery. So the flow is: enroll faces as usual →
let people be seen face-first once → they become recognisable body-only.

**Important — body recognition is appearance/clothing based**, so it is
**day-scoped**: embeddings expire after a TTL (default 24h) and accuracy drops if
someone changes clothes or across days. It shines within the same session / day /
camera set, not as a durable cross-day identity like the face gallery.

`GET /healthz` reports `body_enabled`, `body_on_gpu`, `body_gallery_size`.

### `/v1/recognize` with body on

`faces[]` is unchanged. Two extra arrays appear:

```json
{ "faces": [ /* unchanged FaceResult objects */ ],
  "persons": [
    { "person_id": "ahmet", "matched": true, "similarity": 0.71,
      "source": "face",                 // identity came from the face
      "face": { "bbox":[...], "det_score":0.92, "person_id":"ahmet", "similarity":0.71, "matched":true },
      "body": { "bbox":[...], "det_score":0.86, "similarity":0.71, "matched":true } },
    { "person_id": "aras", "matched": true, "similarity": 0.58,
      "source": "body",                 // no face here — identified by body
      "face": null,
      "body": { "bbox":[...], "det_score":0.84, "similarity":0.58, "matched":true } }
  ],
  "bodies": [ /* every detected body's BodyResult (convenience) */ ] }
```

- `persons[]` is the unified view: one entry per identity in the scene.
  `source` is `"face"` (identity from the face, body box attached if linked) or
  `"body"` (no usable face — identified via the body gallery).
- `BodyResult` = `{bbox, det_score, similarity, matched}` (no `person_id`; read it
  from the enclosing `persons[]` entry).
- `person_id: null` / `matched: false` ⇒ a detected body that matches no known
  identity yet (e.g. that person has not been seen face-first this day).
- `cropped=true` stays face-only (a single-face crop has no body context).

### WebSocket with body on

Each per-frame message adds a `persons` array alongside `faces`:

```json
{ "faces": [ { "track_id":4, "bbox":[...], "person_id":"ahmet", "similarity":0.69, "matched":true } ],
  "persons": [ { "track_id":4, "bbox":[...], "person_id":"ahmet", "similarity":0.69,
                 "matched":true, "source":"face", "body_bbox":[...] } ] }
```

`faces[]` (face-sourced tracks only) stays present so existing WS clients keep
working; `persons[]` adds body-only tracks (`source:"body"`, `face` absent) and a
`body_bbox` for face tracks that have a linked body.

## License & scope

FaceStack code is AGPL-3.0-or-later (© MTIVE SYSTEMS). The bundled InsightFace
`buffalo_l` models are **non-commercial research only** — do not use this service
in a commercial product without resolving the model license. Intended scope as
shipped: non-commercial / research.
