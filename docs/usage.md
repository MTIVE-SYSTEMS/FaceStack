# Library usage

Install (see [deployment.md](deployment.md) for the ROCm/GPU variant):

```bash
pip install -r requirements.txt && pip install -e .
```

## Quick start

```python
import cv2
from facestack import Recognizer

rec = Recognizer()                       # loads buffalo_l (downloads on first run)

# enroll saved faces — several varied shots per person recognise far better
rec.enroll_images("ahmet", [cv2.imread(p) for p in ("ahmet1.jpg", "ahmet2.jpg", "ahmet3.jpg")])
rec.enroll_frame("aras", cv2.imread("aras1.jpg"))   # single shot still works

# recognise
for face in rec.recognize_frame(cv2.imread("group.jpg")):
    who = face.person_id if face.matched else "unknown"
    print(who, round(face.similarity, 3), face.bbox)

rec.save()                               # persist gallery to disk (config paths)
# later: rec.load()
```

## API surface

### `Recognizer`
- `enroll_frame(person_id, img_bgr) -> int` — enroll every face found; returns count.
- `enroll_images(person_id, [img_bgr], cropped=False) -> list[int]` — enroll several
  photos at once (varied angles → far more robust); returns the per-image face count.
- `enroll_crop(person_id, img_bgr) -> bool` — enroll a single cropped face.
- `recognize_frame(img_bgr) -> list[RecognizedFace]` — locate + match every face.
- `recognize_crop(img_bgr) -> RecognizedFace | None` — match one cropped face.
- `save()` / `load()` — persist/restore the gallery (uses `Config` paths).

`RecognizedFace`: `bbox`, `det_score`, `person_id` (None if unknown), `similarity`, `matched`.

### `FaceEngine` (lower level)
- `embed_frame(img) -> list[DetectedFace]` — detect + align + embed all faces.
- `embed_crop(img) -> DetectedFace | None` — embed one cropped face.
- `detect(img) -> list[(bbox, det_score, kps)]` — detection only (per-frame video cost).
- `embed_aligned(img, kps) -> np.ndarray` — embed one located face via its landmarks.
- `active_providers`, `on_gpu` — the provider actually loaded.

### `FaceIndex` (the gallery)
- `add(person_id, embedding)`, `add_many(person_id, [embeddings])`
- `search(embedding, k) -> list[Match]`, `recognize(embedding) -> Match | None`
- `remove_person(person_id) -> int`, `people`, `len(index)`
- `save(index_path, meta_path)`, `FaceIndex.load(index_path, meta_path)`

`Match`: `person_id`, `similarity`, `matched`.

## Live video

```python
from facestack import Recognizer
from facestack.video import VideoRecognizer
import cv2

rec = Recognizer(); rec.load()           # load an enrolled gallery
vr = VideoRecognizer(rec)                # one instance per camera/stream

cap = cv2.VideoCapture("rtsp://...")
while True:
    ok, frame = cap.read()
    if not ok: break
    for t in vr.process_frame(frame):    # detect every frame; embed only on re-id
        print(t.track_id, t.person_id, t.matched, t.bbox)
```

`process_frame` returns `TrackedFace`: `track_id`, `bbox`, `person_id`,
`similarity`, `matched`. Identity is cached per track and refreshed every
`reid_interval` frames (config), so embedding does not run on every frame.

## Enrollment best practices

- **Several shots per person**, varied: angles, distance, lighting, glasses on/off.
  Use `enroll_images()` (library), `POST /v1/enroll/batch` (service), or
  `scripts/enroll_dataset.py` to bulk-load a `dataset/<name>/*.jpg` tree.
- One face per enrollment image — for group photos, crop first, or expect every
  face to be enrolled under that `person_id`.
- After collecting data, **calibrate the threshold** — see [calibration.md](calibration.md).
