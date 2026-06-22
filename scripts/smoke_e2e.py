"""End-to-end smoke test on a real photo (no external download for the image:
uses the group photo bundled with insightface). Proves pixels -> embedding ->
recognition of a *saved* face actually works.
"""

from __future__ import annotations

import numpy as np
from insightface.data import get_image as ins_get_image

from facestack import Recognizer


def main() -> int:
    rec = Recognizer()  # loads buffalo_l (downloads on first run)
    print(f"Engine ready | providers={rec.engine.providers} | gpu={rec.engine.on_gpu}")

    img = ins_get_image("t1")  # bundled group photo with several faces
    faces = rec.engine.embed_frame(img)
    print(f"\nFull-frame detection: found {len(faces)} faces")
    if not faces:
        print("ERROR: no faces detected")
        return 1

    # Enroll ONE person from their face, then recognise the whole frame.
    faces_sorted = sorted(faces, key=lambda f: f.bbox[0])
    target = faces_sorted[0]
    x1, y1, x2, y2 = (int(v) for v in target.bbox)
    rec.index.add("person_0", target.embedding)
    print(f"Enrolled person_0 from face at bbox=({x1},{y1},{x2},{y2})")

    print("\n--- recognize_frame on the same photo ---")
    matched_count = 0
    for i, r in enumerate(rec.recognize_frame(img)):
        who = r.person_id if r.matched else "UNKNOWN"
        if r.matched:
            matched_count += 1
        print(f"  face[{i}] det={r.det_score:.2f}  -> {who:9s}  sim={r.similarity:.3f}")

    # The enrolled face must come back as person_0; the other people must not.
    assert matched_count == 1, f"expected exactly 1 match (person_0), got {matched_count}"
    print(f"\nOK: exactly the enrolled face was recognised ({matched_count}/1).")

    # --- cropped-face path ---
    crop = img[y1:y2, x1:x2].copy()
    rc = rec.recognize_crop(crop)
    print(f"\n--- recognize_crop on person_0's crop ---")
    print(f"  -> {rc.person_id if rc.matched else 'UNKNOWN'}  sim={rc.similarity:.3f}")
    assert rc is not None and rc.matched and rc.person_id == "person_0"
    print("OK: cropped-face path recognised person_0.")

    print("\nALL GOOD — real photo -> embedding -> recognition works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
