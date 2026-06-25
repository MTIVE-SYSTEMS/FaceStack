"""Bulk-enroll a dataset of faces into a running FaceStack service.

Layout: one folder per person, images inside (the folder name is the person_id):

    dataset/
      Aras Tugra Atay/   img1.jpg  img2.jpg  ...
      Yigithan Atay/     a.png     b.jpg     ...

Each person's photos are enrolled in one /v1/enroll/batch call (a few varied
angles recognise far more reliably than a single shot), then the gallery is
persisted with /v1/index/save.

    python scripts/enroll_dataset.py dataset \
        --base-url http://127.0.0.1:8011 --api-key "$FACESTACK_API_KEY"

Talks to the live service over HTTP (so it updates the gallery the server is
actually serving) — depends only on `requests`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _person_dirs(root: Path) -> list[Path]:
    return sorted(d for d in root.iterdir() if d.is_dir())


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _EXTS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bulk-enroll a face dataset into FaceStack.")
    ap.add_argument("root", nargs="?", default="dataset", help="dataset root (one folder per person)")
    ap.add_argument("--base-url", default=os.environ.get("FACESTACK_BASE_URL", "http://127.0.0.1:8011"))
    ap.add_argument("--api-key", default=os.environ.get("FACESTACK_API_KEY", ""))
    ap.add_argument("--cropped", action="store_true", help="images are already tight crops")
    ap.add_argument(
        "--target",
        choices=("face", "body", "both"),
        default="face",
        help="enrol faces, bodies (permanent ReID gallery), or both",
    )
    ap.add_argument("--no-save", action="store_true", help="don't call /v1/index/save at the end")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: dataset root not found: {root}")
        return 1
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    base = args.base_url.rstrip("/")

    people = _person_dirs(root)
    if not people:
        print(f"ERROR: no person sub-folders under {root}")
        return 1

    # face -> /v1/enroll/batch ; body -> /v1/enroll/body/batch (permanent gallery)
    targets = {"face": ["face"], "body": ["body"], "both": ["face", "body"]}[args.target]
    endpoints = {"face": "/v1/enroll/batch", "body": "/v1/enroll/body/batch"}

    print(f"Enrolling {len(people)} people from {root} -> {base} (target={args.target})\n")
    total, total_imgs, failures = 0, 0, 0
    for folder in people:
        person = folder.name
        imgs = _images(folder)
        if not imgs:
            print(f"  {person}: no images, skipped")
            continue
        for kind in targets:
            opened = [open(p, "rb") for p in imgs]
            try:
                r = requests.post(
                    f"{base}{endpoints[kind]}",
                    headers=headers,
                    data={"person_id": person, "cropped": str(args.cropped).lower()},
                    files=[("files", f) for f in opened],
                    timeout=args.timeout,
                )
            finally:
                for f in opened:
                    f.close()

            if r.ok:
                d = r.json()
                total += d["enrolled"]
                total_imgs += d["images"]
                blank = [imgs[i].name for i, c in enumerate(d["per_image"]) if c == 0]
                note = f"  ({len(blank)} with no {kind}: {', '.join(blank)})" if blank else ""
                print(f"  {person} [{kind}]: {d['enrolled']} from {d['images']} images{note}")
            else:
                failures += 1
                detail = r.text
                try:
                    detail = r.json().get("detail", detail)
                except Exception:  # noqa: BLE001
                    pass
                print(f"  {person} [{kind}]: FAILED ({r.status_code}: {detail})")

    print(f"\nTotal: {total} embeddings from {total_imgs} image-passes, {failures} failed")

    if not args.no_save and total:
        r = requests.post(f"{base}/v1/index/save", headers=headers, timeout=args.timeout)
        print("Saved gallery." if r.ok else f"Save FAILED ({r.status_code})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
