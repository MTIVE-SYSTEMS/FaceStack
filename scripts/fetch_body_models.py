"""Body-model fetcher — download the YOLOv8 person detector + OSNet ReID ONNX.

    python scripts/fetch_body_models.py

Idempotent: skips a model that already exists (nonzero size), prints the final
size in MB, then verifies each file opens as an onnxruntime.InferenceSession and
prints its input/output names and shapes. Nothing here is a runtime dependency —
the service loads these with bare onnxruntime; this script only fetches them.

The runtime stays torch-free: there is NO official Ultralytics .onnx release
asset (only yolov8n.pt), so we pull pre-exported ONNX mirrors. To pin a model to
your own release storage, just point the URL at it.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from facestack.config import Config

# Pre-exported, public ONNX mirrors matching the I/O contract BodyEngine expects:
#   detector: input 1x3x640x640, output 1x84x8400 (YOLOv8 nano, COCO class 0 = person)
#   reid    : input 1x3x256x128, output 1x512     (OSNet, 512-d feature)
# Pin to a commit hash instead of `main` once you record one for reproducibility.
_DET_URL = "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx"
_REID_URL = (
    "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/"
    "osnet_x0_25_msmt17.onnx?download=true"
)


def _download(url: str, dest: Path) -> bool:
    """Fetch `url` to `dest`. Returns True if a download happened, False if skipped."""
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  {dest.name}: already present ({dest.stat().st_size / 1e6:.1f} MB), skipping.")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {dest.name} from {url}")
    try:
        # A real UA — some HF/CDN edges reject the bare urllib default.
        req = urllib.request.Request(url, headers={"User-Agent": "facestack-fetch/1.0"})
        with urllib.request.urlopen(req) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                fh.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"  ERROR: failed to download {dest.name}: {exc}")
        print(
            "  If the mirror is unreachable you can export the model locally in a "
            "throwaway env with torch+ultralytics/torchreid (see DESIGN), then drop "
            f"the .onnx at {dest}."
        )
        return False

    tmp.replace(dest)
    print(f"  saved {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")
    return True


def _verify(path: Path) -> bool:
    """Open `path` as an ORT session and print its I/O contract. True if it loads."""
    if not path.is_file() or path.stat().st_size == 0:
        print(f"  {path.name}: MISSING — cannot verify.")
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping load verification.")
        print("    dev/CPU : pip install onnxruntime")
        print("    the GPU server   : pip install onnxruntime-rocm")
        return False

    try:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001 — surface any load failure to the user
        print(f"  {path.name}: FAILED to load as ORT session: {exc}")
        return False

    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"  {path.name}: loads OK")
    print(f"    input : {inp.name} {inp.shape}")
    print(f"    output: {out.name} {out.shape}")
    return True


def main() -> int:
    cfg = Config()
    det_path = Path(cfg.body_detector_path).expanduser()
    reid_path = Path(cfg.body_reid_path).expanduser()

    print("FaceStack body models")
    print("detector dir:", det_path.parent)
    print()

    print("Detector (YOLOv8 person):")
    _download(_DET_URL, det_path)
    print()
    print("ReID (OSNet):")
    _download(_REID_URL, reid_path)
    print()

    print("Verifying:")
    ok_det = _verify(det_path)
    ok_reid = _verify(reid_path)
    print()

    if ok_det and ok_reid:
        print("Both body models present and loadable. Set FACESTACK_ENABLE_BODY=1 to use them.")
        return 0
    print("One or more body models are missing or failed to load (see above).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
