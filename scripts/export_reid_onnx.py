"""Export a torchreid OSNet ReID model to a single self-contained ONNX file.

The FaceStack runtime is torch-free; this one-off produces the `.onnx` the body
engine loads. Run it in a THROWAWAY environment (not the service venv):

    python3 -m venv /tmp/reid && . /tmp/reid/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install torchreid gdown onnx onnxruntime scipy opencv-python-headless tensorboard onnxscript
    python scripts/export_reid_onnx.py --model osnet_ain_x1_0 --out osnet_ain_x1_0_msmt17.onnx

Then point the service at it:  FACESTACK_BODY_REID_PATH=/path/to/osnet_ain_x1_0_msmt17.onnx
NOTE: changing the ReID model invalidates existing body embeddings — wipe the
body gallery (indexes/bodies.*) and re-enrol.

`osnet_ain_x1_0` (AIN = domain-generalizable) is recommended over the default
`osnet_x0_25`: notably stronger cross-view / cross-camera matching (i.e. better
at recognising people turned away or from a different angle). Output is 512-d for
both, input 256x128 — a drop-in swap.
"""

from __future__ import annotations

import argparse

# torchreid MSMT17 model-zoo weights (Google Drive file ids).
_WEIGHTS = {
    "osnet_x1_0": "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
    "osnet_ain_x1_0": "1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="osnet_ain_x1_0", choices=sorted(_WEIGHTS))
    ap.add_argument("--out", required=True, help="output .onnx path (single file)")
    args = ap.parse_args()

    import gdown
    import onnx
    import torch
    import torchreid

    pth = f"{args.model}_msmt17.pth"
    gdown.download(id=_WEIGHTS[args.model], output=pth, quiet=False)

    model = torchreid.models.build_model(name=args.model, num_classes=1000, pretrained=False)
    torchreid.utils.load_pretrained_weights(model, pth)
    model.eval()  # OSNet returns the 512-d feature vector in eval mode

    tmp = args.out + ".tmp.onnx"
    torch.onnx.export(
        model,
        torch.randn(1, 3, 256, 128),
        tmp,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )
    # Newer exporters may write weights as external data; consolidate to one file.
    onnx.save_model(onnx.load(tmp), args.out, save_as_external_data=False)
    print(f"exported single-file ONNX -> {args.out}")


if __name__ == "__main__":
    main()
