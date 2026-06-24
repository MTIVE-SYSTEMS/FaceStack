"""Environment doctor — run on the GPU server to confirm the AMD GPU path is live.

    python scripts/check_env.py
"""

from __future__ import annotations

import sys


def main() -> int:
    print("Python:", sys.version.split()[0])

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.")
        print("  dev/CPU : pip install onnxruntime")
        print("  the GPU server   : pip install onnxruntime-rocm")
        return 1

    print("onnxruntime:", ort.__version__)
    providers = ort.get_available_providers()
    print("available providers:", providers)

    gpu = [p for p in providers if p in {"ROCMExecutionProvider", "MIGraphXExecutionProvider", "CUDAExecutionProvider"}]
    if gpu:
        print(f"GPU acceleration available via: {gpu[0]}")
    else:
        print("No GPU provider built into this onnxruntime — will run on CPU.")
        print("On the GPU server (AMD RX 7900 XT) install onnxruntime-rocm to enable ROCMExecutionProvider.")

    for mod in ("numpy", "cv2", "hnswlib", "insightface", "fastapi"):
        try:
            m = __import__(mod)
            print(f"{mod}:", getattr(m, "__version__", "ok"))
        except ImportError:
            print(f"{mod}: MISSING")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
