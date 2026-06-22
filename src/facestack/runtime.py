"""ONNX Runtime execution-provider selection.

The same code runs on the AMD GPU (motis, ROCm) and on a CPU dev box: we probe
the providers ONNX Runtime actually built with and pick the best available,
always keeping CPU as a fallback.
"""

from __future__ import annotations

import logging

log = logging.getLogger("facestack.runtime")

# Preference order, best first. ROCm is the target on motis (RX 7900 XT / RDNA3).
#
# MIGraphX is deliberately NOT in the default list: on motis (ROCm 7.2.4) it is
# unusable both ways we tried it — AMD's rocm-rel-7.2.4 migraphx wheel produces
# numerically wrong SCRFD output (thousands of phantom detections), and the PyPI
# onnxruntime-rocm wheel ships a MIGraphX provider lib linked against ROCm 6
# (libamdhip64.so.6) that won't load on ROCm 7. ROCMExecutionProvider is correct
# and clean. MIGraphX can still be forced explicitly via force_provider.
_PREFERENCE = [
    "ROCMExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

_GPU_PROVIDERS = {
    "ROCMExecutionProvider",
    "MIGraphXExecutionProvider",
    "CUDAExecutionProvider",
}


def select_providers(force_provider: str = "") -> list[str]:
    """Return an ordered provider list for onnxruntime, CPU always last."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())

    if force_provider:
        if force_provider not in available:
            raise RuntimeError(
                f"Forced provider {force_provider!r} not available. "
                f"Built providers: {sorted(available)}"
            )
        providers = [force_provider]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    providers = [p for p in _PREFERENCE if p in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    log.info("Selected ONNX Runtime providers: %s", providers)
    return providers


def ctx_id_for(providers: list[str]) -> int:
    """InsightFace ctx_id: 0 = first GPU, -1 = CPU."""
    return 0 if providers and providers[0] in _GPU_PROVIDERS else -1


def using_gpu(providers: list[str]) -> bool:
    return bool(providers) and providers[0] in _GPU_PROVIDERS
