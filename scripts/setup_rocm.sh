#!/usr/bin/env bash
# One-shot ROCm GPU setup for the GPU server (RX 7900 XT, ROCm 7.2.4).
#
# Why this is needed: AMD ships no plain ROCm-EP onnxruntime wheel for 7.2.x
# (only MIGraphX, which mis-computes SCRFD). The ROCm-EP wheel built for ROCm
# 7.0 works on 7.2.4 (same major) except it links librocm_smi64.so.7, while
# 7.2.4 ships .so.1 — so we provide a compat symlink. rocm_smi is only used for
# device introspection, not the compute path.
set -euo pipefail

ROCM_DIR="${ROCM_DIR:-/opt/rocm-7.2.4}"
COMPAT_DIR="${COMPAT_DIR:-$HOME/rocm-compat}"
WHEEL="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/onnxruntime_rocm-1.22.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

echo ">> Installing ROCm-EP onnxruntime (built for ROCm 7.0, ABI-compatible with 7.2.x)"
python -m pip uninstall -y onnxruntime onnxruntime-rocm onnxruntime-migraphx 2>/dev/null || true
python -m pip install "$WHEEL"

echo ">> Creating librocm_smi64.so.7 compat symlink in $COMPAT_DIR"
mkdir -p "$COMPAT_DIR"
ln -sf "$ROCM_DIR/lib/librocm_smi64.so.1" "$COMPAT_DIR/librocm_smi64.so.7"

echo ">> Done. Run the service via scripts/serve.sh (it exports LD_LIBRARY_PATH=$COMPAT_DIR)."
echo ">> Verify: LD_LIBRARY_PATH=$COMPAT_DIR python scripts/check_env.py  # expect ROCMExecutionProvider"
