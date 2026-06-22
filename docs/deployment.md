# Deployment

## Dev / CPU

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt      # ships CPU onnxruntime
pip install -e .
python scripts/check_env.py          # CPUExecutionProvider expected
```

> Python 3.10–3.12 only (the ML stack has no 3.13/3.14 wheels yet).

## motis (AMD RX 7900 XT, ROCm 7.2.4) — verified

After the base install above, enable the GPU:

```bash
bash scripts/setup_rocm_motis.sh                          # ROCm-EP wheel + compat symlink
LD_LIBRARY_PATH=$HOME/rocm-compat python scripts/check_env.py   # ROCMExecutionProvider expected
```

### The AMD ROCm 7.2.4 onnxruntime trap

This cost real debugging time — documented so nobody repeats it:

1. **MIGraphX EP is unusable.** AMD's `rocm-rel-7.2.4` MIGraphX wheel returns
   numerically wrong SCRFD output (thousands of phantom detections); the PyPI
   wheel's MIGraphX lib links ROCm 6 and won't load. `runtime.py` excludes
   MIGraphX from auto-selection.
2. **No plain ROCm-EP wheel ships for 7.2.x.** The ROCm-EP wheel built for **7.0**
   (`onnxruntime_rocm-1.22.1`, from `repo.radeon.com/.../rocm-rel-7.0/`) is
   ABI-compatible with 7.2.4 — except it needs `librocm_smi64.so.7` while 7.2.4
   ships `.so.1`. `setup_rocm_motis.sh` symlinks it (rocm_smi is device
   introspection only, not compute).
3. **The PyPI `onnxruntime-rocm` (ROCm 6) silently falls back to CPU** here
   (`libhipblas.so.2` missing) — avoid it. `on_gpu` / `/healthz` expose any such
   silent fallback.

Validated end-to-end on the GPU: correct match + correct rejection, ~82% GPU
utilisation, 184–302 FPS on the video path.

## Persistent service (systemd, no root)

```bash
mkdir -p ~/.config/systemd/user
cp deploy/facestack.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now facestack       # autostart on boot
sudo loginctl enable-linger aras              # survive logout/reboot (one-time)
```

Runs `uvicorn` on **port 8011** (8000 is taken on motis) with `LD_LIBRARY_PATH`
baked in and `Restart=always`.

```bash
systemctl --user status facestack
systemctl --user restart facestack
journalctl --user -u facestack -f
```

## Docker (alternative)

`docker/Dockerfile.rocm` builds a reproducible GPU image (ROCm 7.2 base +
`onnxruntime-rocm==1.22.2.post1`). The native venv above is the primary path on
motis. Mount a volume for `~/.insightface` to cache the model download.

## Configuration (`FACESTACK_*` env vars)

| Var | Default | Meaning |
|---|---|---|
| `FACESTACK_MODEL_PACK` | `buffalo_l` | InsightFace pack (`buffalo_s` = faster, less accurate) |
| `FACESTACK_DET_SIZE` | `640` | Detector input size; `320` is faster on small/near faces |
| `FACESTACK_DET_THRESH` | `0.5` | Min detection confidence |
| `FACESTACK_MATCH_THRESHOLD` | `0.40` | Cosine cutoff for a positive match (calibrate this) |
| `FACESTACK_FORCE_PROVIDER` | `` | Force one ONNX provider (else auto: ROCm > CUDA > CPU) |
| `FACESTACK_REID_INTERVAL` | `15` | Frames between re-embeds per track (video) |
| `FACESTACK_TRACK_IOU_THRESHOLD` | `0.3` | IoU to link a detection to an existing track |
| `FACESTACK_TRACK_MAX_AGE` | `30` | Frames a track survives without a detection |
| `FACESTACK_INDEX_PATH` / `_META_PATH` | `indexes/faces.*` | Gallery persistence paths |
| `FACESTACK_INDEX_CAPACITY` | `10000` | Initial gallery capacity (grows automatically) |
