"""Runtime configuration, overridable via environment variables (FACESTACK_*)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FACESTACK_", env_file=".env", extra="ignore")

    # --- Model ---
    # buffalo_l = SCRFD detector + ArcFace r100 embeddings (512-d). buffalo_s is lighter/faster.
    model_pack: str = "buffalo_l"
    det_size: int = 640  # detector input is (det_size, det_size)
    det_thresh: float = 0.5  # min detection confidence

    # --- Runtime / providers ---
    # "" = auto-detect (prefer ROCm > MIGraphX > CUDA > CPU). Set to force a single provider.
    force_provider: str = ""

    # --- Matching ---
    # ArcFace cosine similarity threshold for a positive identity match.
    # buffalo_l: ~0.40 is a good starting point; calibrate per deployment.
    match_threshold: float = 0.40
    embedding_dim: int = 512

    # --- Index ---
    index_capacity: int = 10_000  # grows automatically beyond this
    index_path: str = "indexes/faces.bin"
    meta_path: str = "indexes/faces.meta.json"

    # --- Video ---
    # Re-run embedding+match for a tracked face every N frames (identity is cached per track).
    reid_interval: int = 15
    track_iou_threshold: float = 0.3
    track_max_age: int = 30  # frames a track survives without a detection
