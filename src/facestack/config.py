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

    # --- Body recognition (opt-in; default OFF) ---
    # When False, no body models load and the package behaves exactly as before.
    # Body ReID is clothing/appearance based, so embeddings expire after a TTL.
    enable_body: bool = False
    body_detector_path: str = "~/.facestack/models/yolov8_person.onnx"
    body_reid_path: str = "~/.facestack/models/osnet_reid.onnx"
    body_det_size: int = 640  # YOLOv8 letterbox input is (body_det_size, body_det_size)
    body_det_thresh: float = 0.5  # YOLOv8 person-class score threshold
    body_nms_iou: float = 0.45  # NMS IoU for person boxes
    body_match_threshold: float = 0.5  # OSNet cosine threshold for a body match
    body_embedding_dim: int = 512  # OSNet output dim
    body_reid_size: tuple[int, int] = (256, 128)  # OSNet input (H, W); cv2 wants (W, H)
    body_ttl_seconds: float = 86400.0  # body embeddings expire after 1 day
    body_index_path: str = "indexes/bodies.bin"
    body_meta_path: str = "indexes/bodies.meta.json"
    body_link_containment: float = 0.6  # min face-in-body containment to link

    # --- Service / access control ---
    # Comma-separated API keys. Empty = auth disabled (dev). When set, every /v1
    # request must send a matching `X-API-Key` header. Keep keys in .env, not code.
    api_keys: str = ""
    # Comma-separated CORS origins for browser clients. Empty = CORS disabled.
    cors_origins: str = ""

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
