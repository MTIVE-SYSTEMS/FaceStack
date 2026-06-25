"""FastAPI service exposing the recognition engine over REST + WebSocket.

All functional endpoints live under /v1 and require an `X-API-Key` header when
api_keys is configured. /healthz stays unversioned and unauthenticated for
liveness probes.

  GET  /healthz                     - liveness + provider/gallery info (no auth)
  POST /v1/enroll                   - save a face under a person_id
  POST /v1/recognize                - recognise faces in an image
  GET  /v1/identities               - list enrolled people
  DELETE /v1/identities/{person_id} - remove a person from the gallery
  POST /v1/index/save | /v1/index/load - persist / restore the gallery
  WS   /v1/stream/recognize         - per-frame recognition for live video

When config.enable_body is set, /v1/recognize also returns unified `persons`
(and `bodies`), the WS stream emits a `persons` array, /healthz reports body
model status, and index save/load covers the body gallery too. With it off
(the default) every endpoint behaves exactly as before. There is no manual
body-enroll endpoint by design — bodies auto-enroll on a confident face match.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from ..config import Config
from ..recognizer import Recognizer
from ..schemas import (
    BatchEnrollResponse,
    BodyResult,
    EnrollResponse,
    FaceResult,
    HealthResponse,
    IdentitiesResponse,
    OkResponse,
    PersonResult,
    RecognizeResponse,
)
from ..video import VideoRecognizer

log = logging.getLogger("facestack.service")


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@asynccontextmanager
async def lifespan(app: FastAPI):
    config: Config = app.state.config
    log.info("Loading recognition engine...")
    rec = Recognizer(config)
    if os.path.exists(config.index_path) and os.path.exists(config.meta_path):
        try:
            rec.load()
            log.info("Restored gallery from %s", config.index_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load existing gallery: %s", exc)
    # Warm up the GPU kernels (MIOpen compiles on first inference, ~seconds) so
    # the first real request isn't slow.
    try:
        warm = np.zeros((config.det_size, config.det_size, 3), dtype=np.uint8)
        rec.engine.detect(warm)
        rec.engine.embed_crop(warm)  # exercises the recognition model too
        log.info("Warmup complete (gpu=%s)", rec.engine.on_gpu)
    except Exception as exc:  # noqa: BLE001
        log.warning("Warmup skipped: %s", exc)
    if config.enable_body and rec.body_engine is not None:
        try:
            warm = np.zeros((config.det_size, config.det_size, 3), dtype=np.uint8)
            rec.body_engine.detect_and_embed(warm)  # letterboxes to 640 internally
            log.info("Body warmup complete (gpu=%s)", rec.body_engine.on_gpu)
        except Exception as exc:  # noqa: BLE001
            log.warning("Body warmup skipped: %s", exc)

    app.state.recognizer = rec
    auth = "on" if config.api_key_set else "OFF (open)"
    log.info("Service ready: auth=%s cors=%s", auth, config.cors_origin_list or "disabled")
    yield


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config()
    app = FastAPI(title="FaceStack", version="0.1.0", lifespan=lifespan)
    app.state.config = config

    if config.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origin_list,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def rec() -> Recognizer:
        return app.state.recognizer

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        allowed = config.api_key_set
        if not allowed:
            return  # auth disabled (dev)
        if x_api_key is None or x_api_key not in allowed:
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    # /healthz is unversioned + unauthenticated (liveness probes)
    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        r = rec()
        body_enabled = getattr(r, "_body_enabled", False)
        return HealthResponse(
            status="ok",
            providers=r.engine.active_providers,
            on_gpu=r.engine.on_gpu,
            gallery_size=len(r.index),
            people=len(r.index.people),
            body_enabled=body_enabled,
            body_on_gpu=(r.body_engine.on_gpu if body_enabled and r.body_engine else False),
            body_gallery_size=(len(r.body_index) if body_enabled and r.body_index else 0),
        )

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    @v1.post("/enroll", response_model=EnrollResponse)
    async def enroll(
        person_id: str = Form(...),
        file: UploadFile = File(...),
        cropped: bool = Form(False),
    ) -> EnrollResponse:
        img = _decode(await file.read())
        if cropped:
            count = 1 if rec().enroll_crop(person_id, img) else 0
        else:
            count = rec().enroll_frame(person_id, img)
        if count == 0:
            raise HTTPException(status_code=422, detail="No face could be enrolled")
        return EnrollResponse(person_id=person_id, enrolled=count)

    @v1.post("/enroll/batch", response_model=BatchEnrollResponse)
    async def enroll_batch(
        person_id: str = Form(...),
        files: list[UploadFile] = File(...),
        cropped: bool = Form(False),
    ) -> BatchEnrollResponse:
        """Enroll several photos of one person in a single call.

        A few varied shots (angles/lighting) recognise far more reliably than one.
        """
        images = [_decode(await f.read()) for f in files]
        per_image = rec().enroll_images(person_id, images, cropped=cropped)
        total = sum(per_image)
        if total == 0:
            raise HTTPException(status_code=422, detail="No face could be enrolled from any image")
        return BatchEnrollResponse(
            person_id=person_id, images=len(images), enrolled=total, per_image=per_image
        )

    def _require_body() -> None:
        if not getattr(rec(), "_body_enabled", False):
            raise HTTPException(
                status_code=503, detail="Body recognition is disabled (FACESTACK_ENABLE_BODY=1)"
            )

    @v1.post("/enroll/body", response_model=EnrollResponse)
    async def enroll_body(
        person_id: str = Form(...),
        file: UploadFile = File(...),
        cropped: bool = Form(False),
    ) -> EnrollResponse:
        """Permanently enroll a body photo (front/side/back) — like /v1/enroll but
        for the body gallery. These never expire (unlike auto-enrolled bodies)."""
        _require_body()
        img = _decode(await file.read())
        if cropped:
            count = 1 if rec().enroll_body_crop(person_id, img) else 0
        else:
            count = rec().enroll_body_frame(person_id, img)
        if count == 0:
            raise HTTPException(status_code=422, detail="No body could be enrolled")
        return EnrollResponse(person_id=person_id, enrolled=count)

    @v1.post("/enroll/body/batch", response_model=BatchEnrollResponse)
    async def enroll_body_batch(
        person_id: str = Form(...),
        files: list[UploadFile] = File(...),
        cropped: bool = Form(False),
    ) -> BatchEnrollResponse:
        """Permanently enroll several body photos of one person (multi-angle)."""
        _require_body()
        images = [_decode(await f.read()) for f in files]
        per_image = rec().enroll_body_images(person_id, images, cropped=cropped)
        total = sum(per_image)
        if total == 0:
            raise HTTPException(status_code=422, detail="No body could be enrolled from any image")
        return BatchEnrollResponse(
            person_id=person_id, images=len(images), enrolled=total, per_image=per_image
        )

    @v1.post("/recognize", response_model=RecognizeResponse)
    async def recognize(
        file: UploadFile = File(...),
        cropped: bool = Form(False),
    ) -> RecognizeResponse:
        img = _decode(await file.read())

        def _face_result(f) -> FaceResult:
            return FaceResult(
                bbox=list(f.bbox),
                det_score=f.det_score,
                person_id=f.person_id,
                similarity=f.similarity,
                matched=f.matched,
            )

        body_on = getattr(rec(), "_body_enabled", False)
        # Cropped single-face requests have no scene/bodies — stay face-only.
        if body_on and not cropped:
            # Drive everything from recognize_scene (one face pass), derive faces
            # from the persons' .face fields to avoid double-detecting faces.
            persons = rec().recognize_scene(img)
            faces = [_face_result(p.face) for p in persons if p.face is not None]
            person_results = [
                PersonResult(
                    person_id=p.person_id,
                    matched=p.matched,
                    similarity=p.similarity,
                    source=p.source,
                    face=_face_result(p.face) if p.face else None,
                    body=BodyResult(
                        bbox=list(p.body.bbox),
                        det_score=p.body.det_score,
                        similarity=p.body.similarity,
                        matched=p.body.matched,
                    )
                    if p.body
                    else None,
                )
                for p in persons
            ]
            return RecognizeResponse(
                faces=faces,
                persons=person_results,
                bodies=[pr.body for pr in person_results if pr.body is not None],
            )

        # Body-disabled (or cropped) path — byte-identical to the original.
        if cropped:
            r = rec().recognize_crop(img)
            faces = [r] if r is not None else []
        else:
            faces = rec().recognize_frame(img)
        return RecognizeResponse(faces=[_face_result(f) for f in faces])

    @v1.get("/identities", response_model=IdentitiesResponse)
    def identities() -> IdentitiesResponse:
        people = rec().index.people
        return IdentitiesResponse(count=len(people), people=people)

    @v1.delete("/identities/{person_id}", response_model=OkResponse)
    def delete_identity(person_id: str) -> OkResponse:
        removed = rec().index.remove_person(person_id)
        if removed == 0:
            raise HTTPException(status_code=404, detail="Unknown person_id")
        return OkResponse(detail=f"Removed {removed} embeddings")

    @v1.get("/body/identities", response_model=IdentitiesResponse)
    def body_identities() -> IdentitiesResponse:
        _require_body()
        people = rec().body_index.people
        return IdentitiesResponse(count=len(people), people=people)

    @v1.delete("/body/identities/{person_id}", response_model=OkResponse)
    def delete_body_identity(person_id: str) -> OkResponse:
        _require_body()
        removed = rec().body_index.remove_person(person_id)
        if removed == 0:
            raise HTTPException(status_code=404, detail="Unknown person_id")
        return OkResponse(detail=f"Removed {removed} body embeddings")

    @v1.post("/index/save", response_model=OkResponse)
    def index_save() -> OkResponse:
        rec().save()
        return OkResponse(detail="Gallery saved")

    @v1.post("/index/load", response_model=OkResponse)
    def index_load() -> OkResponse:
        rec().load()
        return OkResponse(detail="Gallery loaded")

    @v1.websocket("/stream/recognize")
    async def stream_recognize(ws: WebSocket) -> None:
        """Client sends binary JPEG/PNG frames; server replies with JSON per frame.

        Auth: send the key as header `X-API-Key` or query param `?api_key=`.
        """
        allowed = config.api_key_set
        if allowed:
            key = ws.headers.get("x-api-key") or ws.query_params.get("api_key")
            if key not in allowed:
                await ws.close(code=1008)  # policy violation
                return
        await ws.accept()
        video = VideoRecognizer(rec())
        try:
            while True:
                data = await ws.receive_bytes()
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    await ws.send_json({"error": "bad frame"})
                    continue
                if config.enable_body:
                    persons = video.process_frame_persons(img)
                    await ws.send_json(
                        {
                            # `faces` stays present so existing WS clients keep
                            # working; populated from the face-sourced persons.
                            "faces": [
                                {
                                    "track_id": p.track_id,
                                    "bbox": list(p.bbox),
                                    "person_id": p.person_id,
                                    "similarity": p.similarity,
                                    "matched": p.matched,
                                }
                                for p in persons
                                if p.source == "face"
                            ],
                            "persons": [
                                {
                                    "track_id": p.track_id,
                                    "bbox": list(p.bbox),
                                    "person_id": p.person_id,
                                    "similarity": p.similarity,
                                    "matched": p.matched,
                                    "source": p.source,
                                    "body_bbox": list(p.body_bbox) if p.body_bbox else None,
                                }
                                for p in persons
                            ],
                        }
                    )
                else:
                    tracked = video.process_frame(img)
                    await ws.send_json(
                        {
                            "faces": [
                                {
                                    "track_id": t.track_id,
                                    "bbox": list(t.bbox),
                                    "person_id": t.person_id,
                                    "similarity": t.similarity,
                                    "matched": t.matched,
                                }
                                for t in tracked
                            ]
                        }
                    )
        except WebSocketDisconnect:
            log.info("Stream client disconnected")

    app.include_router(v1)
    return app


app = create_app()
