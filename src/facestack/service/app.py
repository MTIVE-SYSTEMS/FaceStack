"""FastAPI service exposing the recognition engine over REST + WebSocket.

Endpoints
  GET  /healthz                  - liveness + provider/gallery info
  POST /enroll                   - save a face under a person_id
  POST /recognize                - recognise faces in an image
  GET  /identities               - list enrolled people
  DELETE /identities/{person_id} - remove a person from the gallery
  POST /index/save | /index/load - persist / restore the gallery
  WS   /stream/recognize         - per-frame recognition for live video
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from ..config import Config
from ..recognizer import Recognizer
from ..schemas import (
    EnrollResponse,
    FaceResult,
    HealthResponse,
    IdentitiesResponse,
    OkResponse,
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
    app.state.recognizer = rec
    yield


def create_app(config: Config | None = None) -> FastAPI:
    app = FastAPI(title="FaceStack", version="0.1.0", lifespan=lifespan)
    app.state.config = config or Config()

    def rec() -> Recognizer:
        return app.state.recognizer

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        r = rec()
        return HealthResponse(
            status="ok",
            providers=r.engine.active_providers,
            on_gpu=r.engine.on_gpu,
            gallery_size=len(r.index),
            people=len(r.index.people),
        )

    @app.post("/enroll", response_model=EnrollResponse)
    async def enroll(
        person_id: str = Form(...),
        file: UploadFile = File(...),
        cropped: bool = Form(False),
    ) -> EnrollResponse:
        img = _decode(await file.read())
        if cropped:
            ok = rec().enroll_crop(person_id, img)
            count = 1 if ok else 0
        else:
            count = rec().enroll_frame(person_id, img)
        if count == 0:
            raise HTTPException(status_code=422, detail="No face could be enrolled")
        return EnrollResponse(person_id=person_id, enrolled=count)

    @app.post("/recognize", response_model=RecognizeResponse)
    async def recognize(
        file: UploadFile = File(...),
        cropped: bool = Form(False),
    ) -> RecognizeResponse:
        img = _decode(await file.read())
        if cropped:
            r = rec().recognize_crop(img)
            faces = [r] if r is not None else []
        else:
            faces = rec().recognize_frame(img)
        return RecognizeResponse(
            faces=[
                FaceResult(
                    bbox=list(f.bbox),
                    det_score=f.det_score,
                    person_id=f.person_id,
                    similarity=f.similarity,
                    matched=f.matched,
                )
                for f in faces
            ]
        )

    @app.get("/identities", response_model=IdentitiesResponse)
    def identities() -> IdentitiesResponse:
        people = rec().index.people
        return IdentitiesResponse(count=len(people), people=people)

    @app.delete("/identities/{person_id}", response_model=OkResponse)
    def delete_identity(person_id: str) -> OkResponse:
        removed = rec().index.remove_person(person_id)
        if removed == 0:
            raise HTTPException(status_code=404, detail="Unknown person_id")
        return OkResponse(detail=f"Removed {removed} embeddings")

    @app.post("/index/save", response_model=OkResponse)
    def index_save() -> OkResponse:
        rec().save()
        return OkResponse(detail="Gallery saved")

    @app.post("/index/load", response_model=OkResponse)
    def index_load() -> OkResponse:
        rec().load()
        return OkResponse(detail="Gallery loaded")

    @app.websocket("/stream/recognize")
    async def stream_recognize(ws: WebSocket) -> None:
        """Client sends binary JPEG/PNG frames; server replies with JSON per frame."""
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

    return app


app = create_app()
