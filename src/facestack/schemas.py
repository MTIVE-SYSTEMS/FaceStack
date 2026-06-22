"""Pydantic schemas for the REST API boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceResult(BaseModel):
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")
    det_score: float
    person_id: str | None = Field(default=None, description="Matched identity, or null")
    similarity: float
    matched: bool


class RecognizeResponse(BaseModel):
    faces: list[FaceResult]


class EnrollResponse(BaseModel):
    person_id: str
    enrolled: int = Field(description="Number of faces added to the gallery")


class IdentitiesResponse(BaseModel):
    count: int
    people: list[str]


class OkResponse(BaseModel):
    ok: bool = True
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    providers: list[str]
    on_gpu: bool
    gallery_size: int
    people: int
