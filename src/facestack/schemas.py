"""Pydantic schemas for the REST API boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceResult(BaseModel):
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")
    det_score: float
    person_id: str | None = Field(default=None, description="Matched identity, or null")
    similarity: float
    matched: bool


class BodyResult(BaseModel):
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")
    det_score: float
    similarity: float
    matched: bool


class PersonResult(BaseModel):
    person_id: str | None = None
    matched: bool
    similarity: float
    source: str = Field(description="'face' or 'body'")
    face: FaceResult | None = None
    body: BodyResult | None = None


class RecognizeResponse(BaseModel):
    # `faces` stays required + unchanged for backward-compat; persons/bodies are
    # only populated when body recognition is enabled.
    faces: list[FaceResult]
    persons: list[PersonResult] | None = None
    bodies: list[BodyResult] | None = None


class EnrollResponse(BaseModel):
    person_id: str
    enrolled: int = Field(description="Number of faces added to the gallery")


class BatchEnrollResponse(BaseModel):
    person_id: str
    images: int = Field(description="Number of images received")
    enrolled: int = Field(description="Total faces added across all images")
    per_image: list[int] = Field(description="Faces enrolled from each image, in order")


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
    # Body fields default so existing clients/tests stay valid when body is off.
    body_enabled: bool = False
    body_on_gpu: bool = False
    body_gallery_size: int = 0
