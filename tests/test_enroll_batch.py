"""POST /v1/enroll/batch — multi-photo enrollment (no model load; fake recognizer)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from facestack.config import Config
from facestack.service.app import create_app

pytest.importorskip("httpx")
import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def _png(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def _client(per_image):
    app = create_app(Config(api_keys=""))
    app.state.recognizer = SimpleNamespace(
        enroll_images=lambda person_id, images, cropped=False: per_image[: len(images)],
    )
    return TestClient(app)


def test_batch_enroll_sums_and_reports_per_image():
    c = _client([1, 2, 0])
    files = [("files", (f"a{i}.png", _png(i), "image/png")) for i in range(3)]
    r = c.post("/v1/enroll/batch", data={"person_id": "aras"}, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["person_id"] == "aras"
    assert body["images"] == 3
    assert body["enrolled"] == 3
    assert body["per_image"] == [1, 2, 0]


def test_batch_enroll_422_when_no_face_anywhere():
    c = _client([0, 0])
    files = [("files", (f"b{i}.png", _png(i + 9), "image/png")) for i in range(2)]
    r = c.post("/v1/enroll/batch", data={"person_id": "nobody"}, files=files)
    assert r.status_code == 422
