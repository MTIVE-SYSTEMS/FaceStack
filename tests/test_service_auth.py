"""API-key auth + /v1 routing tests (no model load: a fake recognizer is injected
and the heavy lifespan is skipped by not using TestClient as a context manager).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from facestack.config import Config
from facestack.service.app import create_app

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


class _Idx:
    people = ["alice", "bob"]

    def __len__(self):
        return 5


def _client(api_keys: str = "") -> TestClient:
    app = create_app(Config(api_keys=api_keys))
    app.state.recognizer = SimpleNamespace(
        engine=SimpleNamespace(active_providers=["CPUExecutionProvider"], on_gpu=False),
        index=_Idx(),
    )
    return TestClient(app)  # no `with` => lifespan skipped


def test_healthz_is_open():
    assert _client("k").get("/healthz").status_code == 200


def test_v1_requires_key_when_configured():
    c = _client("secret")
    assert c.get("/v1/identities").status_code == 401
    assert c.get("/v1/identities", headers={"X-API-Key": "nope"}).status_code == 401
    r = c.get("/v1/identities", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_auth_disabled_when_no_keys():
    assert _client("").get("/v1/identities").status_code == 200
