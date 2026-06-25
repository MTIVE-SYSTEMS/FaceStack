"""Standalone FaceStack client — drop this single file into any project.

Depends only on `requests` (no insightface/onnxruntime). Talks to the /v1 API.

    from facestack_client import FaceStackClient

    fs = FaceStackClient("http://<host>:8011", api_key="...")
    fs.enroll("ahmet", "ahmet.jpg")
    for face in fs.recognize("group.jpg"):
        print(face["person_id"], face["similarity"], face["matched"])
"""

from __future__ import annotations

from typing import Any, BinaryIO

import requests


class FaceStackError(RuntimeError):
    pass


class FaceStackClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"X-API-Key": api_key} if api_key else {}

    # --- internals ---
    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def _check(self, r: requests.Response) -> Any:
        if not r.ok:
            detail = r.text
            try:
                detail = r.json().get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            raise FaceStackError(f"{r.status_code}: {detail}")
        return r.json()

    @staticmethod
    def _as_file(image: str | bytes | BinaryIO):
        if isinstance(image, str):
            return open(image, "rb")  # noqa: SIM115 (closed by caller via context below)
        if isinstance(image, (bytes, bytearray)):
            return ("image.jpg", bytes(image))
        return image  # already a file-like object

    # --- API ---
    def health(self) -> dict:
        return self._check(requests.get(self._url("/healthz"), timeout=self.timeout))

    def enroll(self, person_id: str, image: str | bytes | BinaryIO, cropped: bool = False) -> dict:
        f = self._as_file(image)
        try:
            r = requests.post(
                self._url("/v1/enroll"),
                headers=self._headers,
                data={"person_id": person_id, "cropped": str(cropped).lower()},
                files={"file": f},
                timeout=self.timeout,
            )
        finally:
            if hasattr(f, "close"):
                f.close()
        return self._check(r)

    def enroll_batch(
        self, person_id: str, images: list[str | bytes | BinaryIO], cropped: bool = False
    ) -> dict:
        """Enroll several photos of one person in one call (varied angles help)."""
        opened = [self._as_file(im) for im in images]
        try:
            r = requests.post(
                self._url("/v1/enroll/batch"),
                headers=self._headers,
                data={"person_id": person_id, "cropped": str(cropped).lower()},
                files=[("files", f) for f in opened],
                timeout=self.timeout,
            )
        finally:
            for f in opened:
                if hasattr(f, "close"):
                    f.close()
        return self._check(r)

    def enroll_body(self, person_id: str, image: str | bytes | BinaryIO, cropped: bool = False) -> dict:
        """Permanently enroll one body photo (front/side/back) for person_id."""
        f = self._as_file(image)
        try:
            r = requests.post(
                self._url("/v1/enroll/body"),
                headers=self._headers,
                data={"person_id": person_id, "cropped": str(cropped).lower()},
                files={"file": f},
                timeout=self.timeout,
            )
        finally:
            if hasattr(f, "close"):
                f.close()
        return self._check(r)

    def enroll_body_batch(
        self, person_id: str, images: list[str | bytes | BinaryIO], cropped: bool = False
    ) -> dict:
        """Permanently enroll several body photos of one person (multi-angle)."""
        opened = [self._as_file(im) for im in images]
        try:
            r = requests.post(
                self._url("/v1/enroll/body/batch"),
                headers=self._headers,
                data={"person_id": person_id, "cropped": str(cropped).lower()},
                files=[("files", f) for f in opened],
                timeout=self.timeout,
            )
        finally:
            for f in opened:
                if hasattr(f, "close"):
                    f.close()
        return self._check(r)

    def body_identities(self) -> dict:
        r = requests.get(self._url("/v1/body/identities"), headers=self._headers, timeout=self.timeout)
        return self._check(r)

    def delete_body_identity(self, person_id: str) -> dict:
        r = requests.delete(
            self._url(f"/v1/body/identities/{person_id}"), headers=self._headers, timeout=self.timeout
        )
        return self._check(r)

    def recognize(self, image: str | bytes | BinaryIO, cropped: bool = False) -> list[dict]:
        f = self._as_file(image)
        try:
            r = requests.post(
                self._url("/v1/recognize"),
                headers=self._headers,
                data={"cropped": str(cropped).lower()},
                files={"file": f},
                timeout=self.timeout,
            )
        finally:
            if hasattr(f, "close"):
                f.close()
        return self._check(r)["faces"]

    def identities(self) -> dict:
        r = requests.get(self._url("/v1/identities"), headers=self._headers, timeout=self.timeout)
        return self._check(r)

    def delete_identity(self, person_id: str) -> dict:
        r = requests.delete(
            self._url(f"/v1/identities/{person_id}"), headers=self._headers, timeout=self.timeout
        )
        return self._check(r)

    def save(self) -> dict:
        r = requests.post(self._url("/v1/index/save"), headers=self._headers, timeout=self.timeout)
        return self._check(r)

    def load(self) -> dict:
        r = requests.post(self._url("/v1/index/load"), headers=self._headers, timeout=self.timeout)
        return self._check(r)
