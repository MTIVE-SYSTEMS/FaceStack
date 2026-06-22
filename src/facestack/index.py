"""FaceIndex — the gallery of saved faces and the 1:N matcher.

This is the heart of the product: recognising *saved* faces. We keep an
in-memory hnswlib cosine index of enrolled ArcFace embeddings (<10K identities,
possibly several embeddings per person) and answer "who is this?" queries.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("facestack.index")


@dataclass(slots=True)
class Match:
    person_id: str
    similarity: float  # cosine similarity in [-1, 1]; higher = closer
    matched: bool  # similarity >= threshold


class FaceIndex:
    """Cosine-similarity gallery over L2-normalized embeddings.

    Multiple embeddings can map to the same person_id (recommended: enroll a
    few shots per person). A query returns the best-matching person.
    """

    def __init__(self, dim: int = 512, capacity: int = 10_000, threshold: float = 0.40):
        import hnswlib

        self.dim = dim
        self.threshold = threshold
        self._lock = threading.RLock()
        self._index = hnswlib.Index(space="cosine", dim=dim)
        self._index.init_index(max_elements=max(capacity, 16), ef_construction=200, M=16)
        self._index.set_ef(64)
        self._next_label = 0
        self._label_to_person: dict[int, str] = {}
        self._person_labels: dict[str, list[int]] = {}

    # --- enrollment ---
    def add(self, person_id: str, embedding: np.ndarray) -> int:
        """Add one embedding for person_id. Returns its internal label."""
        vec = self._prep(embedding)
        with self._lock:
            label = self._next_label
            self._next_label += 1
            if label >= self._index.get_max_elements():
                self._index.resize_index(self._index.get_max_elements() * 2)
            self._index.add_items(vec.reshape(1, -1), np.array([label]))
            self._label_to_person[label] = person_id
            self._person_labels.setdefault(person_id, []).append(label)
            return label

    def add_many(self, person_id: str, embeddings: list[np.ndarray]) -> list[int]:
        return [self.add(person_id, e) for e in embeddings]

    # --- recognition ---
    def search(self, embedding: np.ndarray, k: int = 1) -> list[Match]:
        """Return up to k best person matches for an embedding."""
        with self._lock:
            if self._next_label == 0:
                return []
            vec = self._prep(embedding)
            k = min(k, self._next_label)
            labels, distances = self._index.knn_query(vec.reshape(1, -1), k=k)

        # hnswlib cosine "distance" = 1 - cosine_similarity
        out: list[Match] = []
        seen: set[str] = set()
        for label, dist in zip(labels[0], distances[0]):
            person = self._label_to_person.get(int(label))
            if person is None or person in seen:
                continue
            seen.add(person)
            sim = float(1.0 - dist)
            out.append(Match(person_id=person, similarity=sim, matched=sim >= self.threshold))
        return out

    def recognize(self, embedding: np.ndarray) -> Match | None:
        """Best single match, or None if the gallery is empty."""
        matches = self.search(embedding, k=1)
        return matches[0] if matches else None

    # --- management ---
    def remove_person(self, person_id: str) -> int:
        """Mark all of a person's embeddings deleted. Returns count removed."""
        with self._lock:
            labels = self._person_labels.pop(person_id, [])
            for label in labels:
                try:
                    self._index.mark_deleted(label)
                except RuntimeError:
                    pass
                self._label_to_person.pop(label, None)
            return len(labels)

    @property
    def people(self) -> list[str]:
        with self._lock:
            return sorted(self._person_labels.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._label_to_person)

    # --- persistence ---
    def save(self, index_path: str, meta_path: str) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            self._index.save_index(index_path)
            meta = {
                "dim": self.dim,
                "threshold": self.threshold,
                "next_label": self._next_label,
                "label_to_person": self._label_to_person,
                "max_elements": self._index.get_max_elements(),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        log.info("Saved index: %d embeddings -> %s", len(self), index_path)

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "FaceIndex":
        import hnswlib

        with open(meta_path) as f:
            meta = json.load(f)
        self = cls.__new__(cls)
        self.dim = meta["dim"]
        self.threshold = meta["threshold"]
        self._lock = threading.RLock()
        self._index = hnswlib.Index(space="cosine", dim=self.dim)
        self._index.load_index(index_path, max_elements=meta["max_elements"])
        self._index.set_ef(64)
        self._next_label = meta["next_label"]
        self._label_to_person = {int(k): v for k, v in meta["label_to_person"].items()}
        self._person_labels = {}
        for label, person in self._label_to_person.items():
            self._person_labels.setdefault(person, []).append(label)
        log.info("Loaded index: %d embeddings from %s", len(self), index_path)
        return self

    # --- internals ---
    def _prep(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        if vec.shape[0] != self.dim:
            raise ValueError(f"Embedding dim {vec.shape[0]} != index dim {self.dim}")
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
