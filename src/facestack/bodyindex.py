"""BodyIndex — TTL-aware cosine gallery of body (ReID) embeddings.

Mirrors FaceIndex, but body appearance is clothing-based and only valid for a
day or so across cameras, so every embedding carries a unix timestamp and
queries ignore anything older than `now - ttl`. Time is always supplied by the
caller (never time.time() here) so TTL behaviour is deterministic and testable.
"""

from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np

from facestack.index import Match  # reuse the face gallery's match dataclass

log = logging.getLogger("facestack.bodyindex")

# Over-fetch so TTL/dedupe pruning after the k-NN query never starves results.
# We fetch the larger of (k * OVERFETCH) and (k + EXTRA), clamped to gallery size.
_OVERFETCH = 10
_OVERFETCH_EXTRA = 32


class BodyIndex:
    """Cosine gallery over L2-normalized body embeddings, with per-entry TTL.

    Same hnswlib mechanics as FaceIndex plus a parallel label->timestamp map.
    A single person_id may have many embeddings (auto-enrolled across frames /
    cameras); search returns the best *unexpired* embedding per person.
    """

    def __init__(self, dim: int = 512, capacity: int = 10_000, threshold: float = 0.50):
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
        self._label_to_ts: dict[int, float] = {}  # NEW: enrollment time per label

    # --- enrollment ---
    def add(self, person_id: str, embedding: np.ndarray, ts: float) -> int:
        """Add one body embedding for person_id, stamped at unix time `ts`."""
        vec = self._prep(embedding)
        with self._lock:
            label = self._next_label
            self._next_label += 1
            if label >= self._index.get_max_elements():
                self._index.resize_index(self._index.get_max_elements() * 2)
            self._index.add_items(vec.reshape(1, -1), np.array([label]))
            self._label_to_person[label] = person_id
            self._person_labels.setdefault(person_id, []).append(label)
            self._label_to_ts[label] = float(ts)
            return label

    # --- recognition ---
    def search(self, embedding: np.ndarray, now: float, ttl: float, k: int = 1) -> list[Match]:
        """Up to k best person matches, ignoring embeddings older than now-ttl.

        Over-fetches from hnswlib, then filters expired entries, dedupes by
        person (keeping the nearest surviving embedding), and truncates to k.
        """
        cutoff = now - ttl
        with self._lock:
            # Clamp to the number of LIVE (non-deleted) elements, not _next_label:
            # purge()/remove_person() mark_delete entries but leave _next_label
            # untouched, and hnswlib raises if asked for more neighbours than it
            # can return contiguously. len(_label_to_person) is the live count.
            live = len(self._label_to_person)
            if live == 0:
                return []
            vec = self._prep(embedding)
            fetch = min(live, max(k * _OVERFETCH, k + _OVERFETCH_EXTRA))
            labels, distances = self._index.knn_query(vec.reshape(1, -1), k=fetch)
            # Snapshot the maps we read under the lock.
            label_to_person = self._label_to_person
            label_to_ts = self._label_to_ts

            out: list[Match] = []
            seen: set[str] = set()
            for label, dist in zip(labels[0], distances[0]):
                label = int(label)
                person = label_to_person.get(label)
                if person is None or person in seen:
                    continue
                ts = label_to_ts.get(label)
                if ts is None or ts < cutoff:  # expired or orphaned -> skip
                    continue
                seen.add(person)
                sim = float(1.0 - dist)  # hnswlib cosine distance = 1 - cos_sim
                out.append(Match(person_id=person, similarity=sim, matched=sim >= self.threshold))
                if len(out) >= k:
                    break
            return out

    def recognize(self, embedding: np.ndarray, now: float, ttl: float) -> Match | None:
        """Best single unexpired match, or None if nothing valid is found."""
        matches = self.search(embedding, now=now, ttl=ttl, k=1)
        return matches[0] if matches else None

    # --- management ---
    def purge(self, now: float, ttl: float) -> int:
        """mark_deleted every embedding older than now-ttl. Returns count purged."""
        cutoff = now - ttl
        with self._lock:
            expired = [lbl for lbl, ts in self._label_to_ts.items() if ts < cutoff]
            for label in expired:
                try:
                    self._index.mark_deleted(label)
                except RuntimeError:
                    pass  # already deleted
                person = self._label_to_person.pop(label, None)
                self._label_to_ts.pop(label, None)
                if person is not None:
                    labels = self._person_labels.get(person)
                    if labels is not None:
                        labels = [l for l in labels if l != label]
                        if labels:
                            self._person_labels[person] = labels
                        else:
                            self._person_labels.pop(person, None)
            if expired:
                log.info("Purged %d expired body embeddings", len(expired))
            return len(expired)

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
                self._label_to_ts.pop(label, None)
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
        """Persist the hnswlib index + meta (incl. timestamps). No implicit purge."""
        with self._lock:
            os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
            self._index.save_index(index_path)
            meta = {
                "version": 1,
                "dim": self.dim,
                "threshold": self.threshold,
                "next_label": self._next_label,
                "max_elements": self._index.get_max_elements(),
                "label_to_person": self._label_to_person,
                "label_to_ts": self._label_to_ts,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        log.info("Saved body index: %d embeddings -> %s", len(self), index_path)

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "BodyIndex":
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
        self._label_to_ts = {int(k): float(v) for k, v in meta["label_to_ts"].items()}
        self._person_labels = {}
        for label, person in self._label_to_person.items():
            self._person_labels.setdefault(person, []).append(label)
        log.info("Loaded body index: %d embeddings from %s", len(self), index_path)
        return self

    # --- internals ---
    def _prep(self, embedding: np.ndarray) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).flatten()
        if vec.shape[0] != self.dim:
            raise ValueError(f"Embedding dim {vec.shape[0]} != index dim {self.dim}")
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
