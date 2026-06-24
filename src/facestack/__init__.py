"""FaceStack — face recognition engine.

Public API:
    from facestack import FaceEngine, FaceIndex, Config
"""

from .bodyengine import BodyEngine, DetectedBody  # light import; heavy only on construct
from .bodyindex import BodyIndex  # light: numpy + stdlib (hnswlib lazy)
from .config import Config
from .engine import DetectedFace, FaceEngine
from .index import FaceIndex, Match
from .linking import link_faces_to_bodies
from .recognizer import PersonResult, RecognizedBody, RecognizedFace, Recognizer

__all__ = [
    "Config",
    "FaceEngine",
    "DetectedFace",
    "FaceIndex",
    "Match",
    "Recognizer",
    "RecognizedFace",
    "RecognizedBody",
    "PersonResult",
    "BodyEngine",
    "DetectedBody",
    "BodyIndex",
    "link_faces_to_bodies",
]
__version__ = "0.1.0"
