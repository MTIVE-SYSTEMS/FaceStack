"""FaceStack — face recognition engine.

Public API:
    from facestack import FaceEngine, FaceIndex, Config
"""

from .config import Config
from .engine import DetectedFace, FaceEngine
from .index import FaceIndex, Match
from .recognizer import RecognizedFace, Recognizer

__all__ = [
    "Config",
    "FaceEngine",
    "DetectedFace",
    "FaceIndex",
    "Match",
    "Recognizer",
    "RecognizedFace",
]
__version__ = "0.1.0"
