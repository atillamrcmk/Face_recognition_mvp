"""
Dosya tabanlı girişler: resim veya video dosyası.

Amaç: Webcam akışındaki yüz karşılaştırma mantığını bozmadan,
tek bir resim veya video dosyası üzerinden de aynı overlay ile demo yapabilmek.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import safe_imread

logger = logging.getLogger(__name__)


@dataclass
class FrameSource:
    """Ortak arayüz: her çağrıda (ok, frame_bgr) döner."""

    name: str

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        return None


class ImageFileSource(FrameSource):
    def __init__(self, path: Path) -> None:
        super().__init__(name=str(path))
        self._path = Path(path)
        self._frame: np.ndarray | None = None
        self._served = False

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        if self._served:
            return False, np.array([])
        if self._frame is None:
            img = safe_imread(self._path)
            if img is None:
                return False, np.array([])
            self._frame = img
        self._served = True
        return True, self._frame.copy()


class VideoFileSource(FrameSource):
    def __init__(self, path: Path) -> None:
        super().__init__(name=str(path))
        self._path = Path(path)
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {self._path}")

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, np.array([])
        return True, frame

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            logger.exception("Video kapatma hatasi")

