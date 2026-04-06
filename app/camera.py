"""Webcam yakalama."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """OpenCV VideoCapture sarmalayıcı."""

    def __init__(self, index: int = 0) -> None:
        self._index = index
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Kamera acilamadi (index={self._index}). Baska bir CAMERA_INDEX deneyin."
            )
        logger.info("Kamera acildi: index=%s", self._index)

    def read_bgr(self) -> tuple[bool, np.ndarray]:
        """Bir kare okur (BGR). Kapalıysa (False, bos dizi)."""
        if self._cap is None:
            return False, np.array([])
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, np.array([])
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Kamera kapatildi.")

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
