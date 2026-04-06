"""
Yüz tespiti ve embedding çıkarımı.

InsightFace'te tek bir ileri geçiş (app.get) hem kutuları hem embeddingleri üretir;
bu yüzden bu modül hem "tespit" hem o kareden embedding okuma sorumluluğunu üstlenir.
Ayrı dosyalarda tutmak, ileride sadece tespit veya sadece embedding için
model değiştirmeyi kolaylaştırır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Tek yüz: kutu, güven skoru ve tanıma embeddingi."""

    bbox: np.ndarray  # [x1, y1, x2, y2] float32
    embedding: np.ndarray  # (dim,) float32
    det_score: float


def extract_faces(app: Any, bgr_image: np.ndarray) -> list[DetectedFace]:
    """
    BGR görüntüde yüzleri bulur ve her biri için embedding döndürür.
    Boş veya geçersiz görüntüde [] döner; çökmez.
    """
    if bgr_image is None or bgr_image.size == 0:
        return []
    try:
        raw_faces = app.get(bgr_image)
    except Exception:
        logger.exception("InsightFace frame isleme hatasi")
        return []

    out: list[DetectedFace] = []
    for f in raw_faces:
        emb = getattr(f, "embedding", None)
        if emb is None:
            continue
        try:
            bbox = np.array(f.bbox, dtype=np.float32).reshape(4)
            vec = np.array(emb, dtype=np.float32).reshape(-1)
            score = float(getattr(f, "det_score", 0.0))
        except Exception:
            logger.warning("Yuz alanini parse edemedi, atlaniyor.")
            continue
        out.append(DetectedFace(bbox=bbox, embedding=vec, det_score=score))
    return out


def create_face_app(
    model_name: str,
    det_size: tuple[int, int],
    ctx_id: int = -1,
) -> "FaceAnalysis":
    """InsightFace FaceAnalysis örneği oluşturur (CPU: ctx_id=-1)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=model_name,
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app
