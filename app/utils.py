"""Genel yardımcılar: vektör benzerliği, görüntü listeleme, dosya imzası."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Vektörü L2 birim vektörü yapar (cosine benzerliği için)."""
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return vec
    return (vec / n).astype(np.float32)


def max_cosine_similarity(query: np.ndarray, references: np.ndarray) -> float:
    """
    query: (d,)
    references: (n, d)
    Her referansla cosine benzerliğinin maksimumunu döndürür.
    """
    q = l2_normalize(query.reshape(-1))
    refs = references.astype(np.float64)
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    refs_u = refs / norms
    sims = refs_u @ q.astype(np.float64)
    return float(np.max(sims))


def list_identity_folders(known_faces_root: Path) -> list[Path]:
    """
    known_faces altında doğrudan bulunan, içinde en az bir referans görseli olan klasörler.
    Örnek: known_faces/atilla, known_faces/Serkan
    """
    if not known_faces_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(known_faces_root.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        if list_reference_images(p):
            out.append(p)
    return out


def list_reference_images(directory: Path) -> list[Path]:
    """Klasördeki desteklenen görüntü dosyalarını sıralı listeler."""
    from . import config

    if not directory.is_dir():
        return []
    exts = config.REFERENCE_IMAGE_EXTENSIONS
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def safe_imread(path: Path) -> np.ndarray | None:
    """
    Görüntüyü BGR olarak okur; hata olursa loglar ve None döner.
    Windows'ta Unicode yol (Türkçe karakter) ile cv2.imread sorunlu olduğundan
    np.fromfile + cv2.imdecode kullanılır.
    """
    import cv2

    try:
        p = Path(path)
        if not p.is_file():
            logger.warning("Dosya yok: %s", path)
            return None
        buf = np.fromfile(str(p), dtype=np.uint8)
        if buf.size == 0:
            logger.warning("Bos dosya: %s", path)
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Goruntu cozulemedi (format/bozuk?): %s", path)
            return None
        return img
    except Exception:
        logger.exception("Goruntu okuma hatasi: %s", path)
        return None


def file_sha256_hex(path: Path) -> str | None:
    """Dosya içeriğinin SHA256 özeti; okunamazsa None."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        logger.warning("Dosya okunamadi (ozet alinamadi): %s", path)
        return None
