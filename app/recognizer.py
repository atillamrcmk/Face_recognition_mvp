"""Referans embeddingleri: üretim, önbellek, benzerlik ve sınıflandırma."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .utils import (
    file_sha256_hex,
    l2_normalize,
    list_reference_images,
    max_cosine_similarity,
    safe_imread,
)

logger = logging.getLogger(__name__)


class MultiPersonRecognizer:
    """Birden fazla kayıtlı kimlik; her biri için referans embeddinglerle cosine benzerliği."""

    def __init__(self, identities: list[tuple[str, np.ndarray]], threshold: float) -> None:
        """
        identities: (gorunen_isim, referans_matrisi (N, D)) listesi.
        """
        self._identities: list[tuple[str, np.ndarray]] = []
        for name, mat in identities:
            if mat.size == 0:
                logger.warning("Bos referans atlandi: %s", name)
                continue
            rows = np.stack([l2_normalize(mat[i]) for i in range(len(mat))])
            self._identities.append((name, rows))
        if not self._identities:
            raise ValueError("Kayitli kimlik matrisi bos.")
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def identity_names(self) -> list[str]:
        return [n for n, _ in self._identities]

    def similarities_per_identity(self, query_embedding: np.ndarray) -> dict[str, float]:
        """Her kayıtlı kimlik için maksimum cosine benzerliği."""
        q = l2_normalize(query_embedding.reshape(-1))
        out: dict[str, float] = {}
        for name, refs in self._identities:
            out[name] = max_cosine_similarity(q, refs)
        return out

    def best_match_this_frame(self, query_embedding: np.ndarray) -> tuple[str, float]:
        """Bu karede en çok benzeyen kimlik ve o skor (karar için ham değer)."""
        per = self.similarities_per_identity(query_embedding)
        best_name = max(per, key=per.get)
        return best_name, per[best_name]


def cache_path_for_identity(embeddings_dir: Path, folder_name: str) -> Path:
    """Önbellek dosya adı: klasör adından güvenli slug (Serkan -> serkan)."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in folder_name.lower()).strip("_")
    if not slug:
        slug = "identity"
    return embeddings_dir / f"{slug}_reference_embeddings.npz"


def folder_display_name(folder: Path) -> str:
    """Klasör adına göre ekranda gösterilecek isim."""
    key = folder.name.lower()
    return config.IDENTITY_DISPLAY_NAMES.get(key, folder.name)


def load_registered_identities(
    app: Any,
    known_faces_root: Path,
    force_rebuild: bool,
) -> MultiPersonRecognizer:
    """
    data/known_faces/<kimlik>/ alt klasörlerini tarar; her biri için embedding yükler veya üretir.
    """
    from .utils import list_identity_folders

    folders = list_identity_folders(known_faces_root)
    if not folders:
        raise FileNotFoundError(
            f"Kayitli kimlik yok: {known_faces_root}. "
            "Ornek: data/known_faces/atilla/ ve data/known_faces/Serkan/ altina fotograf ekleyin."
        )

    identities: list[tuple[str, np.ndarray]] = []
    for folder in folders:
        display = folder_display_name(folder)
        cache_path = cache_path_for_identity(config.EMBEDDINGS_DIR, folder.name)
        logger.info(
            "Kimlik referansi: klasor='%s' -> ekran='%s', onbellek=%s",
            folder.name,
            display,
            cache_path.name,
        )
        matrix = load_or_build_reference_matrix(
            app,
            folder,
            cache_path,
            force_rebuild=force_rebuild,
        )
        identities.append((display, matrix))

    logger.info(
        "Kayitli kimlikler hazir (%s kisi): %s",
        len(identities),
        ", ".join(n for n, _ in identities),
    )
    return MultiPersonRecognizer(identities, threshold=config.SIMILARITY_THRESHOLD)


def _pick_largest_face(faces: list[Any]) -> Any | None:
    if not faces:
        return None
    return max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )


def _reference_signature(folder: Path) -> tuple[list[str], list[str]]:
    """Dosya yolu + içerik SHA256 ile önbellek anahtarı (aynı isim, farklı içerik yakalanır)."""
    paths = list_reference_images(folder)
    path_strs: list[str] = []
    hashes: list[str] = []
    for p in paths:
        digest = file_sha256_hex(p)
        path_strs.append(str(p.resolve()))
        hashes.append(digest if digest is not None else "__unreadable__")
        if digest is None:
            logger.warning("Imza: dosya okunamadi, icerik ozeti yerine plase tutuldu: %s", p)
    return path_strs, hashes


def build_reference_matrix_from_folder(app: Any, folder: Path) -> tuple[np.ndarray, list[str]]:
    """
    Klasördeki görsellerden yüz embeddingleri üretir (yüz yoksa atlar).
    Hiç geçerli embedding yoksa FileNotFoundError / ValueError benzeri net hata.
    Dönüş: (matris, her satırın kaynak dosya yolu).
    """
    paths = list_reference_images(folder)
    if not paths:
        raise FileNotFoundError(
            f"Referans goruntu yok: {folder}. JPG/PNG dosyalari ekleyin (ornek: 01.jpg)."
        )

    rows: list[np.ndarray] = []
    source_paths: list[str] = []
    for path in paths:
        img = safe_imread(path)
        if img is None:
            continue
        try:
            faces = app.get(img)
        except Exception:
            logger.exception("Referans islenemedi: %s", path)
            continue
        face = _pick_largest_face(faces)
        if face is None:
            logger.warning("Bu goruntude yuz yok, atlaniyor: %s", path)
            continue
        emb = getattr(face, "embedding", None)
        if emb is None:
            logger.warning("Embedding yok: %s", path)
            continue
        rows.append(l2_normalize(np.array(emb, dtype=np.float32).reshape(-1)))
        source_paths.append(str(path.resolve()))

    if not rows:
        raise ValueError(
            "Hic gecerli referans embedding uretilemedi. "
            "Net yuz fotograflari koyun veya bozuk dosyalari cikartin."
        )
    return np.stack(rows, axis=0), source_paths


def load_or_build_reference_matrix(
    app: Any,
    known_folder: Path,
    cache_path: Path,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Önbellek varsa ve dosya yolu + içerik özeti uyuyorsa diskten yükler.
    Aksi halde klasörden üretir ve kaydeder.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cur_paths, cur_hashes = _reference_signature(known_folder)

    if not cur_paths:
        raise FileNotFoundError(
            f"Referans goruntu yok: {known_folder}. JPG/PNG dosyalari ekleyin (ornek: 01.jpg)."
        )

    if not force_rebuild and cache_path.is_file():
        try:
            data = np.load(cache_path, allow_pickle=True)
            emb = data["embeddings"]
            saved_paths = data.get("paths")
            saved_hashes = data.get("file_hashes")
            if saved_hashes is None:
                logger.info(
                    "Embedding onbellegi eski formatta (icerik ozeti yok). "
                    "Yeniden hesaplama yapiliyor; %s",
                    cache_path,
                )
            elif saved_paths is not None:
                sp = list(saved_paths.tolist()) if hasattr(saved_paths, "tolist") else list(saved_paths)
                sh = list(saved_hashes.tolist()) if hasattr(saved_hashes, "tolist") else list(saved_hashes)
                spr = data.get("path_per_row")
                if sp == cur_paths and sh == cur_hashes:
                    mat = np.array(emb, dtype=np.float32)
                    if spr is not None:
                        ppr = list(spr.tolist()) if hasattr(spr, "tolist") else list(spr)
                        if len(ppr) != mat.shape[0]:
                            logger.info(
                                "Onbellekte satir/kaynak uyumsuzlugu; yeniden hesaplaniyor."
                            )
                        else:
                            nvec = int(mat.shape[0])
                            logger.info(
                                "Embedding onbellegi kullaniliyor: %s (%s referans vektoru, "
                                "dosya listesi ve SHA256 ile dogrulandi).",
                                cache_path,
                                nvec,
                            )
                            return mat
                    else:
                        nvec = int(mat.shape[0])
                        logger.info(
                            "Embedding onbellegi kullaniliyor: %s (%s referans vektoru; "
                            "eski onbellek, path_per_row yok).",
                            cache_path,
                            nvec,
                        )
                        return mat
                logger.info(
                    "Onbellek guncel degil (dosya eklendi/silindi veya icerik degisti). "
                    "Yeniden hesaplaniyor."
                )
            else:
                logger.info("Onbellek dosyasi eksik alan iceriyor; yeniden hesaplaniyor.")
        except Exception:
            logger.warning("Onbellek okunamadi, yeniden uretilecek.", exc_info=True)

    logger.info(
        "Referans embeddingler hesaplaniyor: %s dosya, %s klasoru.",
        len(cur_paths),
        known_folder,
    )
    matrix, path_per_row = build_reference_matrix_from_folder(app, known_folder)
    n_out = int(matrix.shape[0])
    logger.info(
        "Hesaplama bitti: %s gecerli referans vektoru uretildi (yuz bulunamayan dosyalar atlanmis olabilir).",
        n_out,
    )
    try:
        np.savez_compressed(
            cache_path,
            embeddings=matrix.astype(np.float32),
            paths=np.array(cur_paths, dtype=object),
            file_hashes=np.array(cur_hashes, dtype=object),
            path_per_row=np.array(path_per_row, dtype=object),
        )
        logger.info(
            "Embeddingler kaydedildi: %s (yol + SHA256 imzasi ile).",
            cache_path,
        )
    except Exception:
        logger.exception("Embedding onbellek yazilamadi (calisma devam ediyor).")
    return matrix
