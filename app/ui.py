"""OpenCV ile kutu ve metin çizimi."""

from __future__ import annotations

import cv2
import numpy as np

from . import config


def _label_kind(label: str) -> str:
    if label == config.LOW_QUALITY_LABEL:
        return "low"
    if label == config.UNKNOWN_LABEL:
        return "nomatch"
    return "match"


def draw_face_overlay(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    label: str,
    score: float,
) -> None:
    """Tek yüz için dikdörtgen ve etiket (yerinde çizer)."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    kind = _label_kind(label)
    if kind == "match":
        color = (0, 220, 0)
    elif kind == "low":
        color = (0, 240, 255)
    else:
        color = (0, 100, 255)

    thickness = 3
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, thickness)

    if kind == "low":
        score_txt = f"tespit {score:.2f}"
    else:
        score_txt = f"benzerlik {score:.2f}"

    text = f"{label}  |  {score_txt}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    ft = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, ft)
    ty = max(y1 - 10, th + 14)
    pad = 6
    cv2.rectangle(
        frame_bgr,
        (x1, ty - th - pad),
        (x1 + tw + 2 * pad, ty + baseline + 2),
        (25, 25, 25),
        -1,
    )
    cv2.rectangle(
        frame_bgr,
        (x1, ty - th - pad),
        (x1 + tw + 2 * pad, ty + baseline + 2),
        color,
        1,
    )
    cv2.putText(
        frame_bgr,
        text,
        (x1 + pad, ty - 4),
        font,
        scale,
        (248, 248, 248),
        ft,
        cv2.LINE_AA,
    )


def draw_fps(frame_bgr: np.ndarray, fps: float) -> None:
    """Sağ üstte FPS."""
    t = f"FPS: {fps:.1f}"
    x = frame_bgr.shape[1] - 160
    cv2.putText(
        frame_bgr,
        t,
        (max(10, x), 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )


def draw_status_banner(frame_bgr: np.ndarray, text: str, position: str = "bottom") -> None:
    """Demo durum metni (üst veya alt şerit)."""
    h, w = frame_bgr.shape[:2]
    bar_h = 36
    if position == "top":
        y0, y1 = 0, bar_h
        ty = 26
    else:
        y0, y1 = h - bar_h, h
        ty = h - 12
    roi = frame_bgr[y0:y1, :]
    tint = np.full(roi.shape, (35, 35, 48), dtype=np.uint8)
    cv2.addWeighted(tint, 0.42, roi, 0.58, 0, roi)
    cv2.putText(
        frame_bgr,
        text,
        (16, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 235),
        2,
        cv2.LINE_AA,
    )


def draw_hint_bar(frame_bgr: np.ndarray) -> None:
    """Sol altta çıkış ipucu (durum şeridinin üstünde küçük)."""
    msg = "Cikis: pencereye tikla -> q / ESC | veya terminalde Ctrl+C"
    y = max(24, frame_bgr.shape[0] - 48)
    cv2.putText(
        frame_bgr,
        msg,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 170),
        2,
        cv2.LINE_AA,
    )
