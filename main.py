"""
Yerel webcam üzerinden kayıtlı kişileri tanıma — uygulama giriş noktası.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2

# Paket içe aktarımı: proje kökünü yola ekle (calisma dizininden bagimsiz)
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import config
from app.camera import Camera
from app.detector import create_face_app, extract_faces
from app.media import ImageFileSource, VideoFileSource
from app.recognizer import load_registered_identities
from app.temporal import TemporalScoreSmoother
from app.ui import draw_face_overlay, draw_fps, draw_hint_bar, draw_status_banner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yerel yuz tanima demo (webcam / image / video).")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--image", type=str, help="Tek bir resim dosyasi yolu (jpg/png).")
    g.add_argument("--video", type=str, help="Video dosyasi yolu (mp4/avi).")
    p.add_argument("--camera", type=int, default=None, help="Webcam index (varsayilan config).")
    return p.parse_args()


def run() -> None:
    args = _parse_args()
    force_rebuild = _env_flag("FACE_MVP_REBUILD_EMBEDDINGS")

    if not config.KNOWN_FACES_ROOT.is_dir():
        logger.error(
            "Referans kok klasoru yok: %s. Olusturun: data/known_faces/<isim>/",
            config.KNOWN_FACES_ROOT,
        )
        sys.exit(1)

    logger.info("InsightFace modeli yukleniyor (ilk seferde indirilebilir)...")
    app = create_face_app(
        model_name=config.INSIGHTFACE_MODEL_NAME,
        det_size=config.DET_SIZE,
        ctx_id=-1,
    )

    try:
        recognizer = load_registered_identities(
            app,
            config.KNOWN_FACES_ROOT,
            force_rebuild=force_rebuild,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        logger.error(
            "Cozum: data/known_faces/ altinda her kisi icin bir klasor acin "
            "(ornek: atilla, Serkan) ve icine en az bir yuz fotografi koyun."
        )
        sys.exit(2)
    except ValueError as e:
        logger.error("%s", e)
        logger.error(
            "Cozum: Gorsellerde net, onde ve tek basina gorunen bir yuz oldugundan emin olun; "
            "bozuk dosyalari kaldirin."
        )
        sys.exit(3)

    smoother = TemporalScoreSmoother(
        window=config.TEMPORAL_WINDOW,
        match_dist_px=config.TEMPORAL_MATCH_DIST_PX,
    )
    source_desc = "webcam"
    if args.image:
        source_desc = f"image: {args.image}"
    elif args.video:
        source_desc = f"video: {args.video}"
    logger.info(
        "Hazir. Kisiler: %s | Kaynak=%s | Benzerlik esigi=%.2f | Tespit esigi det>=%.2f | "
        "zaman yumusatma=%s kare.",
        ", ".join(recognizer.identity_names),
        source_desc,
        recognizer.threshold,
        config.MIN_DET_SCORE,
        config.TEMPORAL_WINDOW,
    )
    logger.info(
        "Cikis: (1) Video penceresine tiklayip q veya ESC "
        "(2) Terminalde Ctrl+C"
    )

    source = None
    cam = None
    try:
        if args.image:
            source = ImageFileSource(Path(args.image))
        elif args.video:
            source = VideoFileSource(Path(args.video))
        else:
            cam_index = config.CAMERA_INDEX if args.camera is None else int(args.camera)
            cam = Camera(index=cam_index)
            cam.open()
            source = cam
    except RuntimeError as e:
        logger.error("%s", e)
        if args.image or args.video:
            logger.error("Dosya acilamadi. Yolu ve dosya uzantisini kontrol edin.")
            sys.exit(5)
        logger.error(
            "Kamera kullanilamiyor. Baska program kamerali kapatmayi deneyin; "
            "veya --camera 0/1 deneyin (config CAMERA_INDEX degerini de degistirebilirsiniz)."
        )
        sys.exit(4)

    window = config.WINDOW_NAME
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_idx = 0
    last_draw: list[tuple[tuple[float, float, float, float], str, float]] = []
    fps_smooth = 0.0
    t_prev = time.perf_counter()

    try:
        while True:
            try:
                ok, frame = source.read_bgr()  # type: ignore[union-attr]
            except KeyboardInterrupt:
                logger.info("Ctrl+C ile cikis.")
                break
            if not ok:
                if args.image:
                    # Resimde tek kare gösterdik; kullanıcı tuşla kapatsın.
                    break
                logger.warning("Kare okunamadi, dongu sonlaniyor.")
                break

            frame_idx += 1
            h, w = frame.shape[:2]
            proc = frame
            scale_x = 1.0
            scale_y = 1.0
            if config.FRAME_SCALE < 0.999:
                proc_w = max(1, int(w * config.FRAME_SCALE))
                proc_h = max(1, int(h * config.FRAME_SCALE))
                proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                scale_x = w / proc.shape[1]
                scale_y = h / proc.shape[0]

            run_det = frame_idx % config.PROCESS_EVERY_N_FRAMES == 0
            if run_det:
                faces = extract_faces(app, proc)
                good_items: list[tuple[float, float, float]] = []
                face_metas: list[tuple[str, tuple[float, float, float, float], float | int, str]] = []

                for f in faces:
                    bx = f.bbox
                    bbox_full = (
                        float(bx[0] * scale_x),
                        float(bx[1] * scale_y),
                        float(bx[2] * scale_x),
                        float(bx[3] * scale_y),
                    )
                    cx = (float(bx[0]) + float(bx[2])) * 0.5
                    cy = (float(bx[1]) + float(bx[3])) * 0.5
                    if f.det_score < config.MIN_DET_SCORE:
                        face_metas.append(("low", bbox_full, float(f.det_score), ""))
                    else:
                        winner_name, raw_sim = recognizer.best_match_this_frame(f.embedding)
                        idx = len(good_items)
                        good_items.append((cx, cy, raw_sim))
                        face_metas.append(("good", bbox_full, idx, winner_name))

                smoothed = smoother.smooth_batch(good_items)

                last_draw = []
                for meta in face_metas:
                    kind = meta[0]
                    if kind == "low":
                        _, bbox_full, ds, _ = meta
                        last_draw.append((bbox_full, config.LOW_QUALITY_LABEL, ds))
                    else:
                        _, bbox_full, sidx, winner_name = meta
                        sm = smoothed[int(sidx)]
                        if sm < recognizer.threshold:
                            last_draw.append((bbox_full, config.UNKNOWN_LABEL, sm))
                        else:
                            last_draw.append((bbox_full, winner_name, sm))

            for bbox_full, label, score in last_draw:
                draw_face_overlay(frame, bbox_full, label, score)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 1e-6:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth <= 0 else fps_smooth * 0.85 + inst_fps * 0.15
            draw_fps(frame, fps_smooth)
            draw_status_banner(frame, config.DEMO_STATUS_TEXT, position="bottom")
            draw_hint_bar(frame)

            cv2.imshow(window, frame)
            try:
                wait = 0 if args.image else 1
                key = cv2.waitKey(wait) & 0xFF
            except KeyboardInterrupt:
                logger.info("Ctrl+C ile cikis.")
                break
            if key == ord("q") or key == ord("Q") or key == 27:
                logger.info("Kullanici cikisi (q veya ESC).")
                break
    except KeyboardInterrupt:
        logger.info("Ctrl+C ile cikis.")
    finally:
        try:
            if source is not None:
                source.release()  # type: ignore[union-attr]
        finally:
            if cam is not None:
                cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
