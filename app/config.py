"""Uygulama sabitleri ve yol ayarları."""

from pathlib import Path

# Proje kökü: face_recognition_mvp/
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Her kişi için alt klasör: data/known_faces/atilla/, data/known_faces/serkan/ vb.
KNOWN_FACES_ROOT: Path = PROJECT_ROOT / "data" / "known_faces"
EMBEDDINGS_DIR: Path = PROJECT_ROOT / "embeddings"

# Cosine benzerliği (L2-normalize edilmiş vektörlerde dot product).
# Bu eşik, yanlış pozitif / yanlış negatif dengesini belirler; ortam ve kamera için ayarlanmalıdır.
# Çok düşük: yanlış kişi eşleşebilir. Çok yüksek: doğru kişi reddedilebilir.
# Varsayılan 0.42: başlangıç değeri; odaya göre kalibre edin.
SIMILARITY_THRESHOLD: float = 0.42

# Tespit güveni (InsightFace det_score, tipik 0–1). Altında embedding karşılaştırması yapılmaz.
MIN_DET_SCORE: float = 0.40

CAMERA_INDEX: int = 0
WINDOW_NAME: str = "Yuz Tanima — Kayitli kisiler"

# Demo penceresinde üst/alt bantta kısa açıklama
DEMO_STATUS_TEXT: str = "Kayitli kisi dogrulama demosu"

# 1.0 = tam çözünürlük (daha yavaş), <1 daha hızlı (algı kalitesi düşebilir)
FRAME_SCALE: float = 0.5

# Her N karede bir yüz analizi (1 = her kare). >1 ise ara karelerde son sonuç tekrar çizilir.
PROCESS_EVERY_N_FRAMES: int = 1

# InsightFace tespit girdisi (büyük = küçük yüzlerde daha iyi, daha yavaş)
DET_SIZE: tuple[int, int] = (640, 640)

# Kayıtlı kimliklerden hiçbirine benzemiyorsa (eşik altı)
UNKNOWN_LABEL: str = "Kayitli degil"
LOW_QUALITY_LABEL: str = "Kalite yetersiz"

# Klasör adı (küçük harf) -> ekranda gösterilecek isim (ornek: Serkan, atilla)
IDENTITY_DISPLAY_NAMES: dict[str, str] = {
    "atilla": "Atilla",
    "serkan": "Serkan",
}

# Geçici yumuşatma: son birkaç karedeki en yüksek benzerlik skorunun medyanı (tek frame değil).
TEMPORAL_WINDOW: int = 7
TEMPORAL_MATCH_DIST_PX: float = 120.0

INSIGHTFACE_MODEL_NAME: str = "buffalo_l"

REFERENCE_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
