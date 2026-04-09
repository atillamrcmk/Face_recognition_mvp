# Yerel Yüz Tanıma MVP (kayıtlı kişiler)

## Bu proje nedir? (önemli)

Bu depo **müdür / ekip içi gösterim için tasarlanmış bir masaüstü demosudur**. Webcam görüntüsünde **önceden kayıtlı** kişiler (ör. Atilla, Serkan) ile benzerlik gösterir; **üretim ortamı güvenlik ürünü değildir** (erişim kontrolü, yasal biyometrik doğrulama veya yüksek güvenlik iddiası taşımaz).

Tamamen **yerel** çalışır: bulut API, veritabanı veya web paneli yoktur.

---

## Amaç

Webcam’den gelen görüntüde yüzleri bulur; `data/known_faces/<kişi>/` altındaki referanslarla karşılaştırır. Her yüz için en yüksek benzerlik hangi kayıtlı kimlikteyse ve eşik üstündeyse o **isim** gösterilir; hiçbiri eşik üstü değilse **Kayitli degil**. Tespit güveni düşükse **Kalite yetersiz** ile embedding karşılaştırması yapılmaz. Kutu, etiket ve benzerlik skoru çizilir.

---

## En iyi çalıştığı koşullar

- **Orta–iyi aydınlatma**, yüzde net gölgeler yok
- Kişi **kameraya dönük**, yüz **karede yeterince büyük** (çok uzak küçük yüz zayıf)
- **720p ve üzeri** webcam veya benzeri net akış
- Referans fotoğraflar ile canlı görüntü **benzer ortam** (ışık, mesafe)
- Arka planda yüzü kapatan hareketli nesne az

---

## Hata yapabileceği durumlar (dürüst liste)

- **Zayıf ışık**, **arka ışık**, aşırı pozlama
- **Yan profil** veya düşük tespit skoru → **Kalite yetersiz** veya yanlış sınıf
- **Sakal, gözlük, saç değişimi**, yaş / makyaj farkı (referansla uyumsuzluk)
- **Çok küçük veya bulanık yüz** (özellikle `FRAME_SCALE` düşükse)
- **Benzer görünümlü başka kişiler** (tek eşik ile ayırt sınırlıdır)
- **Üretim güvenliği** (sahtecilik, baskı foto, ekran saldırısı) — bu demo kapsamında ele alınmaz

---

## Kullanılan teknolojiler

| Bileşen | Rol |
|--------|-----|
| **Python 3.10+** | Ana dil |
| **OpenCV** | Kamera ve görüntüleme |
| **InsightFace** (`buffalo_l`) | Yüz tespiti + ArcFace benzeri embedding |
| **ONNX Runtime** | Model çıkarımı (CPU) |
| **NumPy** | Embedding ve benzerlik hesapları |

## Kurulum

### 1. Sanal ortam (önerilir)

```powershell
cd "face_recognition_mvp"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Bağımlılıklar

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

İlk çalıştırmada InsightFace model dosyaları `~/.insightface` altına indirilebilir; internet gerekir.

### 3. Referans görseller (çoklu kişi)

Her kayıtlı kişi için **ayrı klasör** kullanın:

```
data/known_faces/atilla/   → ör. atilla1.jpeg, atilla2.jpeg
data/known_faces/Serkan/   → ör. serkan1.jpeg, serkan2.jpeg
```

- Klasör adı serbest (`Serkan`, `serkan` vb.); ekranda görünen isim `app/config.py` içindeki `IDENTITY_DISPLAY_NAMES` ile eşlenir (küçük harf anahtar: `atilla`, `serkan`).
- Her klasörde en az **bir** geçerli yüz görüntüsü olmalı; yüzü algılanamayan dosyalar atlanır.
- Dosya **içeriği** değişince özet değişir; önbellek yeniden üretilir veya aşağıdaki ortam değişkeni ile zorlanır.
- `embeddings/` altında kişi başına önbellek dosyası oluşur (ör. `atilla_reference_embeddings.npz`, `serkan_reference_embeddings.npz`).

## Çalıştırma

Proje kökünden (`face_recognition_mvp`):

```powershell
python main.py
```

- Çıkış: Önce **video penceresine tıklayın** (odak orada olmalı), sonra **q** veya **ESC**. Alternatif: programı çalıştırdığınız **terminalde Ctrl+C** (bazen iki kez gerekebilir).
- Pencere başlığı ve parametreler `app/config.py` içindedir.

### Fotoğraf veya video ile çalıştırma

Webcam yerine dosyadan analiz için:

```powershell
# Tek fotoğraf
python main.py --image "C:\path\to\photo.jpg"

# Video dosyası
python main.py --video "C:\path\to\video.mp4"
```

İpucu: Video dosyasında performans düşükse `FRAME_SCALE` ve `PROCESS_EVERY_N_FRAMES` ayarlarını kullanın.

### Embedding önbelleğini zorla yeniden üretme

```powershell
$env:FACE_MVP_REBUILD_EMBEDDINGS="1"
python main.py
```

---

## Threshold (eşik) ayarlama rehberi

### `SIMILARITY_THRESHOLD` (cosine benzerliği)

- Her karede, **tüm kayıtlı kimlikler** için benzerlik hesaplanır; **en yüksek** skor bu eşikle kıyaslanır (tek kare değil; son birkaç karenin **medyan** yumuşatması kullanılır).
- **Düşük eşik** (ör. 0.35–0.40): Daha sık “birine benziyor” kabulü → **yanlış pozitif** riski artar.
- **Yüksek eşik** (ör. 0.48–0.55): Daha seçici → **yanlış negatif** riski artar (kayıtlı kişi reddedilebilir).

**Önerilen kalibrasyon:** Sabit ışıkta deneyin; eşiği **0.02 adımlarla** oynayıp hem kayıtlı hem kayıtsız yüzde gözlemleyin. Varsayılan **0.42** başlangıç değeridir.

### `MIN_DET_SCORE` (tespit güveni)

InsightFace’in `det_score` değeri bu altındaysa **embedding karşılaştırılmaz**; etiket **Kalite yetersiz** gösterilir. Çok düşük seçilirse bulanık yüzler de sınıflandırılır; çok yüksek seçilirse geçerli kareler sürekli reddedilir. Varsayılan **0.40** denge içindir.

### Zaman yumuşatma

- `TEMPORAL_WINDOW`: Kaç karenin benzerlik skorunun **medyanı** alınır (varsayılan 7).
- `TEMPORAL_MATCH_DIST_PX`: Ardışık karelerde aynı yüzü eşlemek için merkez mesafesi (küçültülmüş görüntü pikselinde).

---

## Demo sırasında öneriler

1. **Işığı** yüze verin; pencere arkasında güçlü ışık olmasın.
2. **Kameraya dönün**; mümkünse göz hizası.
3. **1–2 m** mesafede durun; yüz karede küçük kalmasın.
4. Gerekirse `FRAME_SCALE` değerini **1.0** yapın (daha yavaş ama küçük yüzlerde daha stabil).
5. Referans klasörüne **2–5 net fotoğraf** koyun (farklı ifade / hafif açı faydalı olabilir).
6. İlk çalıştırmadan önce başka uygulamanın **kamerayı kullanmadığından** emin olun.

## Performans ipuçları

- `FRAME_SCALE`: 1.0 tam çözünürlük; `0.5` hızlandırır, uzaktaki küçük yüzlerde doğruluk düşebilir.
- `PROCESS_EVERY_N_FRAMES`: `2` yaparsanız her iki karede bir analiz yapılır; ara karelerde son sonuç tekrar çizilir (FPS artabilir).

---

## Sınırlamalar (MVP)

- Tek hedef kimlik; çoklu kayıt, kayıt UI, veritabanı yoktur.
- Web kamerası kalitesi ve arka plan sonuçları etkiler.

---

## Gelecek geliştirmeler

- Çoklu kişi kaydı ve isim-etiket eşlemesi
- Kayıt / kalibrasyon ekranı
- SQLite / PostgreSQL ile kalıcı kayıt
- Yapılandırılabilir dosya logları
- REST API veya mesaj kuyruğu
- Canlı alarm (tanınmayan yüz)
- Video dosyası üzerinden analiz

Bu maddeler şu an **kapsam dışı**dır.
