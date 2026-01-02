# 🎾 Predictive Serve

**Predictive Serve**, tenis maçlarının sonucunu tahmin etmek için tasarlanmış uçtan uca (end-to-end) bir **Python projesidir**.

Proje:

- Geçmiş tenis maçlarını **tennis-data.co.uk** sitesinden otomatik indirir,
- Bu maçları temizleyip zengin **feature**'lar üretir (Elo, form, dinlenme süresi, head-to-head vs.),
- **Logistic Regression** modeli ile "A oyuncusu kazanır mı?" olasılığını tahmin eder,
- Model tahminlerini **bahis şirketi oranlarından türetilen olasılıklarla** kıyaslar.

Bu repo özellikle **ders projesi / akademik kullanım** için tasarlanmıştır:
Tüm adımlar komut satırından çalıştırılabilir ve faz faz (FAZ 0–3) net şekilde ayrılmıştır.

---

## 🔍 Problem Tanımı

Amaç:

> "Bahis şirketlerinin oranlarına ek olarak oyuncuların formu, Elo rating'i, head-to-head geçmişi ve diğer istatistikleri kullanarak tenis maç sonucu tahmininde ne kadar iyi olabiliriz?"

Bu kapsamda:

- Geçmiş maç verisi → tennis-data.co.uk
- Hedef değişken → `y` (1 = playerA kazandı, 0 = playerB kazandı)
- Model çıktısı → `P(playerA kazanır)`

---

## 📚 Kullanılan Kütüphaneler ve Nedenleri

### Veri İşleme

- **`pandas`**: Veri manipülasyonu ve analizi için temel kütüphane
  - **Kullanım yerleri**: Tüm veri işleme modüllerinde (`src/data/*`, `src/features/*`)
  - **Neden**: CSV okuma/yazma, DataFrame işlemleri, veri temizleme, feature engineering
  - **Örnek**: `pd.read_csv()`, `df.groupby()`, `df.merge()`

- **`numpy`**: Sayısal hesaplamalar ve array işlemleri
  - **Kullanım yerleri**: Feature engineering (`src/features/*`), model eğitimi (`src/models/*`)
  - **Neden**: Hızlı matematiksel işlemler, array operasyonları, NaN handling
  - **Örnek**: `np.nan`, `np.array()`, matematiksel hesaplamalar

### Makine Öğrenmesi

- **`scikit-learn`**: ML model eğitimi ve değerlendirme
  - **Kullanım yerleri**: `src/models/train_logreg.py`, `src/analysis/metrics.py`
  - **Neden**: 
    - `LogisticRegression`: Basit, yorumlanabilir, overfit riski düşük
    - `SimpleImputer`: Eksik değer doldurma (median strategy)
    - `log_loss`, `brier_score_loss`, `accuracy_score`: Model performans metrikleri
  - **Örnek**: Model eğitimi, validation metrikleri

- **`joblib`**: Model serialization (kaydetme/yükleme)
  - **Kullanım yerleri**: `src/models/train_logreg.py`, `src/models/score_all_matches.py`, `streamlit_app.py`
  - **Neden**: Büyük numpy array'leri ve sklearn modellerini verimli kaydetme/yükleme
  - **Örnek**: `joblib.dump(model, "model.pkl")`, `joblib.load("model.pkl")`

### Veri Toplama

- **`requests`**: HTTP istekleri
  - **Kullanım yerleri**: `src/data/fetch_data.py`
  - **Neden**: tennis-data.co.uk'ten Excel dosyalarını indirmek için
  - **Örnek**: `requests.get(url)` ile Excel dosyası indirme

- **`openpyxl`**: Excel dosyalarını okuma
  - **Kullanım yerleri**: `src/data/fetch_data.py` (pandas ile birlikte)
  - **Neden**: `.xlsx` formatındaki verileri okumak için pandas'a gerekli
  - **Örnek**: `pd.read_excel()` fonksiyonu bu kütüphaneyi kullanır

### Web Arayüzü

- **`streamlit`**: İnteraktif web uygulaması
  - **Kullanım yerleri**: `streamlit_app.py`
  - **Neden**: Hızlı prototipleme, minimal kod ile interaktif dashboard
  - **Özellikler**: Filtreleme, grafikler, interaktif widget'lar

### Görselleştirme (Sunum İçin)

- **`matplotlib`**: Grafik ve görselleştirme kütüphanesi
  - **Kullanım yerleri**: Sunum görselleri oluşturma (`create_presentation_visuals.py`)
  - **Neden**: Profesyonel grafikler, diyagramlar ve görselleştirmeler
  - **Örnek**: Pipeline diyagramları, metrik grafikleri, confusion matrix

- **`seaborn`**: İstatistiksel veri görselleştirme
  - **Kullanım yerleri**: Sunum görselleri oluşturma (`create_presentation_visuals.py`)
  - **Neden**: Matplotlib üzerine kurulu, daha estetik ve kolay kullanım
  - **Örnek**: Heatmap'ler, istatistiksel grafikler

**Not**: `matplotlib` ve `seaborn` sadece sunum görselleri oluşturmak için kullanılmıştır. Ana uygulama bu kütüphanelere bağımlı değildir.

### Standart Kütüphaneler

- **`pathlib`**: Dosya yolu işlemleri
  - **Kullanım yerleri**: Tüm modüllerde
  - **Neden**: Cross-platform dosya yolu yönetimi, daha temiz kod
  - **Örnek**: `Path("data/raw/allyears.csv")`

- **`typing`**: Tip hint'leri
  - **Kullanım yerleri**: Tüm modüllerde
  - **Neden**: Kod okunabilirliği, IDE desteği, tip güvenliği

- **`collections`**: Özel veri yapıları
  - **Kullanım yerleri**: `src/features/elo.py` (defaultdict), `src/features/form.py` (deque)
  - **Neden**: 
    - `defaultdict`: Varsayılan değerlerle dictionary (Elo rating'ler için)
    - `deque`: Hızlı FIFO queue (son N maç için sliding window)

---

## 🧱 Proje Fazları ve Dosya Yapısı

### FAZ 0 – Proje İskeleti & Ortam

#### `src/utils/config.py`
**Amaç**: Proje genelinde kullanılan dosya yollarını merkezi olarak tanımlar.

**İçerik**:
- `PROJECT_ROOT`: Proje kök dizini (otomatik hesaplanır)
- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`: Veri klasörleri
- `MODELS_DIR`: Model dosyalarının bulunduğu klasör
- `NOTEBOOKS_DIR`: Jupyter notebook'ları
- `ALLYEARS_PATH`: Ham veri dosyası yolu

**Bağlantılar**:
- Tüm modüller bu dosyadan path'leri import eder
- Değişiklik yapıldığında tek yerden güncellenir

**Kullanılan kütüphaneler**: `pathlib` (Path objeleri için)

---

#### `src/utils/feature_utils.py`
**Amaç**: Feature listesi yükleme gibi ortak fonksiyonları içerir.

**İçerik**:
- `load_feature_list(path)`: `feature_columns.txt` dosyasından feature listesini okur

**Bağlantılar**:
- `src/models/score_all_matches.py`: Model tahminleri için feature listesi yükler
- `src/predict/whatif.py`: What-if tahminleri için feature listesi yükler
- `streamlit_app.py`: Web arayüzünde feature listesi yükler

**Kullanılan kütüphaneler**: `pathlib`, `typing`

---

### FAZ 1 – Veri Toplama & Temizleme

#### `src/data/schema.py`
**Amaç**: Proje genelinde kullanılan standart veri şemasını tanımlar.

**İçerik**:
- `MATCH_COLUMNS`: Tüm maç verilerinde bulunması gereken kolonların listesi
  - `date`, `tourney`, `surface`, `round`: Maç bilgileri
  - `playerA`, `playerB`, `rankA`, `rankB`: Oyuncu bilgileri
  - `oddsA`, `oddsB`: Bahis oranları
  - `winner`: Maç sonucu ('A' veya 'B')
  - `playerA_norm`, `playerB_norm`: Normalize edilmiş oyuncu isimleri

**Bağlantılar**:
- `src/data/preprocess.py`: Şemaya uygun veri üretir
- `src/data/cleaning.py`: Şemaya göre veri temizler

**Kullanılan kütüphaneler**: Yok (sadece liste tanımı)

---

#### `src/data/fetch_data.py`
**Amaç**: tennis-data.co.uk'ten yıllık Excel dosyalarını indirip tek bir CSV'de birleştirir.

**İşlevler**:
- `download_season(year)`: Belirli bir yılın Excel dosyasını indirir
- `build_allyears_csv()`: 2000-2025 arası tüm yılları indirip birleştirir

**Çıktı**: `data/raw/allyears.csv`

**Bağlantılar**:
- `src/data/preprocess.py`: Bu dosyayı okur ve işler

**Kullanılan kütüphaneler**:
- `requests`: HTTP istekleri ile Excel dosyalarını indirir
- `pandas`: Excel okuma (`pd.read_excel()`) ve birleştirme (`pd.concat()`)
- `openpyxl`: Excel dosyalarını okumak için (pandas'ın arka planında)
- `pathlib`: Dosya yolu yönetimi

**Neden bu kütüphaneler**:
- `requests`: Basit HTTP istekleri için standart kütüphane
- `pandas`: Excel okuma ve veri birleştirme için en uygun
- `openpyxl`: `.xlsx` formatını destekler

---

#### `src/data/preprocess.py`
**Amaç**: Ham veriyi normalize eder, kolon isimlerini standartlaştırır.

**İşlevler**:
- `build_matches_from_allyears()`: 
  - Kolon isimlerini normalize eder (Date → date, Tournament → tourney)
  - Tarih formatlarını düzeltir
  - Oyuncu isimlerini normalize eder (lowercase, trim)
  - Bahis oranlarından implied probability hesaplar
  - `MATCH_COLUMNS` şemasına uygun hale getirir

**Girdi**: `data/raw/allyears.csv`
**Çıktı**: `data/processed/matches_allyears.csv`

**Bağlantılar**:
- `src/data/schema.py`: `MATCH_COLUMNS` şemasını kullanır
- `src/data/cleaning.py`: Bu dosyayı okur ve temizler

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, kolon dönüşümleri, string işlemleri
- `pathlib`: Dosya yolu yönetimi

---

#### `src/data/cleaning.py`
**Amaç**: Veri setindeki hataları ve eksiklikleri temizler.

**İşlevler**:
- `build_clean_matches()`:
  - Eksik kritik bilgileri olan satırları kaldırır
  - Geçersiz bahis oranlarını temizler
  - Bozuk/hatalı satırları filtreler

**Girdi**: `data/processed/matches_allyears.csv`
**Çıktı**: `data/processed/matches_clean.csv`

**Bağlantılar**:
- `src/data/schema.py`: `MATCH_COLUMNS` şemasını kullanır
- `src/features/elo.py`: Bu dosyayı okur ve Elo hesaplar

**Kullanılan kütüphaneler**:
- `pandas`: Veri filtreleme, temizleme işlemleri
- `pathlib`: Dosya yolu yönetimi

---

### FAZ 2 – Feature Engineering

#### `src/features/elo.py`
**Amaç**: Oyuncular için Elo rating sistemini hesaplar.

**İşlevler**:
- `expected_score(r_a, r_b)`: İki rating arasından beklenen skoru hesaplar
  - Formül: `1 / (1 + 10^((r_b - r_a) / 400))`
- `compute_elo_for_matches()`: Tüm maçlar için Elo güncellemesi yapar
  - Global Elo: Tüm maçlar için tek rating
  - Surface Elo: Zemin bazlı ayrı rating (Hard, Clay, Grass, Carpet)

**Üretilen Feature'lar**:
- `eloA`, `eloB`: Oyuncuların global Elo rating'leri
- `elo_diff`: Elo farkı (eloA - eloB)
- `elo_surfaceA`, `elo_surfaceB`: Zemin bazlı Elo rating'leri
- `elo_surface_diff`: Zemin bazlı Elo farkı

**Parametreler**:
- `BASE_ELO = 1500.0`: Başlangıç rating'i
- `K_OVERALL = 32.0`: Global Elo için güncelleme katsayısı
- `K_SURFACE = 24.0`: Surface Elo için güncelleme katsayısı (daha yavaş değişim)

**Girdi**: `data/processed/matches_clean.csv`
**Çıktı**: `data/processed/matches_with_elo.csv`

**Bağlantılar**:
- `src/features/form.py`: Bu dosyayı okur ve form feature'ları ekler

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, tarih sıralama, iterasyon
- `collections.defaultdict`: Her oyuncu için rating takibi (varsayılan değer: BASE_ELO)
- `pathlib`: Dosya yolu yönetimi

**Neden Elo Rating**:
- Satranç'tan uyarlanmış, tenis için de etkili
- Oyuncuların gücünü tek bir sayıyla ifade eder
- Zaman içinde performans değişimini yansıtır
- Zemin bazlı Elo, oyuncuların zemin tercihlerini yakalar

---

#### `src/features/form.py`
**Amaç**: Oyuncuların kısa vadeli formunu ve maç yoğunluğunu hesaplar.

**İşlevler**:
- `compute_form_features()`:
  - **Son N maç kazanma oranı**: Son 5 ve 10 maçtaki başarı oranı
  - **Dinlenme süresi**: Son maçtan bu maça kadar geçen gün sayısı
  - **Maç yoğunluğu**: Son 30 gündeki maç sayısı

**Üretilen Feature'lar**:
- `form_winrateA_5`, `form_winrateB_5`: Son 5 maç kazanma oranı
- `form_winrateA_10`, `form_winrateB_10`: Son 10 maç kazanma oranı
- `days_since_lastA`, `days_since_lastB`: Son maçtan bu yana geçen gün
- `matches_last30A`, `matches_last30B`: Son 30 gündeki maç sayısı

**Girdi**: `data/processed/matches_with_elo.csv`
**Çıktı**: `data/processed/matches_with_elo_form.csv`

**Bağlantılar**:
- `src/features/sets.py`: Bu dosyayı okur ve set feature'ları ekler

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, tarih işlemleri, groupby operasyonları
- `numpy`: Matematiksel hesaplamalar, NaN handling
- `collections.deque`: Son N maçı takip etmek için (sliding window)
- `collections.defaultdict`: Oyuncu bazlı istatistik takibi
- `pathlib`: Dosya yolu yönetimi

**Neden bu feature'lar**:
- **Form**: Son performans, gelecek performansın göstergesi
- **Dinlenme**: Yorgunluk faktörü, sakatlık riski
- **Yoğunluk**: Aşırı maç yorgunluğu, form düşüşü

---

#### `src/features/sets.py`
**Amaç**: Set bazlı performans istatistiklerini hesaplar.

**İşlevler**:
- `load_bo5_tournaments_from_raw()`: Grand Slam gibi best-of-5 turnuvaları tespit eder
- `infer_best_of()`: Maçın best-of-3 mü best-of-5 mi olduğunu çıkarır
- `add_set_based_features()`: Set kazanma oranlarını hesaplar

**Üretilen Feature'lar**:
- `set_winrate_overallA`, `set_winrate_overallB`: Genel set kazanma oranı
- `set_winrate_bo3A`, `set_winrate_bo3B`: Best-of-3 maçlarda set kazanma oranı
- `set_winrate_bo5A`, `set_winrate_bo5B`: Best-of-5 maçlarda set kazanma oranı
- `inferred_best_of`: Maçın best-of değeri (3 veya 5)

**Girdi**: `data/processed/matches_with_elo_form.csv`
**Çıktı**: `data/processed/matches_with_elo_form_sets.csv`

**Bağlantılar**:
- `src/features/build_features.py`: Bu dosyayı okur ve tüm feature'ları birleştirir

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, set skorlarını parse etme
- `collections.defaultdict`: Oyuncu bazlı set istatistik takibi
- `pathlib`: Dosya yolu yönetimi

**Neden set feature'ları**:
- Set kazanma oranı, maç kazanma oranından daha detaylı bilgi verir
- Best-of-3 ve best-of-5 farklı stratejiler gerektirir
- Oyuncuların dayanıklılığını ölçer

---

#### `src/features/build_features.py`
**Amaç**: Tüm feature'ları birleştirip model eğitimi için hazır hale getirir.

**İşlevler**:
- `add_h2h_features()`: Head-to-head (karşılaşma geçmişi) feature'ları
  - `h2h_matches`: Daha önce kaç kez karşılaştılar
  - `h2h_winrateA`, `h2h_winrateB`: H2H kazanma oranları
- `add_tournament_round_features()`: Turnuva turu feature'ları
  - `round_importance`: Tur önemi (1-7 arası, Final=7)
  - `is_final`, `is_semi`, `is_quarter`: Boolean flag'ler
- `add_series_features()`: Turnuva seviyesi feature'ları
  - `series_tier`: Turnuva seviyesi (Grand Slam=4.0, Masters=3.0, vb.)
  - `is_grand_slam`: Grand Slam flag'i
  - `is_bo5_match`: Best-of-5 flag'i
- `random_flip_perspective()`: Veri augmentasyonu
  - Satırların yarısında playerA/playerB yer değiştirir
  - `y` etiketi buna göre ayarlanır (1 veya 0)
  - Modelin A/B perspektifine bağlı kalmamasını sağlar
- `add_market_features()`: Bahis oranlarından feature'lar
  - `pA_market`, `pB_market`: Market'in implied probability'si
  - `logit_pA_market`: Logit transformasyonu
  - `p_diff`: Olasılık farkı
- `clip_days_features()`: Aşırı değerleri kırpar (0-365 gün arası)
- `add_diff_features()`: Tüm A/B feature'ları arasındaki farkları hesaplar
- `build_feature_dataset()`: Ana pipeline fonksiyonu

**Üretilen Feature Kategorileri**:

1. **Elo Features** (6 adet):
   - `eloA`, `eloB`, `elo_diff`
   - `elo_surfaceA`, `elo_surfaceB`, `elo_surface_diff`

2. **Form Features** (8 adet):
   - `form_winrateA_5`, `form_winrateB_5`, `form_winrate_diff_5`
   - `form_winrateA_10`, `form_winrateB_10`, `form_winrate_diff_10`
   - `days_since_lastA_clipped`, `days_since_lastB_clipped`, `days_since_last_diff_clipped`
   - `matches_last30A`, `matches_last30B`, `matches_last30_diff`

3. **Rank Features** (3 adet):
   - `rankA`, `rankB`, `rank_diff`

4. **H2H Features** (4 adet):
   - `h2h_matches`, `h2h_winrateA`, `h2h_winrateB`, `h2h_winrate_diff`

5. **Round Features** (4 adet):
   - `round_importance`, `is_final`, `is_semi`, `is_quarter`

6. **Tournament Features** (3 adet):
   - `series_tier`, `is_grand_slam`, `is_bo5_match`

7. **Set Features** (9 adet):
   - `set_winrate_overallA`, `set_winrate_overallB`, `set_winrate_overall_diff`
   - `set_winrate_bo3A`, `set_winrate_bo3B`, `set_winrate_bo3_diff`
   - `set_winrate_bo5A`, `set_winrate_bo5B`, `set_winrate_bo5_diff`

8. **Market Features** (6 adet):
   - `oddsA`, `oddsB`
   - `pA_market`, `pB_market`, `p_diff`, `logit_pA_market`

9. **Surface Features** (4-5 adet, one-hot encoding):
   - `surface_Hard`, `surface_Clay`, `surface_Grass`, `surface_Carpet`

**Toplam**: ~40-45 feature

**Girdi**: `data/processed/matches_with_elo_form_sets.csv`
**Çıktı**: `data/processed/train_dataset.csv`

**Bağlantılar**:
- `src/models/train_logreg.py`: Bu dosyayı okur ve model eğitir

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, birleştirme, feature engineering
- `numpy`: Matematiksel işlemler, NaN handling
- `pathlib`: Dosya yolu yönetimi

**Neden bu feature'lar**:
- **Elo**: Oyuncu gücü
- **Form**: Son performans
- **H2H**: Geçmiş karşılaşmalar
- **Round/Tournament**: Maç önemi
- **Market**: Bahis şirketlerinin görüşü (baseline)
- **Surface**: Zemin tercihi
- **Set winrate**: Detaylı performans

---

### FAZ 3 – Model Eğitimi ve Değerlendirme

#### `src/models/train_logreg.py`
**Amaç**: Logistic Regression modelini eğitir ve kaydeder.

**İşlevler**:
- `train_logistic_regression()`:
  1. `train_dataset.csv` dosyasını okur
  2. Feature ve meta kolonlarını ayırır
  3. Zaman bazlı train/validation split yapar (2022 öncesi → train, 2022+ → validation)
  4. Eksik değerleri `SimpleImputer(strategy="median")` ile doldurur
  5. Logistic Regression modelini eğitir
  6. Validation metriklerini hesaplar ve yazdırır
  7. Model, imputer ve feature listesini kaydeder

**Model Parametreleri**:
- `penalty="l2"`: Ridge regularization (overfitting'i önler)
- `C=1.0`: Regularization gücü (küçük = daha fazla regularization)
- `solver="lbfgs"`: Optimizasyon algoritması (küçük-orta veri setleri için uygun)
- `max_iter=1000`: Maksimum iterasyon sayısı

**Çıktılar**:
- `models/logreg_final.pkl`: Eğitilmiş model
- `models/imputer_final.pkl`: Eksik değer doldurucu (tahmin sırasında da kullanılır)
- `models/feature_columns.txt`: Kullanılan feature listesi

**Girdi**: `data/processed/train_dataset.csv`
**Çıktı**: Model dosyaları (`models/` klasöründe)

**Bağlantılar**:
- `src/models/score_all_matches.py`: Bu modeli yükler ve tüm maçlara tahmin yapar
- `src/predict/whatif.py`: Bu modeli yükler ve what-if tahminleri yapar
- `streamlit_app.py`: Bu modeli yükler ve web arayüzünde kullanır

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, tarih filtreleme
- `numpy`: Array işlemleri
- `sklearn.linear_model.LogisticRegression`: Model eğitimi
- `sklearn.impute.SimpleImputer`: Eksik değer doldurma
- `sklearn.metrics`: Performans metrikleri (log_loss, brier_score_loss, accuracy_score)
- `joblib`: Model kaydetme
- `pathlib`: Dosya yolu yönetimi

**Neden Logistic Regression**:
- Basit ve yorumlanabilir
- Overfitting riski düşük
- Bahis şirketleri ile başa baş performans
- Ders projesi için anlatması kolay
- XGBoost/RandomForest'a göre daha stabil

**Metrikler**:
- **LogLoss**: Olasılık tahminlerinin kalitesi (düşük = iyi)
- **Brier Score**: Olasılık karesel hata (düşük = iyi)
- **Accuracy**: Doğru tahmin oranı (yüksek = iyi)

---

#### `src/models/score_all_matches.py`
**Amaç**: Eğitilmiş modeli kullanarak tüm maçlara tahmin yapar.

**İşlevler**:
- `main()`:
  1. `train_dataset.csv` dosyasını okur
  2. Model, imputer ve feature listesini yükler
  3. Her maç için model tahmini yapar (`p_model`)
  4. Market olasılığı varsa edge hesaplar (`p_model - pA_market`)
  5. Sonuçları CSV'ye kaydeder

**Çıktı**: `data/processed/all_predictions.csv`

**Kolonlar**:
- `date`, `surface`, `playerA`, `playerB`, `y`: Meta bilgiler
- `p_model`: Model tahmini (A kazanır olasılığı)
- `pA_market`: Market tahmini (varsa)
- `edge`: Model - Market farkı (varsa)

**Bağlantılar**:
- `streamlit_app.py`: Bu dosyayı okur ve web arayüzünde gösterir

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma/yazma
- `numpy`: Array işlemleri
- `joblib`: Model yükleme
- `pathlib`: Dosya yolu yönetimi
- `src.utils.feature_utils`: Feature listesi yükleme

---

#### `src/analysis/metrics.py`
**Amaç**: Model ve market performans metriklerini hesaplar.

**İşlevler**:
- `compute_overall_metrics(df)`: 
  - Model metrikleri: logloss, brier, accuracy
  - Market metrikleri: logloss, brier, accuracy (varsa)
  - `OverallMetrics` dataclass'ı döndürür

**Kullanım**: `streamlit_app.py` içinde metrik gösterimi için

**Kullanılan kütüphaneler**:
- `pandas`: Veri işleme
- `numpy`: Array işlemleri
- `sklearn.metrics`: Metrik hesaplama
- `dataclasses`: Structured data için

---

### FAZ 4 – Tahmin & Arayüz

#### `src/predict/whatif.py`
**Amaç**: Senaryo bazlı tahmin yapar (what-if analizi).

**İşlevler**:
- `get_player_snapshot()`: Belirli bir tarihteki oyuncu durumunu çıkarır
- `compute_h2h()`: İki oyuncu arasındaki head-to-head istatistiklerini hesaplar
- `build_feature_row()`: Verilen parametreler için feature vektörü oluşturur
- `predict_single_match()`: Tek maç için tahmin yapar
- `main()`: CLI arayüzü

**Kullanım**:
```bash
py -m src.predict.whatif --playerA "Roger Federer" --playerB "Rafael Nadal" --surface "Hard" --date "2020-01-15"
```

**Bağlantılar**:
- `streamlit_app.py`: What-if sekmesinde bu fonksiyonları kullanır

**Kullanılan kütüphaneler**:
- `pandas`: Veri okuma, tarih işlemleri
- `numpy`: Matematiksel işlemler
- `joblib`: Model yükleme
- `argparse`: CLI argüman parsing
- `dataclasses`: PlayerSnapshot için
- `pathlib`: Dosya yolu yönetimi
- `src.utils.feature_utils`: Feature listesi yükleme

---

#### `streamlit_app.py`
**Amaç**: İnteraktif web arayüzü sağlar.

**Özellikler**:

1. **Matches Sekmesi**:
   - Maç filtreleme (tarih, zemin, oyuncu, turnuva)
   - Model vs Market karşılaştırması
   - Edge (model - market) filtreleme
   - Maç detayları görüntüleme

2. **What-if Sekmesi**:
   - Senaryo bazlı tahmin
   - Oyuncu, tarih, zemin seçimi
   - Opsiyonel: Bahis oranları girişi
   - Model tahmini ve güven seviyesi

3. **Leaderboard Sekmesi**:
   - Oyuncu performans sıralaması
   - Win rate, wins, avg edge metrikleri
   - Minimum maç sayısı filtresi

**Bağlantılar**:
- `src/analysis/metrics.py`: Metrik hesaplama
- `src/predict/whatif.py`: What-if tahminleri
- `src/utils/feature_utils.py`: Feature listesi yükleme
- `src/utils/config.py`: Dosya yolları

**Kullanılan kütüphaneler**:
- `streamlit`: Web arayüzü framework'ü
- `pandas`: Veri işleme
- `numpy`: Array işlemleri
- `joblib`: Model yükleme
- `pathlib`: Dosya yolu yönetimi

**Neden Streamlit**:
- Hızlı prototipleme
- Minimal kod ile interaktif dashboard
- Python tabanlı (ekstra frontend bilgisi gerekmez)
- Otomatik widget'lar ve layout yönetimi

---

## 📂 Klasör Yapısı (Detaylı)

```
predictive-serve/
├─ data/                          # Veri dosyaları
│  ├─ raw/                        # Ham veri
│  │  └─ allyears.csv            # tennis-data.co.uk'tan indirilen birleşik veri
│  └─ processed/                 # İşlenmiş veri
│     ├─ matches_allyears.csv     # preprocess sonrası (normalize edilmiş)
│     ├─ matches_clean.csv       # cleaning sonrası (temizlenmiş)
│     ├─ matches_with_elo.csv    # Elo feature'ları eklenmiş
│     ├─ matches_with_elo_form.csv # Form feature'ları eklenmiş
│     ├─ matches_with_elo_form_sets.csv # Set feature'ları eklenmiş
│     ├─ train_dataset.csv       # Final feature seti (model eğitimi için)
│     ├─ all_predictions.csv     # Tüm maçlara model tahminleri
│     └─ val_predictions.csv     # Validation set tahminleri (opsiyonel)
│
├─ models/                        # Eğitilmiş modeller
│  ├─ logreg_final.pkl           # Logistic Regression modeli
│  ├─ imputer_final.pkl          # Eksik değer doldurucu
│  └─ feature_columns.txt        # Kullanılan feature listesi
│
├─ notebooks/                     # Jupyter notebook'lar
│  ├─ 01_eda_matches_allyears.ipynb  # Exploratory Data Analysis
│  └─ 02_train_models.ipynb          # Model karşılaştırma ve metrikler
│
├─ src/                           # Kaynak kod
│  ├─ __init__.py                # Paket marker
│  │
│  ├─ data/                       # Veri toplama ve temizleme
│  │  ├─ fetch_data.py           # Veri indirme (tennis-data.co.uk)
│  │  ├─ preprocess.py           # Veri normalizasyonu
│  │  ├─ cleaning.py             # Veri temizleme
│  │  └─ schema.py               # Veri şeması tanımları
│  │
│  ├─ features/                   # Feature engineering
│  │  ├─ elo.py                  # Elo rating hesaplama
│  │  ├─ form.py                 # Form ve yoğunluk feature'ları
│  │  ├─ sets.py                 # Set bazlı performans feature'ları
│  │  └─ build_features.py       # Tüm feature'ları birleştirme
│  │
│  ├─ models/                     # Model eğitimi
│  │  ├─ train_logreg.py         # Logistic Regression eğitimi
│  │  └─ score_all_matches.py    # Tüm maçlara tahmin
│  │
│  ├─ predict/                    # Tahmin fonksiyonları
│  │  └─ whatif.py               # What-if senaryo tahminleri
│  │
│  ├─ analysis/                   # Analiz ve metrikler
│  │  └─ metrics.py              # Performans metrikleri
│  │
│  └─ utils/                      # Yardımcı fonksiyonlar
│     ├─ config.py               # Dosya yolu konfigürasyonu
│     └─ feature_utils.py       # Feature listesi yükleme
│
├─ test_quick.py                  # Hızlı test (paket gerektirmez)
├─ test_system.py                 # Tam sistem testi (paket gerektirir)
├─ streamlit_app.py               # Streamlit web arayüzü
├─ requirements.txt               # Python bağımlılıkları
├─ .gitignore                     # Git ignore kuralları
└─ README.md                      # Bu dosya
```

---

## 🧪 Test ve Kullanım

### 🚀 Hızlı Başlangıç (Windows - Tek Tıkla Çalıştırma)

**En kolay yöntem:** `run_predictive_serve.bat` dosyasını çift tıklayın!

Bu script otomatik olarak:
- ✅ Gerekli paketleri kontrol eder ve yükler
- ✅ Tennis-data.co.uk'den **güncel verileri** çeker (2000-2025)
- ✅ Tüm veri işleme adımlarını çalıştırır (preprocess, cleaning, features)
- ✅ Modeli eğitir ve tahminler yapar
- ✅ Streamlit web arayüzünü başlatır

**Kullanım:**
```bash
# Windows'ta çift tıklayın veya:
run_predictive_serve.bat
```

**Not:** İlk çalıştırmada veri indirme ve model eğitimi birkaç dakika sürebilir. Sonraki çalıştırmalarda sadece güncel veriler indirilir ve pipeline yeniden çalıştırılır.

### Kurulum (Manuel)

```bash
pip install -r requirements.txt
```

### Hızlı Test (Paket Gerektirmez)

Sadece dosya ve klasör yapısını kontrol eder:

```bash
py test_quick.py
```

### Tam Sistem Testi

Tüm bileşenleri test eder (paketler gerekli):

```bash
py test_system.py
```

**Test Edilenler**:
- ✅ Proje yapısı ve dosyalar
- ✅ Python modülleri import
- ✅ Model yükleme
- ✅ Veri seti kontrolü
- ✅ Feature engineering fonksiyonları
- ✅ Tahmin fonksiyonları
- ✅ Metrik hesaplama
- ✅ End-to-end pipeline

### Pipeline Çalıştırma

Eğer veriler eksikse, pipeline'ı baştan çalıştırın:

```bash
# 1. Veri toplama (opsiyonel)
py -m src.data.fetch_data

# 2. Preprocess
py -m src.data.preprocess

# 3. Cleaning
py -m src.data.cleaning

# 4. Elo hesaplama
py -m src.features.elo

# 5. Form feature'ları
py -m src.features.form

# 6. Set feature'ları
py -m src.features.sets

# 7. Tüm feature'ları oluştur
py -m src.features.build_features

# 8. Model eğitimi
py -m src.models.train_logreg

# 9. Tüm maçlara tahmin
py -m src.models.score_all_matches
```

### Streamlit Uygulaması

Web arayüzünü başlatın:

```bash
streamlit run streamlit_app.py
```

**Özellikler**:
- 📊 **Matches**: Maç filtreleme, arama ve detay görüntüleme
- 🔮 **What-if**: Senaryo bazlı tahmin (oyuncu, tarih, zemin seçimi)
- 🏆 **Leaderboard**: Oyuncu performans sıralaması
- 📈 Model vs Market karşılaştırması

### What-if Tahmini (CLI)

Komut satırından tahmin yapın:

```bash
py -m src.predict.whatif --playerA "Roger Federer" --playerB "Rafael Nadal" --surface "Hard" --date "2020-01-15"
```

### Sorun Giderme

**Problem: "ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**Problem: "FileNotFoundError: train_dataset.csv"**
→ Pipeline'ı baştan çalıştırın (yukarıdaki adımlar)

**Problem: "Model yüklenemedi"**
```bash
py -m src.models.train_logreg
```

---

## 📊 Feature Detayları

### Elo Features

**Amaç**: Oyuncuların gücünü tek bir sayıyla ifade eder.

**Hesaplama**:
- Başlangıç: 1500.0
- Her maç sonrası güncellenir: `new_rating = old_rating + K * (actual_score - expected_score)`
- Global Elo: Tüm maçlar için (K=32)
- Surface Elo: Zemin bazlı (K=24, daha yavaş değişim)

**Kullanım**: Model, Elo farkını kullanarak oyuncu gücü farkını öğrenir.

---

### Form Features

**Amaç**: Oyuncuların son performansını ve yorgunluk durumunu ölçer.

**Hesaplama**:
- **Son 5/10 maç winrate**: Son N maçtaki kazanma oranı
- **Days since last**: Son maçtan bu yana geçen gün (0-365 arası kırpılır)
- **Matches last 30 days**: Son 30 gündeki maç sayısı

**Kullanım**: 
- Yüksek form → Yüksek kazanma şansı
- Çok maç (yoğunluk) → Yorgunluk riski
- Uzun dinlenme → Paslanma riski

---

### H2H (Head-to-Head) Features

**Amaç**: İki oyuncu arasındaki geçmiş karşılaşmaları yansıtır.

**Hesaplama**:
- Bu maça kadar olan tüm karşılaşmalar sayılır
- Kazanma oranları hesaplanır

**Kullanım**: Bazı oyuncular belirli rakiplere karşı daha iyi/beter performans gösterir.

---

### Market Features

**Amaç**: Bahis şirketlerinin görüşünü yansıtır (baseline).

**Hesaplama**:
- Bahis oranlarından implied probability: `p = 1/odds / (1/oddsA + 1/oddsB)`
- Logit transformasyonu: `logit(p) = log(p / (1-p))`

**Kullanım**: 
- Model'in market'i geçip geçmediğini ölçer
- Edge = Model tahmini - Market tahmini

---

### Round & Tournament Features

**Amaç**: Maçın önemini ve turnuva seviyesini yansıtır.

**Hesaplama**:
- Round importance: Final=7, Semi=6, Quarter=5, ...
- Series tier: Grand Slam=4.0, Masters=3.0, ATP 500=2.0, ...

**Kullanım**: Önemli maçlarda oyuncular daha fazla çaba gösterir.

---

### Set Features

**Amaç**: Set bazlı performans, maç bazlı performanstan daha detaylıdır.

**Hesaplama**:
- Set kazanma oranı (genel, BO3, BO5)
- Best-of-3 ve best-of-5 farklı stratejiler gerektirir

**Kullanım**: Set kazanma oranı, oyuncunun dayanıklılığını ve set bazlı gücünü gösterir.

---

## 🔄 Veri Akışı (Pipeline)

```
1. fetch_data.py
   ↓
   data/raw/allyears.csv
   
2. preprocess.py
   ↓
   data/processed/matches_allyears.csv
   
3. cleaning.py
   ↓
   data/processed/matches_clean.csv
   
4. elo.py
   ↓
   data/processed/matches_with_elo.csv
   
5. form.py
   ↓
   data/processed/matches_with_elo_form.csv
   
6. sets.py
   ↓
   data/processed/matches_with_elo_form_sets.csv
   
7. build_features.py
   ↓
   data/processed/train_dataset.csv
   
8. train_logreg.py
   ↓
   models/logreg_final.pkl
   models/imputer_final.pkl
   models/feature_columns.txt
   
9. score_all_matches.py
   ↓
   data/processed/all_predictions.csv
   
10. streamlit_app.py (veya whatif.py)
    ↓
    Kullanıcı arayüzü / CLI tahminleri
```

---

## 📝 Notlar

- Tüm dosya yolları `src/utils/config.py` içinde merkezi olarak tanımlanır
- Feature listesi `models/feature_columns.txt` içinde saklanır (model ve tahmin için gerekli)
- Model eğitimi zaman bazlı split kullanır (2022 öncesi → train, 2022+ → validation)
- Veri augmentasyonu: `random_flip_perspective()` ile A/B perspektifi rastgele değiştirilir
- Eksik değerler: Median imputation kullanılır (SimpleImputer)

---
