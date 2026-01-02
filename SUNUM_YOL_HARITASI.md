# 🎯 Predictive Serve - Python Dersi Sunum Yol Haritası

## 👥 EKİP (4 KİŞİ - EŞİT AĞIRLIK)

### 📊 **GÖREV DAĞILIMI** (Her biri ~8-10 dakika)

| Kişi | Rol | Ana Görev | Süre | Dosyalar |
|------|-----|-----------|------|----------|
| **Kişi 1** | **Veri Mühendisi** | Veri Toplama ve Temizleme | 8-9 dk | `src/data/*`, `src/utils/config.py` |
| **Kişi 2** | **Feature Engineer** | Feature Engineering | 8-9 dk | `src/features/*`, `src/utils/feature_utils.py` |
| **Kişi 3** | **ML Engineer** | Model Eğitimi ve Değerlendirme | 8-9 dk | `src/models/*`, `src/analysis/*` |
| **Kişi 4** | **Frontend Developer** | Tahmin ve Arayüz | 8-9 dk | `streamlit_app.py`, `src/predict/*` |

---

## 👤 **KİŞİ 1 - Veri Mühendisi: Veri Toplama ve Temizleme**

### 🎯 Sorumlu Olduğu Bölüm: **FAZ 1 - Veri Toplama ve Temizleme**

**Sorumlu Olduğu Dosyalar:**
- `src/data/fetch_data.py` - Veri indirme
- `src/data/preprocess.py` - Veri normalizasyonu
- `src/data/cleaning.py` - Veri temizleme
- `src/data/schema.py` - Veri şeması tanımları
- `src/utils/config.py` - Dosya yolu konfigürasyonu

**Sunum Süresi:** 8-9 dakika

---

## 👤 **KİŞİ 2 - Feature Engineer: Feature Engineering**

### 🎯 Sorumlu Olduğu Bölüm: **FAZ 2 - Feature Engineering**

**Sorumlu Olduğu Dosyalar:**
- `src/features/elo.py` - Elo rating hesaplama
- `src/features/form.py` - Form ve yoğunluk feature'ları
- `src/features/sets.py` - Set bazlı performans feature'ları
- `src/features/build_features.py` - Tüm feature'ları birleştirme
- `src/utils/feature_utils.py` - Feature listesi yükleme

**Sunum Süresi:** 8-9 dakika

---

## 👤 **KİŞİ 3 - ML Engineer: Model Eğitimi ve Değerlendirme**

### 🎯 Sorumlu Olduğu Bölüm: **FAZ 3 - Model Eğitimi ve Değerlendirme**

**Sorumlu Olduğu Dosyalar:**
- `src/models/train_logreg.py` - Model eğitimi
- `src/models/score_all_matches.py` - Tüm maçlara tahmin
- `src/analysis/metrics.py` - Performans metrikleri

**Sunum Süresi:** 8-9 dakika

---

## 👤 **KİŞİ 4 - Frontend Developer: Tahmin ve Arayüz**

### 🎯 Sorumlu Olduğu Bölüm: **FAZ 4 - Tahmin ve Arayüz**

**Sorumlu Olduğu Dosyalar:**
- `src/predict/whatif.py` - What-if senaryo tahminleri
- `streamlit_app.py` - Streamlit web arayüzü

**Sunum Süresi:** 8-9 dakika

---

## 📊 SUNUM ZAMAN ÇİZELGESİ (Toplam: 35-40 dakika)

| Sıra | Süre | Bölüm | Sorumlu | Görseller |
|------|------|-------|---------|-----------|
| 0 | 2-3 dk | **Proje Tanıtımı** | **Tüm Ekip** (Herkes 30-45 sn) | GÖRSELL 1, 8, 18 |
| 1 | 8-9 dk | **Veri Toplama ve Temizleme** | **Kişi 1** | GÖRSELL 2, 15 |
| 2 | 8-9 dk | **Feature Engineering** | **Kişi 2** | GÖRSELL 3, 4, 5, 9, 11, 12, 13, 17 |
| 3 | 8-9 dk | **Model Eğitimi ve Değerlendirme** | **Kişi 3** | GÖRSELL 6, 7, 10, 14 |
| 4 | 8-9 dk | **Tahmin ve Arayüz** | **Kişi 4** | GÖRSELL 16, Canlı Demo |
| 5 | 3-4 dk | **Sonuç ve Soru-Cevap** | **Tüm Ekip** (Herkes 30-45 sn) | - |

**Toplam:** ~37-43 dakika

---

## 🎨 GÖRSELLER LİSTESİ (18 Görsel)

Görseller `presentation_visuals/` klasöründe mevcut.

| Görsel | Dosya Adı | Kullanım Yeri | Sorumlu | Açıklama |
|--------|-----------|---------------|---------|----------|
| **GÖRSELL 1** | `visual_1_pipeline_diagram.png` | Proje Tanıtımı | Tüm Ekip | Pipeline diyagramı |
| **GÖRSELL 2** | `visual_2_data_stats.png` | Veri Toplama | Kişi 1 | Veri istatistikleri |
| **GÖRSELL 3** | `visual_3_feature_categories.png` | Feature Engineering | Kişi 2 | Feature kategorileri |
| **GÖRSELL 4** | `visual_4_elo_example.png` | Feature Engineering | Kişi 2 | Elo rating örneği |
| **GÖRSELL 5** | `visual_5_form_features.png` | Feature Engineering | Kişi 2 | Form features |
| **GÖRSELL 6** | `visual_6_model_metrics.png` | Model Eğitimi | Kişi 3 | Model metrikleri |
| **GÖRSELL 7** | `visual_7_confusion_matrix.png` | Model Eğitimi | Kişi 3 | Confusion matrix |
| **GÖRSELL 8** | `visual_8_data_flow.png` | Proje Tanıtımı | Tüm Ekip | Veri akış diyagramı |
| **GÖRSELL 9** | `visual_9_feature_importance.png` | Feature Engineering | Kişi 2 | Feature importance |
| **GÖRSELL 10** | `visual_10_edge_distribution.png` | Model Eğitimi | Kişi 3 | Edge dağılımı |
| **GÖRSELL 11** | `visual_11_h2h_example.png` | Feature Engineering | Kişi 2 | H2H örneği |
| **GÖRSELL 12** | `visual_12_market_features.png` | Feature Engineering | Kişi 2 | Market features analizi |
| **GÖRSELL 13** | `visual_13_surface_performance.png` | Feature Engineering | Kişi 2 | Zemin bazlı performans |
| **GÖRSELL 14** | `visual_14_training_process.png` | Model Eğitimi | Kişi 3 | Eğitim süreci |
| **GÖRSELL 15** | `visual_15_data_quality.png` | Veri Toplama | Kişi 1 | Veri kalitesi metrikleri |
| **GÖRSELL 16** | `visual_16_streamlit_ui.png` | Tahmin ve Arayüz | Kişi 4 | Streamlit UI özellikleri |
| **GÖRSELL 17** | `visual_17_feature_correlation.png` | Feature Engineering | Kişi 2 | Feature correlation |
| **GÖRSELL 18** | `visual_18_project_architecture.png` | Proje Tanıtımı | Tüm Ekip | Proje mimarisi |

---

## 🎯 BÖLÜM 0: PROJE TANITIMI (2-3 dakika)

### 👥 Sorumlu: **Tüm Ekip** (Herkes 30-45 saniye)

### 📝 Sunum İçeriği:

#### 1. Açılış (30 saniye) - **Kişi 1**
**Söylenecekler:**
- "Merhaba, ben [İsim], veri mühendisiyim"
- "Bugün sizlere **Predictive Serve** projemizi sunacağız"
- "Bu proje, tenis maç sonucu tahmini için end-to-end bir Python projesidir"

#### 2. Problem Tanımı (30 saniye) - **Kişi 2**
**Söylenecekler:**
- "Projemizin amacı: Bahis şirketlerinin oranlarına ek olarak oyuncuların formu, Elo rating'i ve diğer istatistikleri kullanarak tenis maç sonucu tahmininde ne kadar iyi olabiliriz?"
- "Python ile makine öğrenmesi kullanarak bu soruyu cevaplıyoruz"

**GÖRSELL 1'i göster** - Pipeline Diyagramı
- "Projemiz 5 ana aşamadan oluşuyor"

#### 3. Proje Yapısı (30 saniye) - **Kişi 3**
**Söylenecekler:**
- **Python Kütüphaneleri:** pandas, numpy, scikit-learn, streamlit
- "Her aşama Python modülleri ile gerçekleştirildi"

**GÖRSELL 8'i göster** - Veri Akış Diyagramı
- "Veri akışını gösteren diyagram"

**GÖRSELL 18'i göster** - Proje Mimarisi
- "Modüler yapı ve katmanlar"

#### 4. Ekip Tanıtımı ve Geçiş (30-45 saniye) - **Kişi 4**
**Söylenecekler:**
- **Ekip tanıtımı:**
  - "[Kişi 1] - Veri toplama ve temizleme"
  - "[Kişi 2] - Feature engineering"
  - "[Kişi 3] - Model eğitimi ve değerlendirme"
  - "[Kişi 4] - Tahmin ve arayüz"
- "Şimdi projenin ilk aşamasına geçiyoruz"
- "[Kişi 1]'e sözü veriyorum"

### 🎤 Sunum İpuçları:
- **Hızlı geçişler**: Herkes kısa ve öz konuşmalı
- **Görseller**: GÖRSELL 1, 8, 18'i göster
- **Enerji**: Açılışı enerjik yapın

---

## 🎯 BÖLÜM 1: VERİ TOPLAMA VE TEMİZLEME (8-9 dakika)

### 👤 Sorumlu: **Kişi 1 - Veri Mühendisi**

### 📝 Detaylı Sunum İçeriği:

#### 1. Giriş (30 saniye)
**Söylenecekler:**
- "Ben [İsim], veri toplama ve temizleme aşamasından sorumluyum"
- "Bu aşamada, ham veriyi toplayıp, Python ile temizleyip, model eğitimi için hazır hale getiriyoruz"

#### 2. Veri Kaynağı ve İndirme (2 dakika)
**Söylenecekler:**
- **Veri Kaynağı**: tennis-data.co.uk
- **Python ile İndirme**: `requests` kütüphanesi kullanıldı
- **Veri Formatı**: Excel dosyaları (.xlsx)
- **Veri Kapsamı**: 2000-2025 yılları arası (25 yıl)

**Python Kod Örneği** (ekranda göster):
```python
# src/data/fetch_data.py
import requests
import pandas as pd

def download_season(year: int) -> pd.DataFrame:
    url = f"https://www.tennis-data.co.uk/{year}/{year}.xlsx"
    response = requests.get(url)
    df = pd.read_excel(response.content, engine='openpyxl')
    return df

def build_allyears_csv():
    all_years = []
    for year in range(2000, 2026):
        df = download_season(year)
        all_years.append(df)
    combined = pd.concat(all_years, ignore_index=True)
    combined.to_csv(RAW_DIR / "allyears.csv", index=False)
```

**GÖRSELL 2'yi göster** - Veri İstatistikleri
- "Yıllara göre maç sayısı grafiği"
- "Veri temizleme öncesi/sonrası karşılaştırması"
- "Zemin dağılımı"

#### 3. Veri Ön İşleme (2 dakika)
**Söylenecekler:**
- **Problem**: Ham veride tutarsızlıklar var
  - Farklı kolon isimleri (Date vs date)
  - Farklı tarih formatları
  - Oyuncu isimlerinde büyük/küçük harf farkları
- **Python Çözümü**: `pandas` ile normalizasyon

**Python Kod Örneği** (ekranda göster):
```python
# src/data/preprocess.py
import pandas as pd

def build_matches_from_allyears():
    df = pd.read_csv("data/raw/allyears.csv")
    
    # Kolon isimlerini normalize et
    df.columns = df.columns.str.lower().str.strip()
    
    # Tarih formatını düzelt
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Oyuncu isimlerini normalize et
    df['playerA_norm'] = df['playerA'].str.lower().str.strip()
    df['playerB_norm'] = df['playerB'].str.lower().str.strip()
    
    # Bahis oranlarından olasılık hesapla
    df['pA_market'] = (1/df['oddsA']) / (1/df['oddsA'] + 1/df['oddsB'])
    
    return df
```

**Çıktı**: `data/processed/matches_allyears.csv`

#### 4. Veri Temizleme (2 dakika)
**Söylenecekler:**
- **Python ile Filtreleme**: `pandas` boolean indexing kullanıldı

**Python Kod Örneği** (ekranda göster):
```python
# src/data/cleaning.py
def build_clean_matches():
    df = pd.read_csv("data/processed/matches_allyears.csv")
    
    # Eksik değerleri kaldır
    df = df.dropna(subset=['playerA', 'playerB', 'date', 'surface', 'winner'])
    
    # Geçersiz bahis oranlarını temizle
    df = df[(df['oddsA'] >= 1.0) & (df['oddsA'] <= 100)]
    df = df[(df['oddsB'] >= 1.0) & (df['oddsB'] <= 100)]
    
    return df
```

**GÖRSELL 15'i göster** - Veri Kalitesi Metrikleri
- "Temizleme öncesi: ~250,000 satır"
- "Temizleme sonrası: ~220,000 satır"
- "%12 veri kaybı (kalite için gerekli)"
- "Eksik değer oranı düşürüldü"
- "Yıllara göre veri dağılımı"
- "Zemin tipi dağılımı"

#### 5. Veri Şeması (1 dakika)
**Söylenecekler:**
- **Standart Veri Yapısı**: `schema.py` içinde `MATCH_COLUMNS` tanımı
- **Önemli Kolonlar**: date, tourney, surface, playerA, playerB, oddsA, oddsB, winner
- **Neden Önemli?**: Diğer modüller bu şemaya bağımlı

**Python Kod Örneği** (ekranda göster):
```python
# src/data/schema.py
MATCH_COLUMNS = [
    'date', 'tourney', 'surface', 'round',
    'playerA', 'playerB', 'rankA', 'rankB',
    'oddsA', 'oddsB', 'winner',
    'playerA_norm', 'playerB_norm'
]
```

#### 6. Sonuç ve Geçiş (30 saniye)
**Söylenecekler:**
- "Temiz, normalize edilmiş veri hazır"
- "Bir sonraki aşama: Feature Engineering"
- "[Kişi 2]'ye sözü veriyorum"

### 🎤 Sunum İpuçları:
- **Python vurgusu**: Her adımda Python kodunu göster
- **Görseller**: GÖRSELL 2 ve GÖRSELL 15'i kullan
- **Kod gösterimi**: En önemli 2-3 fonksiyonu kısaca göster
- **Zamanlama**: Maksimum 9 dakika

---

## 🎯 BÖLÜM 2: FEATURE ENGINEERING (8-9 dakika)

### 👤 Sorumlu: **Kişi 2 - Feature Engineer**

### 📝 Detaylı Sunum İçeriği:

#### 1. Giriş (30 saniye)
**Söylenecekler:**
- "Ben [İsim], feature engineering aşamasından sorumluyum"
- "Bu aşamada, Python ile ~47 feature ürettik"
- "Feature'lar, modelin tahmin yapabilmesi için gerekli bilgileri içerir"

**GÖRSELL 3'ü göster** - Feature Kategorileri Tablosu
- "Toplam 47 feature ürettik"
- "9 farklı kategoride feature'lar var"

#### 2. Elo Rating Sistemi (2 dakika)
**Söylenecekler:**
- **Konsept**: Satranç'tan uyarlanmış rating sistemi
- **Python ile Hesaplama**: `collections.defaultdict` kullanıldı

**Matematiksel Temel**:
- Beklenen skor: `E_A = 1 / (1 + 10^((R_B - R_A) / 400))`
- Rating güncelleme: `R_new = R_old + K * (actual_score - expected_score)`
- **İki tür Elo**:
  - Global Elo (K=32): Tüm maçlar için
  - Surface Elo (K=24): Zemin bazlı (Hard, Clay, Grass, Carpet)

**Python Kod Örneği** (ekranda göster):
```python
# src/features/elo.py
from collections import defaultdict
import numpy as np

def expected_score(r_a: float, r_b: float) -> float:
    """Beklenen skoru hesapla"""
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

def compute_elo_for_matches():
    global_elo = defaultdict(lambda: 1500.0)
    surface_elo = defaultdict(lambda: defaultdict(lambda: 1500.0))
    
    for match in matches_sorted_by_date:
        # Global Elo güncelleme
        e_a = expected_score(global_elo[playerA], global_elo[playerB])
        actual = 1.0 if winner == 'A' else 0.0
        global_elo[playerA] += 32 * (actual - e_a)
        global_elo[playerB] += 32 * ((1-actual) - (1-e_a))
        
        # Surface Elo güncelleme (benzer mantık, K=24)
```

**GÖRSELL 4'ü göster** - Elo Rating Örneği
- "Bir oyuncunun Elo'sunun zaman içinde nasıl değiştiğini gösteren grafik"
- "Global Elo ve Surface Elo karşılaştırması"

**Üretilen Feature'lar**: `eloA`, `eloB`, `elo_diff`, `elo_surfaceA`, `elo_surfaceB`, `elo_surface_diff` (6 feature)

#### 3. Form Features (1.5 dakika)
**Söylenecekler:**
- **Python ile Hesaplama**: `collections.deque` ile sliding window

**Feature Kategorileri**:
1. **Son N Maç Kazanma Oranı**: `form_winrateA_5`, `form_winrateB_5`, `form_winrateA_10`, `form_winrateB_10`
2. **Dinlenme Süresi**: `days_since_lastA`, `days_since_lastB` (0-365 gün arası kırpılır)
3. **Maç Yoğunluğu**: `matches_last30A`, `matches_last30B` (son 30 gündeki maç sayısı)

**Python Kod Örneği** (ekranda göster):
```python
# src/features/form.py
from collections import deque, defaultdict
import pandas as pd
import numpy as np

def compute_form_features():
    player_recent_matches = defaultdict(lambda: deque(maxlen=10))
    
    for match in matches_sorted_by_date:
        # Son 5 maç kazanma oranı
        recent_5 = list(player_recent_matches[playerA])[-5:]
        winrate_5 = sum(recent_5) / len(recent_5) if recent_5 else 0.5
        
        # Dinlenme süresi
        days_since = (match['date'] - last_match_date[playerA]).days
        days_since_clipped = np.clip(days_since, 0, 365)
        
        # Son 30 gündeki maç sayısı
        matches_30d = count_matches_in_last_30_days(playerA, match['date'])
```

**GÖRSELL 5'i göster** - Form Features Örneği
- "Son 5 maç kazanma oranı grafiği"
- "Dinlenme süresi ve maç yoğunluğu"
- "Birleşik form skoru"

**Üretilen Feature'lar**: 8 feature (4 kazanma oranı + 2 dinlenme + 2 yoğunluk)

#### 4. H2H (Head-to-Head) Features (1 dakika)
**Söylenecekler:**
- **Amaç**: İki oyuncu arasındaki geçmiş karşılaşmaları yansıtır
- **Python ile Hesaplama**: Zaman bazlı filtreleme

**GÖRSELL 11'i göster** - H2H Örneği
- "Federer vs Nadal karşılaşma geçmişi"
- "Kazanma/kayıp sayıları ve kazanma oranı"

**Üretilen Feature'lar**: `h2h_matches`, `h2h_winrateA`, `h2h_winrateB`, `h2h_winrate_diff` (4 feature)

#### 5. Market Features (1 dakika)
**Söylenecekler:**
- **Amaç**: Bahis şirketlerinin görüşünü yansıtır (baseline)
- **Python ile Hesaplama**: Bahis oranlarından implied probability

**GÖRSELL 12'yi göster** - Market Features Analizi
- "Bahis oranları dağılımı"
- "Implied probability dağılımı"
- "Logit transformasyonu"
- "Market vs Model olasılıkları karşılaştırması"

**Üretilen Feature'lar**: `pA_market`, `pB_market`, `p_diff`, `logit_pA_market` (4 feature)

#### 6. Surface Features (1 dakika)
**Söylenecekler:**
- **Amaç**: Zemin bazlı performans farklarını yakalar
- **Python ile Hesaplama**: Surface Elo ve zemin bazlı istatistikler

**GÖRSELL 13'ü göster** - Zemin Bazlı Performans
- "Surface Elo karşılaştırması (Hard, Clay, Grass, Carpet)"
- "Zemin bazlı kazanma oranı"

**Üretilen Feature'lar**: Surface bazlı Elo ve performans metrikleri

#### 7. Set Features ve Diğer Features (1 dakika)
**Söylenecekler:**
- **Set Features**: Set bazlı performans istatistikleri
- **Round/Tournament Features**: Turnuva önemi
- **Rank Features**: ATP sıralaması

**Python Kod Örneği** (ekranda göster):
```python
# src/features/build_features.py
def add_h2h_features(df):
    # Her maç için, bu maça kadar olan karşılaşmaları say
    h2h_stats = compute_h2h_before_match(df)
    df = df.merge(h2h_stats, on=['playerA', 'playerB', 'date'])
    return df

def add_tournament_round_features(df):
    df['round_importance'] = df['round'].map(ROUND_IMPORTANCE_MAP)
    df['is_final'] = (df['round'] == 'F').astype(int)
    return df
```

**Üretilen Feature'lar**: 
- Set Features: 9 feature
- Round/Tournament Features: 7 feature
- Rank Features: 3 feature

#### 8. Feature Correlation (30 saniye)
**GÖRSELL 17'yi göster** - Feature Correlation Heatmap
- "Feature'lar arası ilişki analizi"
- "Yüksek korelasyonlu feature'ları gösterir"
- "Model seçimi için önemli"

#### 9. Feature Importance (30 saniye)
**GÖRSELL 9'u göster** - Feature Importance
- "En önemli 10 feature"
- "Model katsayılarına göre sıralama"

#### 10. Sonuç ve Geçiş (30 saniye)
**Söylenecekler:**
- "Toplam 47 feature üretildi"
- "Tüm feature'lar Python ile hesaplandı"
- "Model eğitimi için hazır"
- "Bir sonraki aşama: Model Eğitimi"
- "[Kişi 3]'e sözü veriyorum"

### 🎤 Sunum İpuçları:
- **Görseller**: GÖRSELL 3, 4, 5, 9, 11, 12, 13, 17'yi kullan
- **Python vurgusu**: Her feature'ın Python kodunu göster
- **Zamanlama**: Maksimum 9 dakika

---

## 🎯 BÖLÜM 3: MODEL EĞİTİMİ VE DEĞERLENDİRME (8-9 dakika)

### 👤 Sorumlu: **Kişi 3 - ML Engineer**

### 📝 Detaylı Sunum İçeriği:

#### 1. Giriş (30 saniye)
**Söylenecekler:**
- "Ben [İsim], model eğitimi ve değerlendirme aşamasından sorumluyum"
- "Python'da scikit-learn kullanarak Logistic Regression modelini eğittik"
- "Model, market ile başa baş performans gösteriyor"

#### 2. Model Seçimi (1.5 dakika)
**Söylenecekler:**
- **Neden Logistic Regression?**
  - Basit ve yorumlanabilir
  - Python'da scikit-learn ile kolay implementasyon
  - Overfitting riski düşük
  - Market ile başa baş performans
- **Alternatif Modeller**: XGBoost, Random Forest (daha karmaşık, overfitting riski yüksek)

**Python Kod Örneği** (ekranda göster):
```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        penalty='l2',      # Ridge regularization
        C=1.0,             # Regularization gücü
        solver='lbfgs',    # Optimizasyon algoritması
        max_iter=1000      # Maksimum iterasyon
    )
)
```

#### 3. Veri Hazırlama ve Eğitim Süreci (2 dakika)
**Söylenecekler:**
- **Zaman bazlı split**: 2022 öncesi → train, 2022+ → validation
- **Neden zaman bazlı?**: Gelecek tahminleri için daha gerçekçi
- **Eksik Değer Doldurma**: `SimpleImputer` kullanıldı
- **Scaling**: `StandardScaler` ile feature'lar normalize edildi

**GÖRSELL 14'ü göster** - Model Eğitim Süreci
- "Zaman bazlı train/validation split"
- "Feature sayısı evrimi"
- "Model eğitim metrikleri"
- "Pipeline adımları"

**Python Kod Örneği** (ekranda göster):
```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Zaman bazlı split
train_df = df[df['date'] < '2022-01-01']
val_df = df[df['date'] >= '2022-01-01']

# Feature ve meta kolonlarını ayır
feature_cols = [col for col in df.columns if col not in META_COLS]
X_train = train_df[feature_cols]
y_train = train_df['y']
X_val = val_df[feature_cols]
y_val = val_df['y']

# Eksik değer doldurma
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Model eğitimi
model.fit(X_train_imputed, y_train)
```

**Çıktılar**:
- `models/logreg_final.pkl`: Eğitilmiş model
- `models/imputer_final.pkl`: Eksik değer doldurucu
- `models/feature_columns.txt`: Kullanılan feature listesi

#### 4. Model Değerlendirme (3 dakika)
**Söylenecekler:**
- **Metrikler**: LogLoss, Brier Score, Accuracy
- **Python ile Hesaplama**: `sklearn.metrics` kullanıldı

**Python Kod Örneği** (ekranda göster):
```python
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# Tahmin
y_pred_proba = model.predict_proba(X_val_imputed)[:, 1]
y_pred = model.predict(X_val_imputed)

# Metrikleri hesapla
logloss = log_loss(y_val, y_pred_proba)
brier = brier_score_loss(y_val, y_pred_proba)
accuracy = accuracy_score(y_val, y_pred)
```

**GÖRSELL 6'yı göster** - Model vs Market Metrikleri
- "LogLoss, Brier Score, Accuracy karşılaştırması"
- **Model Metrikleri**:
  - LogLoss: 0.5859
  - Brier Score: 0.2012
  - Accuracy: 68.06%
- **Market Metrikleri**:
  - LogLoss: 0.5852
  - Brier Score: 0.2011
  - Accuracy: 67.99%
- **Sonuç**: Başa baş performans!

**GÖRSELL 7'yi göster** - Confusion Matrix
- "Model ve Market confusion matrix karşılaştırması"
- "Doğru tahminler ve hatalar"

**GÖRSELL 10'u göster** - Edge Dağılımı
- "Model - Market farkı (edge) dağılımı"
- "Pozitif edge: Model market'i geçiyor"
- "Negatif edge: Market model'i geçiyor"

#### 5. Tüm Maçlara Tahmin (1 dakika)
**Söylenecekler:**
- **Amaç**: Tüm geçmiş maçlara model tahmini yapmak
- **Python ile**: `score_all_matches.py` modülü

**Python Kod Örneği** (ekranda göster):
```python
# src/models/score_all_matches.py
import joblib
import pandas as pd

model = joblib.load(MODELS_DIR / "logreg_final.pkl")
imputer = joblib.load(MODELS_DIR / "imputer_final.pkl")

for match in matches:
    features = extract_features(match)
    features_imputed = imputer.transform([features])
    p_model = model.predict_proba(features_imputed)[0, 1]
    
    if 'pA_market' in match:
        edge = p_model - match['pA_market']
```

**Çıktı**: `data/processed/all_predictions.csv`

#### 6. Sonuç ve Geçiş (30 saniye)
**Söylenecekler:**
- "Model başarıyla eğitildi"
- "Market ile başa baş performans"
- "Bir sonraki aşama: Tahmin ve Arayüz"
- "[Kişi 4]'e sözü veriyorum"

### 🎤 Sunum İpuçları:
- **Görseller**: GÖRSELL 6, 7, 10, 14'ü kullan
- **Python vurgusu**: scikit-learn kodlarını göster
- **Metrikler**: Gerçek değerleri vurgula
- **Zamanlama**: Maksimum 9 dakika

---

## 🎯 BÖLÜM 4: TAHMİN VE ARAYÜZ (8-9 dakika)

### 👤 Sorumlu: **Kişi 4 - Frontend Developer**

### 📝 Detaylı Sunum İçeriği:

#### 1. Giriş (30 saniye)
**Söylenecekler:**
- "Ben [İsim], tahmin ve arayüz aşamasından sorumluyum"
- "Python'da Streamlit kullanarak interaktif web arayüzü geliştirdik"
- "Kullanıcılar web arayüzü ile tahmin yapabilir"

#### 2. What-if Tahminleri (2 dakika)
**Söylenecekler:**
- **Python ile Senaryo Bazlı Tahmin**: `whatif.py` modülü
- **Kullanım Senaryoları**: 
  - "Eğer Federer ve Nadal 2020'de Hard court'ta karşılaşsaydı, sonuç ne olurdu?"
  - Senaryo bazlı analizler

**Python Kod Örneği** (ekranda göster):
```python
# src/predict/whatif.py
def predict_single_match(playerA, playerB, surface, date):
    # Oyuncu snapshot'larını al
    snapshotA = get_player_snapshot(playerA, date)
    snapshotB = get_player_snapshot(playerB, date)
    
    # H2H hesapla
    h2h = compute_h2h(playerA, playerB, date)
    
    # Feature vektörü oluştur
    features = build_feature_row(snapshotA, snapshotB, surface, h2h)
    
    # Tahmin yap
    p_model = model.predict_proba([features])[0, 1]
    return p_model
```

**CLI Kullanımı** (ekranda göster):
```bash
py -m src.predict.whatif --playerA "Roger Federer" --playerB "Rafael Nadal" --surface "Hard" --date "2020-01-15"
```

#### 3. Streamlit Web Arayüzü (5 dakika)
**Söylenecekler:**
- **Streamlit**: Python ile hızlı web uygulaması geliştirme
- **3 Ana Sekme**: Matches, What-if, Leaderboard

**GÖRSELL 16'yı göster** - Streamlit UI Özellikleri
- "3 ana sekme: Matches, What-if, Leaderboard"
- "Her sekmenin özellikleri"
- "Interaktif özellikler"

**Python Kod Örneği** (ekranda göster):
```python
# streamlit_app.py
import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Predictive Serve", layout="wide")
st.title("Predictive Serve")

tab1, tab2, tab3 = st.tabs(["Matches", "What-if", "Leaderboard"])

with tab1:
    # Maç filtreleme
    date_range = st.date_input("Tarih Aralığı", value=[])
    surface_filter = st.selectbox("Zemin", ["Hepsi", "Hard", "Clay", "Grass"])
    player_filter = st.text_input("Oyuncu Ara")
    
    # Filtrelenmiş veriyi göster
    filtered_df = df[(df['playerA'].str.contains(player_filter))]
    st.dataframe(filtered_df)

with tab2:
    # What-if tahmini
    playerA = st.text_input("Oyuncu A")
    playerB = st.text_input("Oyuncu B")
    surface = st.selectbox("Zemin", ["Hard", "Clay", "Grass"])
    date = st.date_input("Tarih")
    
    if st.button("Tahmin Yap"):
        prediction = predict_single_match(playerA, playerB, surface, date)
        st.write(f"Kazanma Olasılığı: {prediction:.2%}")

with tab3:
    # Leaderboard
    min_matches = st.slider("Minimum Maç Sayısı", 0, 100, 10)
    leaderboard = compute_leaderboard(min_matches)
    st.dataframe(leaderboard)
```

**CANLI DEMO** (ekranda göster - 2-3 dakika):
1. **Streamlit uygulamasını aç**
2. **Matches sekmesi**:
   - Bir maç ara (örn: "Federer")
   - Filtreleme yap
   - Sonuçları göster
3. **What-if sekmesi**:
   - Oyuncu seç (örn: "Roger Federer" vs "Rafael Nadal")
   - Tarih ve zemin seç
   - "Tahmin Yap" butonuna tıkla
   - Sonucu göster
4. **Leaderboard sekmesi**:
   - Minimum maç sayısı filtresi ayarla
   - Sıralamayı göster

#### 4. Arayüz Özellikleri (1 dakika)
**Söylenecekler:**
- **Filtreleme**: Tarih, zemin, oyuncu, turnuva
- **Görselleştirme**: Grafikler, tablolar
- **Interaktif Widget'lar**: Slider, selectbox, text input
- **Model vs Market Karşılaştırması**: Edge hesaplama ve gösterim

#### 5. Sonuç ve Geçiş (30 saniye)
**Söylenecekler:**
- "Python ile end-to-end sistem tamamlandı"
- "Kullanıcılar web arayüzü ile tahmin yapabilir"
- "Şimdi proje sonuçlarına geçiyoruz"

### 🎤 Sunum İpuçları:
- **Canlı Demo**: Mutlaka canlı demo yap (en önemli kısım)
- **Python vurgusu**: Streamlit kodlarını göster
- **Hazırlık**: Demo öncesi mutlaka test et
- **Zamanlama**: Maksimum 9 dakika

---

## 🎯 BÖLÜM 5: SONUÇ VE SORU-CEVAP (3-4 dakika)

### 👥 Sorumlu: **Tüm Ekip** (Herkes 30-45 saniye)

### 📝 Detaylı Sunum İçeriği:

#### 1. Proje Özeti (1 dakika) - **Kişi 1**
**Söylenecekler:**
- "Projemiz başarıyla tamamlandı"
- "Python ile end-to-end pipeline oluşturduk"
- "Veri toplama, temizleme, feature engineering, model eğitimi ve arayüz geliştirme aşamalarını tamamladık"

#### 2. Başarılar (1 dakika) - **Kişi 2**
**Söylenecekler:**
- **Başarılar**:
  - 47 feature üretildi
  - Model, market ile başa baş performans (68.06% vs 67.99%)
  - Streamlit ile kullanıcı dostu arayüz
  - Python ekosistemi ile hızlı geliştirme

#### 3. Zorluklar ve Çözümler (1 dakika) - **Kişi 3**
**Söylenecekler:**
- **Zorluklar**:
  - Veri tutarsızlıkları → pandas ile çözüldü
  - Feature engineering karmaşıklığı → Modüler Python yapısı
  - Model seçimi → scikit-learn ile test edildi
  - Arayüz geliştirme → Streamlit ile hızlı çözüm

#### 4. Gelecek Çalışmalar ve Soru-Cevap (1 dakika) - **Kişi 4**
**Söylenecekler:**
- **Gelecek Çalışmalar**:
  - XGBoost, Neural Network gibi daha karmaşık modeller
  - Daha fazla feature
  - Gerçek zamanlı tahminler
- "Sorularınızı bekliyoruz"

**Olası Sorular ve Cevaplar** (Tüm ekip hazırlıklı olmalı):
1. **"Neden Python?"**
   - Zengin kütüphane ekosistemi (pandas, scikit-learn, streamlit)
   - Hızlı prototipleme
   - Kolay öğrenilebilir

2. **"Neden Logistic Regression?"**
   - Basit ve yorumlanabilir
   - scikit-learn ile kolay implementasyon
   - Market ile başa baş performans

3. **"En önemli feature'lar?"**
   - Elo rating, form, market features

4. **"Model market'i geçebildi mi?"**
   - Bazı durumlarda evet, genel olarak başa baş

### 🎤 Sunum İpuçları:
- **Enerji**: Sonuç bölümünü enerjik bitirin
- **Python vurgusu**: Python'un avantajlarını vurgula
- **Takım Çalışması**: Herkes sorulara cevap verebilir

---

## 📝 SUNUM HAZIRLIK KONTROL LİSTESİ

### 👤 Kişi 1 (Veri Mühendisi) İçin:
- [ ] Veri toplama ve temizleme sunumu hazır
- [ ] GÖRSELL 2 ve GÖRSELL 15 hazır
- [ ] Python kod örnekleri hazır (`fetch_data.py`, `preprocess.py`, `cleaning.py`)
- [ ] Proje tanıtımı kısmı hazır (30 saniye)
- [ ] Sonuç kısmı hazır (30 saniye)
- [ ] Sunum süresi test edildi (toplam ~9 dakika)

### 👤 Kişi 2 (Feature Engineer) İçin:
- [ ] Feature Engineering sunumu hazır
- [ ] GÖRSELL 3, 4, 5, 9, 11, 12, 13, 17 hazır
- [ ] Python kod örnekleri hazır (`elo.py`, `form.py`, `build_features.py`)
- [ ] Proje tanıtımı kısmı hazır (30 saniye)
- [ ] Sonuç kısmı hazır (30 saniye)
- [ ] Sunum süresi test edildi (toplam ~9 dakika)

### 👤 Kişi 3 (ML Engineer) İçin:
- [ ] Model eğitimi sunumu hazır
- [ ] GÖRSELL 6, 7, 10, 14 hazır
- [ ] Python kod örnekleri hazır (`train_logreg.py`, `metrics.py`)
- [ ] Metrik değerleri hazır (gerçek değerler)
- [ ] Proje tanıtımı kısmı hazır (30 saniye)
- [ ] Sonuç kısmı hazır (30 saniye)
- [ ] Sunum süresi test edildi (toplam ~9 dakika)

### 👤 Kişi 4 (Frontend Developer) İçin:
- [ ] Tahmin ve arayüz sunumu hazır
- [ ] Streamlit demo hazır ve test edildi
- [ ] GÖRSELL 16 hazır
- [ ] Python kod örnekleri hazır (`whatif.py`, `streamlit_app.py`)
- [ ] Proje tanıtımı kısmı hazır (30 saniye)
- [ ] Sonuç kısmı hazır (30 saniye)
- [ ] Sunum süresi test edildi (toplam ~9 dakika)

### 👥 Tüm Ekip İçin:
- [ ] Görseller hazır (`presentation_visuals/` klasöründe - 18 görsel)
- [ ] Sunum sırası belirlendi
- [ ] Geçişler planlandı (bir kişiden diğerine)
- [ ] Soru-cevap için hazırlık yapıldı
- [ ] Sunum öncesi prova yapıldı

---

## 🎤 SUNUM İPUÇLARI (Genel)

### ⏰ Zamanlama:
1. Her kişi kendi süresine dikkat etmeli (8-9 dakika)
2. Fazla detaya girmeyin, önemli noktaları vurgulayın
3. Geçişler hızlı ve akıcı olmalı

### 💻 Python Kod Gösterimi:
1. Sadece en önemli kısımları göster
2. Detaya girmeyin, yüksek seviyede açıklayın
3. Kod örnekleri kısa ve anlaşılır olsun

### 📊 Görseller:
1. Her görseli mutlaka kullan (18 görsel)
2. Görselleri ekranda gösterirken açıkla
3. Görseller sunumu daha etkili yapar

### 🎬 Canlı Demo:
1. Mutlaka canlı demo yapın (Kişi 4)
2. Demo öncesi mutlaka test edin
3. Hataları önceden düzeltin

### ❓ Soru-Cevap:
1. Zor sorulara hazırlıklı olun
2. Takım çalışması: Herkes sorulara cevap verebilir
3. Python'un avantajlarını vurgulayın

---

## 🎯 SUNUM AKIŞI ÖZETİ

```
1. Proje Tanıtımı (2-3 dk)
   ├─ Kişi 1: Açılış (30 sn) - GÖRSELL 1
   ├─ Kişi 2: Problem Tanımı (30 sn)
   ├─ Kişi 3: Proje Yapısı (30 sn) - GÖRSELL 8, 18
   └─ Kişi 4: Ekip Tanıtımı + Geçiş (30-45 sn)

2. Veri Toplama ve Temizleme (8-9 dk) - Kişi 1
   ├─ Veri Kaynağı ve İndirme (2 dk) - GÖRSELL 2
   ├─ Veri Ön İşleme (2 dk)
   ├─ Veri Temizleme (2 dk) - GÖRSELL 15
   ├─ Veri Şeması (1 dk)
   └─ Sonuç + Geçiş (30 sn)

3. Feature Engineering (8-9 dk) - Kişi 2
   ├─ Giriş (30 sn) - GÖRSELL 3
   ├─ Elo Rating Sistemi (2 dk) - GÖRSELL 4
   ├─ Form Features (1.5 dk) - GÖRSELL 5
   ├─ H2H Features (1 dk) - GÖRSELL 11
   ├─ Market Features (1 dk) - GÖRSELL 12
   ├─ Surface Features (1 dk) - GÖRSELL 13
   ├─ Set Features ve Diğer (1 dk)
   ├─ Feature Correlation (30 sn) - GÖRSELL 17
   ├─ Feature Importance (30 sn) - GÖRSELL 9
   └─ Sonuç + Geçiş (30 sn)

4. Model Eğitimi ve Değerlendirme (8-9 dk) - Kişi 3
   ├─ Giriş (30 sn)
   ├─ Model Seçimi (1.5 dk)
   ├─ Veri Hazırlama ve Eğitim (2 dk) - GÖRSELL 14
   ├─ Model Değerlendirme (3 dk) - GÖRSELL 6, 7, 10
   ├─ Tüm Maçlara Tahmin (1 dk)
   └─ Sonuç + Geçiş (30 sn)

5. Tahmin ve Arayüz (8-9 dk) - Kişi 4
   ├─ Giriş (30 sn)
   ├─ What-if Tahminleri (2 dk)
   ├─ Streamlit Web Arayüzü (5 dk) - GÖRSELL 16, Canlı Demo
   ├─ Arayüz Özellikleri (1 dk)
   └─ Sonuç (30 sn)

6. Sonuç ve Soru-Cevap (3-4 dk) - Tüm Ekip
   ├─ Kişi 1: Proje Özeti (30 sn)
   ├─ Kişi 2: Başarılar (30 sn)
   ├─ Kişi 3: Zorluklar ve Çözümler (30 sn)
   └─ Kişi 4: Gelecek Çalışmalar + Soru-Cevap (1-2 dk)
```

---

**Başarılar! 🎉**

**Son Güncelleme**: [Tarih]  
**Hazırlayan**: Ekip  
**Toplam Görsel Sayısı**: 18

