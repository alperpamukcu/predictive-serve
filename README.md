# 🎾 Predictive Serve

**Predictive Serve**, tenis maçlarının sonucunu tahmin etmek için tasarlanmış uçtan uca (end-to-end) bir **Python projesidir**.

Proje:

- Geçmiş tenis maçlarını **tennis-data.co.uk** sitesinden otomatik indirir,
- Bu maçları temizleyip zengin **feature**’lar üretir (Elo, form, dinlenme süresi, head-to-head vs.),
- **Logistic Regression** modeli ile “A oyuncusu kazanır mı?” olasılığını tahmin eder,
- Model tahminlerini **bahis şirketi oranlarından türetilen olasılıklarla** kıyaslar.

Bu repo özellikle **ders projesi / akademik kullanım** için tasarlanmıştır:
Tüm adımlar komut satırından çalıştırılabilir ve faz faz (FAZ 0–3) net şekilde ayrılmıştır.

---

## 🔍 Problem Tanımı

Amaç:

> “Bahis şirketlerinin oranlarına ek olarak oyuncuların formu, Elo rating’i, head-to-head geçmişi ve diğer istatistikleri kullanarak tenis maç sonucu tahmininde ne kadar iyi olabiliriz?”

Bu kapsamda:

- Geçmiş maç verisi → tennis-data.co.uk
- Hedef değişken → `y` (1 = playerA kazandı, 0 = playerB kazandı)
- Model çıktısı → `P(playerA kazanır)`

---

## 🧱 Proje Fazları

### FAZ 0 – Proje İskeleti & Ortam

- Klasör ve modül yapısının oluşturulması:
  - `src/data`, `src/features`, `src/models`, `src/utils`
  - `data/raw`, `data/processed`
  - `notebooks`, `tests`
- `requirements.txt` ile bağımlılıkların belirlenmesi.
- Basit konfigürasyon ve logging yardımcı fonksiyonları:
  - `src/utils/config.py`
  - `src/utils/logging_utils.py`

---

### FAZ 1 – Veri Toplama & Temizleme

**Amaç:** tennis-data.co.uk’teki yıllık Excel’lerden tek bir temiz maç veri seti oluşturmak.

- `src/data/fetch_data.py`
  - 2000–2025 arası yılları dolaşarak:
    - `http://www.tennis-data.co.uk/{year}/{year}.xlsx` adresinden veriyi indirir,
    - Bütün yılları birleştirerek:
      - `data/raw/allyears/allyears.csv` dosyasını üretir.
  - Bağlantı problemi olması durumunda uyarı log’ları yazar (örn. VPN ihtiyacı).

- `src/data/preprocess.py`
  - `allyears.csv` dosyasını okur,
  - Kolon isimlerini ve tiplerini normalize eder,
  - Tarih kolonlarını `datetime` formatına çevirir,
  - Temel temizlikleri yapar ve:
    - `data/processed/matches_allyears.csv` çıktısını üretir.

- `src/data/cleaning.py`
  - Maç verisindeki bariz hataları temizler:
    - Eksik kritik bilgiler,
    - Oransız/bozuk satırlar,
  - Daha güvenilir bir set oluşturur:
    - `data/processed/matches_clean.csv`

---

### FAZ 2 – Feature Engineering

**Amaç:** Maç başına daha anlamlı, model için kullanılabilir değişkenler üretmek.

- `src/features/elo.py`
  - Oyuncular için **Elo rating** hesaplar:
    - Global Elo (`eloA`, `eloB`, `elo_diff`)
    - Zemin bazlı Elo:
      - `elo_surfaceA`, `elo_surfaceB`, `elo_surface_diff`
  - Çıktı: `data/processed/matches_with_elo.csv`

- `src/features/form.py`
  - Kısa vadeli form ve yoğunluk feature’ları:
    - Son 5 maç kazanma oranı: `form_winrateA_5`, `form_winrateB_5`, `form_winrate_diff_5`
    - Son 10 maç kazanma oranı: `form_winrateA_10`, `form_winrateB_10`, `form_winrate_diff_10`
    - Son maçtan bu maça kadar geçen gün sayısı:
      - `days_since_lastA`, `days_since_lastB`
    - Son 30 gündeki maç sayısı:
      - `matches_last30A`, `matches_last30B`
  - Çıktı: `data/processed/matches_with_elo_form.csv`

- `src/features/build_features.py`
  - Tüm feature setini oluşturur:
    - Elo & surface Elo
    - Form/yoğunluk (son 5/10 maç, 30 gün içi maç sayısı)
    - **Head-to-head (H2H)**:
      - `h2h_matches_before`
      - `h2h_winrateA`, `h2h_winrateB`
    - **Bahis oranları** → implied probability:
      - `oddsA`, `oddsB`
      - `pA_market`, `pB_market`, `p_diff`, `logit_pA_market`
    - **Turnuva/round bilgisi**:
      - Örneğin `round` alanından encode edilen feature’lar (tur önemi).
    - Zemin (surface) için one-hot encoding:
      - `surface_Grass`, `surface_Clay`, `surface_Hard`, `surface_Carpet` vb.
  - Meta kolonları (`date`, `surface`, `playerA`, `playerB`, `y`) ile birlikte,
    model eğitimine hazır data setini üretir:
    - `data/processed/train_dataset.csv`

---

### FAZ 3 – Model Eğitimi ve Değerlendirme

**Hedef:** Farklı modelleri ve bahis şirketi tahminlerini kıyaslayıp, final modeli seçmek.

#### Zaman bazlı train/validation split

- `train_dataset.csv` içindeki `date` kolonuna göre:
  - **Train set:** 2022’den önceki maçlar
  - **Validation set:** 2022 ve sonrası maçlar
- Bu sayede:
  - Geçmişe bakarak geleceği tahmin ediyormuşuz gibi daha gerçekçi bir senaryo kuruluyor.

#### Kullanılan modeller

Notebook’ta ( `notebooks/02_train_models.ipynb` ) test edilenler:

- Logistic Regression
- XGBoost
- RandomForest
- Bahis şirketlerinin implied olasılıkları (baseline)

**Ölçülen metrikler:**

- `logloss`   → Olasılık tahmini ne kadar “kalibre” ve doğru?
- `brier_score` → Olasılık karesel hata metriği
- `accuracy` → Doğru/yanlış tahmin oranı

#### Neden Logistic Regression?

Karşılaştırma sonucunda:

- Logistic Regression:
  - Bahis şirketleri ile **neredeyse başa baş** performans veriyor,
  - Oldukça stabil ve yorumlanabilir,
  - XGBoost / RandomForest’a göre:
    - Çok daha basit,
    - Overfit riskini azaltıyor,
    - Ders projesi için anlatması çok daha kolay.

Bu nedenle:

> **Final model** olarak Logistic Regression seçildi.

#### Final eğitim script’i

- `src/models/train_logreg.py`
  - `data/processed/train_dataset.csv` dosyasını okur,
  - Train/validation split uygular,
  - Eksik değerleri `SimpleImputer(strategy="median")` ile doldurur,
  - Logistic Regression modelini eğitir,
  - Validation metriklerini (`logloss`, `brier`, `accuracy`) konsola yazar,
  - Aşağıdaki dosyaları kaydeder:
    - `models/logreg_final.pkl` (eğitilmiş model)
    - `models/imputer_final.pkl` (eksik değer doldurucu)
    - `models/feature_columns.txt` (kullanılan feature isimleri)

---

### FAZ 4 – Tahmin & Arayüz (Planlanan)

Bu faz henüz geliştirme aşamasındadır. Plan:

1. **Prediction script (CLI)**:
   - Komut satırından çalışacak bir araç:
     - Belirli bir maç satırını (veya filtreyi) seçip:
       - Model tahmini,
       - Bahis şirketi tahmini,
       - Gerçek sonucu (geçmiş maçsa) gösterme.

2. **Basit Streamlit arayüzü**:
   - Yıl / turnuva / oyuncu seçilebilen combobox’lar,
   - Grafikli gösterimler:
     - Model vs. market kıyası,
     - Oyuncuların son X maç formu,
     - Head-to-head özetleri,
   - Gelecekte:
     - Kullanıcının **manuel olarak girdiği hayali maçları** (ör. “2009 Federer vs 2024 Alcaraz (hard, best-of-5)”) tahmin etme.

---

## 📂 Klasör Yapısı (Güncel)

Temizlenmiş, şu an kullanılan mimari özetle şöyle:

```text
predictive-serve/
├─ data/
│  ├─ raw/
│  │  └─ allyears/                # tennis-data.co.uk'tan indirilen birleşik veri
│  └─ processed/
│     ├─ matches_allyears.csv     # preprocess sonrası
│     ├─ matches_clean.csv        # cleaning sonrası
│     ├─ matches_with_elo.csv     # elo sonrası
│     ├─ matches_with_elo_form.csv# form feature'ları sonrası
│     └─ train_dataset.csv        # tüm feature seti, model eğitimi için
├─ models/
│  ├─ logreg_final.pkl            # final logistic regression modeli
│  ├─ imputer_final.pkl           # eksik değer doldurma için imputer
│  └─ feature_columns.txt         # modelde kullanılan feature isimleri
├─ notebooks/
│  ├─ 01_eda_matches_allyears.ipynb  # EDA
│  └─ 02_train_models.ipynb          # model karşılaştırma & metrikler
├─ src/
│  ├─ data/
│  │  ├─ fetch_data.py
│  │  ├─ preprocess.py
│  │  └─ cleaning.py
│  ├─ features/
│  │  ├─ elo.py
│  │  ├─ form.py
│  │  └─ build_features.py
│  ├─ models/
│  │  └─ train_logreg.py
│  └─ utils/
│     ├─ config.py
│     └─ logging_utils.py
├─ tests/
│  ├─ test_config.py               # config ile ilgili basit testler
│  └─ test_logger.py               # logging utils testleri
├─ requirements.txt
└─ README.md
