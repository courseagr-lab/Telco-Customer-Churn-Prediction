# 📡 Telco Customer Churn Prediction & Retention Dashboard 2026

Project end-to-end data science untuk memprediksi churn pelanggan
telekomunikasi Indonesia, dilengkapi dashboard interaktif Streamlit untuk
tim retensi.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.77-success)

---

## 🎯 Business Problem

Industri telekomunikasi Indonesia 2026 menghadapi persaingan ketat
(Telkomsel, Indosat, XL, Smartfren, dan provider satelit baru seperti
Starlink). Churn rate yang tidak terkelola menyebabkan kehilangan revenue
signifikan — cost akuisisi pelanggan baru 5-25x lebih mahal dari retensi
pelanggan eksisting.

**Tim retensi tidak memiliki visibility** pelanggan mana yang berisiko
churn, sehingga budget promo/diskon tidak terdistribusi efektif.

## 💡 Solution

Model **XGBoost Classifier** memprediksi probabilitas churn per pelanggan
berdasarkan 17 fitur (demografi, layanan, perilaku, kepuasan). Output
probabilitas di-segmentasi menjadi **3 tier risiko** (High/Medium/Low),
masing-masing dengan rekomendasi aksi retensi spesifik. Seluruh insight
disajikan via **dashboard Streamlit interaktif**.

---

## 📊 Hasil Utama

| Metrik | Nilai |
|---|---|
| ROC-AUC (test set) | **0.771** |
| ROC-AUC (5-fold CV) | **0.773 ± 0.007** |
| Recall (churn) | 0.533 |
| Top predictor | `skor_kepuasan_csat` (27.9% importance) |
| Segmen High Risk | 12.6% populasi → 73.6% actual churn rate |
| Segmen Low Risk | 63.3% populasi → 1.9% actual churn rate |
| Business impact (simulasi) | ROI +91.3%, payback ~3.1 bulan |

📄 Laporan lengkap: [`reports/business_insight.md`](reports/business_insight.md)

---

## 📁 Struktur Project

```
telco-churn-2026/
├── data/
│   ├── raw/telco_customer_churn.csv       # Dataset sintetis (7,010 baris)
│   ├── processed/telco_clean.csv          # Setelah cleaning (7,000 baris)
│   ├── processed/telco_features.csv       # Setelah feature engineering (24 kolom)
│   └── README_data.md                     # Dokumentasi skema dataset
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # EDA awal, cleaning, validasi
│   ├── 02_eda.ipynb                       # Univariate/bivariate/korelasi
│   ├── 03_feature_engineering.ipynb       # Feature creation, encoding, scaling
│   └── 04_modeling.ipynb                  # Training, evaluasi, SHAP, business impact
├── src/                                    # Modul reusable (single source of truth)
│   ├── utils.py                           # Path config, logging, konstanta bisnis
│   ├── data_processing.py                 # Generate data + cleaning pipeline
│   ├── feature_engineering.py             # Feature creation + encoding + scaling
│   └── model.py                           # Training, evaluasi, inference
├── models/                                 # Artifact hasil training
│   ├── churn_model_xgb.pkl
│   ├── encoder.pkl
│   ├── scaler.pkl
│   ├── feature_list.pkl
│   └── metrics.pkl
├── app/                                    # Dashboard Streamlit
│   ├── streamlit_app.py                   # Halaman utama (KPI overview)
│   └── pages/
│       ├── 1_📊_Overview.py               # Statistik agregat & distribusi
│       ├── 2_🔍_Customer_Detail.py        # Cari & lihat profil pelanggan
│       ├── 3_🎯_Prediksi_Churn.py         # Form prediksi pelanggan baru
│       └── 4_💡_Rekomendasi.py            # Feature importance & business impact
├── reports/business_insight.md             # Laporan bisnis lengkap
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
git clone <repo-url>
cd telco-churn-2026

python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

### 2. Jalankan Pipeline Data → Model

Pipeline harus dijalankan **berurutan** (setiap langkah menghasilkan
artifact yang dipakai langkah berikutnya):

```bash
python -m src.data_processing       # -> data/raw/, data/processed/telco_clean.csv
python -m src.feature_engineering   # -> data/processed/telco_features.csv, models/encoder.pkl, models/scaler.pkl
python -m src.model                 # -> models/churn_model_xgb.pkl, models/metrics.pkl
```

Dengan `random_state=42`, hasil **reproducible** — setiap eksekusi ulang
akan menghasilkan dataset dan model yang identik.

### 3. (Opsional) Eksplorasi via Jupyter Notebook

```bash
jupyter notebook notebooks/
```

Jalankan `01` → `02` → `03` → `04` secara berurutan. Setiap notebook
sudah berisi output (plot, tabel, insight) dari eksekusi sebelumnya —
bisa langsung dibaca tanpa re-run, atau jalankan ulang untuk verifikasi.

### 4. Jalankan Dashboard

```bash
streamlit run app/streamlit_app.py
```

Buka browser ke `http://localhost:8501`.

---

## 🖥️ Halaman Dashboard

### 🏠 Home
KPI cards (total pelanggan, churn rate, ARPU, tenure) + quick charts.

### 📊 Overview
Filter interaktif (kota, paket, jaringan) dengan 4 tab analisis:
demografi, layanan, perilaku pelanggan (komplain × CSAT heatmap), dan
korelasi antar variabel.

### 🔍 Customer Detail
Cari pelanggan via dropdown/search, lihat profil lengkap + gauge chart
probabilitas churn + rekomendasi aksi personal + faktor risiko spesifik.

### 🎯 Prediksi Churn
Form input untuk pelanggan **baru** (belum ada di database) — prediksi
real-time menggunakan model terlatih, lengkap dengan breakdown faktor
risiko dan rekomendasi.

### 💡 Rekomendasi
- **Performa Model**: ROC curve, confusion matrix, CV scores.
- **Feature Importance**: ranking + interpretasi bisnis top-3 fitur.
- **Segmentasi Risiko**: distribusi populasi + validasi monotonic churn rate.
- **Business Impact**: kalkulator interaktif (ARPU, cost promo,
  efektivitas, horizon) dengan waterfall chart net impact.

---

## 🧠 Metodologi

### Dataset
Sintetis, 7.000 pelanggan, dirancang merefleksikan pola industri telco
Indonesia 2026 (5G, e-wallet, streaming bundle). Churn rate ~15.3% —
**imbalanced**, ditangani dengan `scale_pos_weight` pada XGBoost.
Dataset sengaja mengandung *data quality issues* (missing values,
mixed-type column, outlier, duplikat) untuk melatih pipeline cleaning
yang representatif. Detail: [`data/README_data.md`](data/README_data.md).

### Feature Engineering
5 fitur derivatif: `tenure_group` (binning), `usage_per_cost` (value
perception), `high_complain_flag`, `low_csat_flag`, `engagement_score`
(kombinasi tertimbang). Encoding via `LabelEncoder`, scaling via
`StandardScaler` — kedua artifact disimpan untuk konsistensi
training-inference.

### Model
**XGBoost Classifier** (`n_estimators=200, max_depth=5, learning_rate=0.05`),
`scale_pos_weight` otomatis dihitung dari rasio kelas. Evaluasi:
ROC-AUC, classification report, confusion matrix, 5-fold stratified CV,
SHAP (feature importance + waterfall per-individu).

### Validasi Segmentasi
Probabilitas churn → 3 segmen (High ≥70%, Medium 40-69%, Low <40%).
Tervalidasi: actual churn rate meningkat **monoton** dari Low (1.9%) →
Medium (19.9%) → High (73.6%).

---

## 🔄 Reproduksibilitas

Semua langkah menggunakan `random_state=42`. Untuk regenerasi penuh dari
nol:

```bash
rm -rf data/raw data/processed models
mkdir -p data/raw data/processed models

python -m src.data_processing
python -m src.feature_engineering
python -m src.model
```

Hasil (`models/metrics.pkl`, `data/processed/*.csv`) akan **identik**
dengan yang sudah ada di repo.

---

## 🛣️ Roadmap Pengembangan Lanjutan

1. Ganti dataset sintetis dengan data CRM real (anonymized).
2. Survival analysis (waktu-hingga-churn) untuk prioritisasi granular.
3. A/B testing efektivitas program retensi (validasi asumsi 40%).
4. Integrasi API real-time → prediksi otomatis saat data pelanggan update.
5. Deploy ke Streamlit Community Cloud / Hugging Face Spaces.
6. Model monitoring & retraining triwulanan (deteksi data/concept drift).

---

## 📜 Lisensi & Disclaimer

Project ini menggunakan **dataset sintetis** untuk tujuan edukasi/demo
metodologi data science end-to-end. Tidak merepresentasikan data
pelanggan riil perusahaan manapun. Untuk penggunaan produksi, lakukan
retraining dengan data aktual dan validasi ulang seluruh asumsi bisnis
(lihat Bagian 6, [`reports/business_insight.md`](reports/business_insight.md)).

---

## 🤝 Kontribusi & Kontak

Struktur project mengikuti praktik **production-grade**: modul `src/`
sebagai single source of truth (dipakai ulang oleh notebooks & app),
artifact training di-pickle untuk konsistensi inference, dan setiap
fungsi terdokumentasi dengan docstring + logging.

Untuk pertanyaan atau kontribusi, buka issue/PR pada repository ini.

---

*© 2026 — Telco Churn Prediction Project*
