# Business Insight Report
## Customer Churn Prediction — Telco Indonesia 2026

**Tanggal**: Juni 2026
**Model**: XGBoost Classifier
**Dataset**: 7.000 pelanggan (sintetis, representatif pola industri telco Indonesia)

---

## 1. Executive Summary

Model machine learning berhasil dikembangkan untuk memprediksi probabilitas
churn pelanggan dengan **ROC-AUC 0.77** (konsisten pada 5-fold cross-validation:
0.773 ± 0.007). Model mengidentifikasi **skor kepuasan pelanggan (CSAT)** dan
**jumlah komplain** sebagai dua faktor paling dominan — bersama-sama
menyumbang **~47% dari total importance** model.

Segmentasi risiko 3-tingkat (High/Medium/Low) berhasil memisahkan populasi
secara bermakna: pelanggan **High Risk (12.6% populasi)** memiliki actual
churn rate **73.6%**, dibanding **Low Risk (63.3% populasi)** yang hanya
**1.9%**. Validasi ini menjadi dasar kuat untuk prioritisasi program retensi.

Simulasi business impact menunjukkan program retensi tertarget pada segmen
High Risk memiliki **payback period ~3.1 bulan**, dengan ROI **+91.3%**
pada horizon 6 bulan (basis 100.000 pelanggan, asumsi konservatif).

---

## 2. Performa Model

### 2.1 Metrik Evaluasi (Test Set, n=1.400)

| Metrik | Nilai |
|---|---|
| ROC-AUC | **0.7714** |
| ROC-AUC (5-fold CV) | **0.7732 ± 0.0067** |
| Accuracy | 0.7529 |
| Precision (churn) | 0.3167 |
| **Recall (churn)** | **0.5327** |
| F1-score (churn) | 0.3972 |

### 2.2 Confusion Matrix (threshold = 0.5)

| | Predicted: Tidak Churn | Predicted: Churn |
|---|---|---|
| **Actual: Tidak Churn** | 940 (TN) | 246 (FP) |
| **Actual: Churn** | 100 (FN) | 114 (TP) |

### 2.3 Interpretasi Metrik untuk Stakeholder Non-Teknis

- **ROC-AUC 0.77** berarti model memiliki kemampuan diskriminasi yang
  **baik-cukup** (skala umum: 0.5 = random, 1.0 = perfect). Untuk konteks
  churn prediction industri, AUC 0.70-0.80 dianggap **solid dan actionable**.
- **Recall 0.53** berarti dari setiap 100 pelanggan yang benar-benar akan
  churn, model berhasil mengidentifikasi ~53 di antaranya pada threshold
  default (0.5). Sisanya (False Negative, 100 dari 214) lolos deteksi.
- **Mengapa Recall belum lebih tinggi?** Trade-off Precision-Recall pada
  dataset imbalanced (~15% churn) adalah hal yang diharapkan. Solusi
  operasional: **gunakan segmentasi 3-tier**, bukan threshold tunggal —
  pelanggan Medium Risk (yang banyak masuk ke FN pada threshold 0.5) tetap
  mendapat aksi retensi (push notification), sehingga *coverage* efektif
  jauh lebih tinggi dari recall pada threshold 0.5 saja.
- **Cross-validation std rendah (0.0067)** menunjukkan model **stabil**
  dan tidak overfit pada split tertentu — performa dapat diandalkan saat
  diterapkan ke data baru dengan distribusi serupa.

---

## 3. Faktor Pendorong Churn (Feature Importance)

| Rank | Fitur | Importance | Insight |
|---|---|---|---|
| 1 | `skor_kepuasan_csat` | 0.279 | **Driver TERKUAT.** CSAT rendah = sinyal churn paling jelas. |
| 2 | `jumlah_komplain_6bln` | 0.187 | Komplain ≥2x dalam 6 bulan = red flag kuat. |
| 3 | `paket` | 0.061 | Prabayar (tanpa kontrak) = churn struktural lebih tinggi. |
| 4 | `punya_streaming_bundle` | 0.054 | Bundle = "stickiness", menurunkan churn. |
| 5 | `punya_ewallet_linked` | 0.044 | E-wallet linking = integrasi ekosistem, menurunkan churn. |
| 6 | `tenure_bulan` | 0.040 | Pelanggan baru (<6 bulan) = window kritis churn. |
| 7 | `engagement_score` | 0.034 | Engagement rendah = sinyal disengagement bertahap. |
| 8 | `pernah_upgrade_paket` | 0.030 | Riwayat upgrade = sinyal investasi/komitmen pelanggan. |
| 9 | `total_pemakaian_data_gb` | 0.029 | Pemakaian sangat rendah/tinggi berkorelasi dengan churn. |
| 10 | `frekuensi_login_app` | 0.027 | Login jarang = early sign of disengagement. |

### Key Takeaway
**CSAT dan Komplain bersama menyumbang ~47% dari kekuatan prediktif model.**
Ini mengindikasikan bahwa **kualitas layanan dan penanganan komplain**
adalah lever bisnis paling berdampak untuk menurunkan churn — lebih besar
dari faktor harga (`biaya_bulanan` tidak masuk top 10) atau demografi.

---

## 4. Segmentasi Risiko Populasi (Full Dataset, n=7.000)

| Segmen | Jumlah Pelanggan | % Populasi | Actual Churn Rate |
|---|---|---|---|
| 🔴 **High Risk** (prob ≥ 70%) | 885 | 12.6% | **73.6%** |
| 🟡 **Medium Risk** (40-69%) | 1.681 | 24.0% | 19.9% |
| 🟢 **Low Risk** (< 40%) | 4.434 | 63.3% | 1.9% |

### Validasi Segmentasi
Actual churn rate **meningkat monoton** dari Low (1.9%) → Medium (19.9%) →
High (73.6%). Ini membuktikan model **dapat dipercaya** untuk
memprioritaskan target program retensi — segmen High Risk benar-benar
berisi pelanggan dengan risiko churn ~38x lebih tinggi dibanding segmen
Low Risk.

> **Catatan metodologi**: Angka di atas dihitung pada seluruh dataset
> (termasuk data yang digunakan untuk training), sehingga merepresentasikan
> *profil populasi* untuk keperluan operasional. Untuk klaim performa
> model yang ketat (generalisasi ke data baru), rujuk Bagian 2 (test set).

---

## 5. Rekomendasi Aksi per Segmen

### 🔴 High Risk (885 pelanggan, 12.6% — prioritas tertinggi)
**Aksi**: Kontak personal CS dalam 48 jam. Tawarkan voucher data 10GB +
diskon 20% untuk 1 bulan berikutnya. Eskalasi ke tim retensi senior jika
tenure < 6 bulan (kombinasi high-risk + new customer = critical).

**Mengapa**: 7 dari 10 pelanggan di segmen ini akan churn jika tidak ada
intervensi. Cost of inaction sangat tinggi.

### 🟡 Medium Risk (1.681 pelanggan, 24.0% — secondary net)
**Aksi**: Push notification personalisasi dengan tawaran upgrade ke paket
bundle streaming. Monitor CSAT pada interaksi berikutnya — jika turun ≤2,
upgrade otomatis ke High Risk handling.

**Mengapa**: ~1 dari 5 pelanggan di segmen ini akan churn — signifikan
secara absolut (1.681 × 19.9% ≈ 335 pelanggan) meski rate-nya moderate.

### 🟢 Low Risk (4.434 pelanggan, 63.3% — maintain & grow)
**Aksi**: Program loyalty points & cross-sell e-wallet linking (bagi yang
belum). Tidak perlu intervensi mendesak — fokus pada *upsell* dan
*advocacy* (referral program).

**Mengapa**: Risiko churn sangat rendah (1.9%) — resource lebih efektif
dialokasikan ke segmen lain, namun tetap jaga *engagement* jangka panjang.

---

## 6. Estimasi Business Impact

### Asumsi Simulasi (Ilustratif)
| Parameter | Nilai |
|---|---|
| ARPU (Average Revenue Per User) bulanan | Rp 100.000 |
| Biaya promo retensi per pelanggan (one-time) | Rp 50.000 |
| Efektivitas program retensi | 40% |
| Horizon revenue yang dihitung | 6 bulan |
| Basis pelanggan (skala simulasi) | 100.000 |

### Hasil Simulasi (Skala 100.000 Pelanggan)

| Metrik | Nilai |
|---|---|
| Pelanggan High Risk | 9.857 |
| ...yang benar-benar akan churn | 3.928 |
| ...berhasil diretensi (40% efektivitas) | 1.571 |
| Biaya program promo (one-time) | Rp 492.850.000 |
| Revenue terselamatkan (per bulan, recurring) | Rp 157.100.000 |
| Revenue terselamatkan (6 bulan horizon) | Rp 942.600.000 |
| **Net Impact (6 bulan)** | **+Rp 449.750.000** |
| **ROI (6 bulan)** | **+91.3%** |
| **Payback Period** | **~3.1 bulan** |

### Interpretasi
Biaya promo (Rp 492.85 juta, one-time) tertutup oleh revenue yang
diselamatkan dalam **~3.1 bulan**. Setelah payback, program ini
menghasilkan **net positive Rp 157.1 juta per bulan secara recurring**
selama pelanggan yang diretensi tetap aktif. Ini adalah **business case
yang layak dijalankan** dengan asumsi konservatif — jika efektivitas
retensi aktual lebih tinggi dari 40% (misal melalui personalisasi
berbasis SHAP per pelanggan), payback period akan lebih singkat.

> ⚠️ **Disclaimer**: Parameter ARPU, biaya promo, dan efektivitas retensi
> di atas adalah **asumsi ilustratif** untuk demonstrasi metodologi.
> Sebelum keputusan investasi riil, validasi dengan data historis aktual
> tim Finance & Marketing perusahaan.

---

## 7. Rekomendasi Strategis Tambahan

1. **Program "First 6 Months"** — Pelanggan baru (tenure < 6 bulan) adalah
   *window kritis* churn (kontribusi feature importance #6). Bangun
   *onboarding journey* khusus dengan check-in CSAT lebih sering (misal
   minggu ke-2, ke-4, ke-8).

2. **Early Warning System berbasis CSAT & Komplain** — Karena kedua fitur
   ini adalah top predictor (47% importance gabungan), integrasikan model
   ke CRM agar tim CS menerima notifikasi **real-time** saat:
   - Skor CSAT pelanggan turun ke ≤2, ATAU
   - Komplain ke-2 dalam 6 bulan tercatat.

   Intervensi pada titik ini terjadi **sebelum** churn, bukan reaktif
   setelahnya.

3. **Bundle & E-Wallet Adoption Campaign** — Kedua fitur ini (`punya_streaming_bundle`,
   `punya_ewallet_linked`) masuk top-5 importance dan keduanya **menurunkan**
   churn. Targetkan pelanggan Prabayar tanpa bundle untuk *free trial*
   1 bulan streaming bundle, dan kampanye e-wallet linking dengan insentif
   cashback pertama.

4. **Migrasi Prabayar → Pascabayar/Hybrid** — `paket` adalah fitur
   importance #3. Tawarkan benefit migrasi (misal kuota tambahan 3 bulan
   pertama) untuk pelanggan Prabayar high-value yang stabil.

5. **Model Monitoring & Retraining** — Jadwalkan retraining triwulanan.
   Lanskap kompetitif 2026 (masuknya provider satelit seperti Starlink)
   dapat mengubah pola churn dari waktu ke waktu — *model drift* harus
   dipantau via metrik ROC-AUC pada data terbaru.

---

## 8. Limitasi & Next Steps

| Limitasi | Mitigasi / Next Step |
|---|---|
| Dataset sintetis, bukan data CRM riil | Retrain dengan data CRM aktual (anonymized) saat tersedia |
| Recall 0.53 pada threshold 0.5 | Operasional gunakan segmentasi 3-tier (lihat Bagian 4-5), bukan threshold tunggal |
| Model klasifikasi biner (churn/tidak) | Eksplorasi **survival analysis** (waktu-hingga-churn) untuk prioritisasi lebih granular |
| Parameter business impact ilustratif | Validasi ARPU, cost promo, efektivitas dengan data Finance aktual |
| Belum ada A/B testing efektivitas retensi | Jalankan A/B test pada subset High Risk untuk mengukur efektivitas riil (vs asumsi 40%) |

---

## 9. Lampiran: Cara Reproduksi Analisis

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Generate data & jalankan pipeline
python -m src.data_processing        # data/raw, data/processed/telco_clean.csv
python -m src.feature_engineering    # data/processed/telco_features.csv, models/encoder.pkl, scaler.pkl
python -m src.model                  # models/churn_model_xgb.pkl, metrics.pkl

# 3. (Opsional) Eksplorasi via Jupyter Notebook
jupyter notebook notebooks/

# 4. Jalankan dashboard interaktif
streamlit run app/streamlit_app.py
```

Seluruh angka pada laporan ini diekstrak langsung dari
`models/metrics.pkl` hasil run pipeline di atas dengan `random_state=42`
— hasil akan identik pada eksekusi ulang.

---

*Report ini dihasilkan sebagai bagian dari project "Telco Customer Churn
Prediction & Retention Dashboard 2026". Untuk pertanyaan teknis, lihat
dokumentasi kode di `src/` atau jalankan notebook di `notebooks/`.*
