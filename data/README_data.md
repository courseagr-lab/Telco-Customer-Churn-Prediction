# Dokumentasi Dataset — Telco Customer Churn 2026

## 1. Ringkasan

| | |
|---|---|
| **Sumber** | Sintetis (synthetic), dibuat oleh `src/data_processing.py` |
| **Jumlah baris (raw)** | 7.010 (termasuk 10 duplikat sengaja) |
| **Jumlah baris (clean)** | 7.000 |
| **Jumlah kolom (raw)** | 18 |
| **Jumlah kolom (clean)** | 19 (+`is_unlimited`) |
| **Jumlah kolom (features)** | 24 (+5 fitur derivatif) |
| **Target** | `churn` (0 = aktif, 1 = churn) |
| **Churn rate** | ~15.3% (imbalanced) |
| **Random seed** | 42 (reproducible) |

## 2. Mengapa Sintetis?

Dataset publik Telco Churn klasik (IBM/Kaggle, ~2018) sudah tidak
merefleksikan lanskap industri telco Indonesia 2026 (5G, e-wallet,
streaming bundle, kompetisi Starlink). Dataset sintetis ini dirancang agar:

1. **Reproducible** — seed tetap (42), siapa pun bisa regenerate data
   yang identik.
2. **Relevan 2026** — mencakup fitur 5G, streaming bundle, e-wallet linking.
3. **Realistic churn logic** — probabilitas churn dikonstruksi dari
   kombinasi fitur (bukan acak murni), sehingga model yang dilatih
   memiliki sinyal yang bermakna dan dapat diinterpretasi.
4. **Mengandung data quality issues** secara sengaja (missing values,
   mixed-type column, outlier, duplikat) untuk melatih pipeline cleaning
   yang representatif terhadap data operasional nyata.

## 3. Skema Kolom — `data/raw/telco_customer_churn.csv`

| Kolom | Tipe | Deskripsi | Catatan |
|---|---|---|---|
| `customer_id` | string | ID unik pelanggan (format `CUST-XXXXX`) | 10 duplikat sengaja |
| `gender` | category | `Male` / `Female` | |
| `usia` | int | Usia pelanggan (18-64) | |
| `kota` | category | Kota domisili (8 kategori) | |
| `tenure_bulan` | int | Lama berlangganan (0-71 bulan) | |
| `paket` | category | `Prabayar` / `Pascabayar` / `Hybrid` | |
| `jenis_jaringan` | category | `4G` / `5G` / `Fiber` | |
| `kuota_gb` | mixed | Kuota data: 5/10/25/50/100/`'Unlimited'` | **Mixed-type**, perlu cleaning |
| `biaya_bulanan` | float | Biaya bulanan (ribuan Rupiah) | Mengandung outlier |
| `punya_streaming_bundle` | category | `Yes` / `No` | |
| `punya_ewallet_linked` | category | `Yes` / `No` | |
| `metode_bayar` | category | `E-Wallet` / `Transfer Bank` / `Auto-debit` / `Kartu Kredit` | |
| `jumlah_komplain_6bln` | int | Jumlah komplain 6 bulan terakhir | |
| `skor_kepuasan_csat` | float | Skor kepuasan 1-5 | **~2% missing** |
| `pernah_upgrade_paket` | category | `Yes` / `No` | |
| `total_pemakaian_data_gb` | float | Total pemakaian data (GB) | **~1.5% missing** |
| `frekuensi_login_app` | int | Frekuensi login app per minggu | |
| `churn` | int | **TARGET**: 0 = aktif, 1 = churn | |

## 4. Kolom Tambahan — `data/processed/telco_clean.csv`

| Kolom | Deskripsi |
|---|---|
| `is_unlimited` | 1 jika `kuota_gb` aslinya `'Unlimited'`, 0 jika numerik |

`kuota_gb` di-cast ke `999` untuk merepresentasikan unlimited secara numerik.

## 5. Kolom Tambahan — `data/processed/telco_features.csv`

Hasil dari `src/feature_engineering.py`:

| Kolom | Deskripsi | Formula/Logika |
|---|---|---|
| `tenure_group` | Binning tenure ke 5 kategori | `0-6`, `6-12`, `12-24`, `24-48`, `48-72` |
| `usage_per_cost` | Proxy value perception | `total_pemakaian_data_gb / (biaya_bulanan + 1)` |
| `high_complain_flag` | Flag komplain tinggi | `1` jika `jumlah_komplain_6bln >= 2` |
| `low_csat_flag` | Flag kepuasan rendah | `1` jika `skor_kepuasan_csat <= 2` |
| `engagement_score` | Skor engagement gabungan | `frekuensi_login*0.4 + pemakaian*0.01 + csat*2` |

**Semua kolom kategorikal** (termasuk `tenure_group`) di-encode dengan
`LabelEncoder` (lihat `models/encoder.pkl`).

**Kolom numerik** di-scale dengan `StandardScaler` (lihat `models/scaler.pkl`):
`usia`, `tenure_bulan`, `biaya_bulanan`, `total_pemakaian_data_gb`,
`frekuensi_login_app`, `engagement_score`, `usage_per_cost`.

## 6. Data Quality Issues yang Disengaja (untuk latihan cleaning)

| Issue | Lokasi | Jumlah | Penanganan |
|---|---|---|---|
| Duplikat `customer_id` | Raw | 10 baris | `drop_duplicates(keep='first')` |
| Mixed-type `kuota_gb` | Raw | 1,105 baris (`'Unlimited'`) | Convert ke 999 + flag `is_unlimited` |
| Missing `skor_kepuasan_csat` | Raw | 140 baris (~2%) | Imputasi median per grup `paket` |
| Missing `total_pemakaian_data_gb` | Raw | 105 baris (~1.5%) | Imputasi median global |
| Outlier `biaya_bulanan` | Raw | 132 baris | IQR capping |

## 7. Cara Regenerasi Dataset

```bash
python -m src.data_processing       # generate raw + clean
python -m src.feature_engineering   # generate features + encoder/scaler
python -m src.model                 # train model + save artifacts
```

Karena `random_state=42` digunakan secara konsisten, hasil akan identik
setiap kali dijalankan ulang.

## 8. Disclaimer

Dataset ini **SINTETIS** dan dibuat untuk tujuan demonstrasi metodologi
data science end-to-end. Pola statistik (churn rate, korelasi fitur)
dirancang realistis berdasarkan domain knowledge industri telco, namun
**TIDAK merepresentasikan data pelanggan riil** perusahaan manapun. Untuk
penggunaan produksi, ganti dengan data CRM aktual (anonymized) dan ulangi
seluruh pipeline (cleaning → feature engineering → modeling) dengan
validasi statistik terhadap data baru.
