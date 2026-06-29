"""
Modul untuk:
1. Generate dataset sintetis pelanggan telco Indonesia 2026 (realistic synthetic data).
2. Data cleaning: handle missing values, mixed-type columns, outliers, duplikat, dtype.

Dataset sintetis dibuat dengan logika churn yang TIDAK acak murni — probabilitas
churn dipengaruhi oleh kombinasi fitur (komplain, CSAT, tenure, dll) sehingga
pola yang dipelajari model merefleksikan hubungan bisnis yang realistis.

Author : Data Science Team
Project: Telco Churn Prediction 2026
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import (
    CLEAN_DATA_PATH,
    RAW_DATA_PATH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    get_logger,
)

logger = get_logger(__name__)

RANDOM_STATE = 42
N_CUSTOMERS = 7000


# ---------------------------------------------------------------------------
# 1. DATA GENERATION
# ---------------------------------------------------------------------------
def generate_synthetic_data(
    n: int = N_CUSTOMERS, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Generate dataset sintetis pelanggan telco dengan 18 fitur + target `churn`.

    Probabilitas churn dikonstruksi dari kombinasi linear fitur-fitur kunci
    (komplain, kepuasan, tenure, jenis paket, engagement) supaya model ML
    yang dilatih nanti memiliki sinyal yang jelas dan dapat diinterpretasi.

    Parameters
    ----------
    n : int
        Jumlah baris (pelanggan) yang akan dibuat.
    random_state : int
        Seed untuk reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan 18 kolom fitur + 1 kolom target `churn`.
    """
    rng = np.random.default_rng(random_state)
    logger.info("Generating synthetic dataset: n=%d, seed=%d", n, random_state)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "usia": rng.integers(18, 65, n),
            "kota": rng.choice(
                ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang",
                 "Makassar", "Yogyakarta", "Lainnya"],
                n,
                p=[0.25, 0.12, 0.12, 0.08, 0.08, 0.07, 0.08, 0.20],
            ),
            "tenure_bulan": rng.integers(0, 72, n),
            "paket": rng.choice(
                ["Prabayar", "Pascabayar", "Hybrid"], n, p=[0.5, 0.35, 0.15]
            ),
            "jenis_jaringan": rng.choice(
                ["4G", "5G", "Fiber"], n, p=[0.45, 0.35, 0.20]
            ),
            # kuota_gb sengaja dibuat mixed-type (angka + 'Unlimited') agar
            # melatih skill data cleaning untuk kolom tidak konsisten.
            "kuota_gb": rng.choice([5, 10, 25, 50, 100, "Unlimited"], n),
            "biaya_bulanan": np.round(rng.gamma(5, 30, n), 2),
            "punya_streaming_bundle": rng.choice(["Yes", "No"], n, p=[0.4, 0.6]),
            "punya_ewallet_linked": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "metode_bayar": rng.choice(
                ["E-Wallet", "Transfer Bank", "Auto-debit", "Kartu Kredit"], n
            ),
            "jumlah_komplain_6bln": rng.poisson(0.8, n),
            "skor_kepuasan_csat": rng.integers(1, 6, n),
            "pernah_upgrade_paket": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "total_pemakaian_data_gb": np.round(rng.exponential(15, n), 2),
            "frekuensi_login_app": rng.poisson(8, n),
        }
    )

    # -----------------------------------------------------------------
    # Logika churn realistis: kombinasi linear faktor risiko & retensi.
    # Setiap koefisien merepresentasikan "bobot" pengaruh fitur terhadap
    # kemungkinan churn berdasarkan domain knowledge industri telco.
    # -----------------------------------------------------------------
    churn_prob = (
        0.05
        + 0.25 * (df["jumlah_komplain_6bln"] >= 2)
        + 0.20 * (df["skor_kepuasan_csat"] <= 2)
        + 0.15 * (df["tenure_bulan"] < 6)
        + 0.10 * (df["paket"] == "Prabayar")
        - 0.10 * (df["punya_streaming_bundle"] == "Yes")
        - 0.08 * (df["punya_ewallet_linked"] == "Yes")
        - 0.05 * (df["pernah_upgrade_paket"] == "Yes")
        + 0.10 * (df["frekuensi_login_app"] < 2)
    )
    churn_prob = churn_prob.clip(0, 1)

    df["churn"] = rng.binomial(1, churn_prob)

    # Inject sedikit missing values secara sengaja (realistis pada data
    # operasional) supaya pipeline cleaning punya kasus nyata untuk ditangani.
    missing_idx = rng.choice(df.index, size=int(n * 0.02), replace=False)
    df.loc[missing_idx, "skor_kepuasan_csat"] = np.nan

    missing_idx_2 = rng.choice(df.index, size=int(n * 0.015), replace=False)
    df.loc[missing_idx_2, "total_pemakaian_data_gb"] = np.nan

    # Inject sedikit duplikat customer_id untuk melatih dedup logic.
    dup_idx = rng.choice(df.index[:-5], size=10, replace=False)
    dup_rows = df.loc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    logger.info(
        "Dataset generated: shape=%s, churn_rate=%.2f%%, missing_total=%d",
        df.shape,
        df["churn"].mean() * 100,
        df.isna().sum().sum(),
    )
    return df


def save_raw_data(df: pd.DataFrame, path=RAW_DATA_PATH) -> None:
    """Simpan DataFrame mentah ke CSV, otomatis membuat parent directory."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Raw data saved to %s (shape=%s)", path, df.shape)


# 2. DATA CLEANING
def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV. Raise FileNotFoundError dengan instruksi jika belum ada."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data tidak ditemukan di {path}. "
            f"Jalankan `python -m src.data_processing` untuk generate data."
        )
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline data cleaning end-to-end.

    Tahapan:
    1. Drop duplikat berdasarkan customer_id (keep first).
    2. Normalisasi kolom `kuota_gb` (mixed-type > numeric + flag unlimited).
    3. Handle missing values:
       - skor_kepuasan_csat > imputasi median per grup paket.
       - total_pemakaian_data_gb > imputasi median global.
    4. Outlier treatment pada `biaya_bulanan` menggunakan metode IQR (capping).
    5. Cast tipe data kategorikal > category, target > int.

    Parameters:
    df : pd.DataFrame
        Raw DataFrame hasil `generate_synthetic_data` / `load_raw_data`.

    Returns:
    pd.DataFrame
        DataFrame yang sudah dibersihkan, siap untuk EDA & feature engineering.
    """
    df = df.copy()
    n_before = len(df)

    # 1. Drop duplikat
    df = df.drop_duplicates(subset="customer_id", keep="first").reset_index(
        drop=True
    )
    n_after_dedup = len(df)
    logger.info(
        "Duplikat dihapus: %d baris (%d -> %d)",
        n_before - n_after_dedup,
        n_before,
        n_after_dedup,
    )

    # 2. Normalisasi kolom kuota_gb (mixed type)
    df["kuota_gb"] = df["kuota_gb"].astype(str).replace("Unlimited", "999")
    df["kuota_gb"] = pd.to_numeric(df["kuota_gb"], errors="coerce")
    df["is_unlimited"] = (df["kuota_gb"] == 999).astype(int)
    logger.info(
        "Kolom kuota_gb dinormalisasi. is_unlimited rate=%.2f%%",
        df["is_unlimited"].mean() * 100,
    )

    # 3. Handle missing values
    n_missing_csat_before = df["skor_kepuasan_csat"].isna().sum()
    df["skor_kepuasan_csat"] = df.groupby("paket")["skor_kepuasan_csat"].transform(
        lambda s: s.fillna(s.median())
    )
    # fallback jika ada grup yang seluruhnya NaN
    df["skor_kepuasan_csat"] = df["skor_kepuasan_csat"].fillna(
        df["skor_kepuasan_csat"].median()
    )
    logger.info(
        "Missing skor_kepuasan_csat diisi: %d nilai (median per grup paket)",
        n_missing_csat_before,
    )

    n_missing_usage_before = df["total_pemakaian_data_gb"].isna().sum()
    df["total_pemakaian_data_gb"] = df["total_pemakaian_data_gb"].fillna(
        df["total_pemakaian_data_gb"].median()
    )
    logger.info(
        "Missing total_pemakaian_data_gb diisi: %d nilai (median global)",
        n_missing_usage_before,
    )

    remaining_na = df.isna().sum().sum()
    if remaining_na > 0:
        logger.warning("Masih ada %d missing values setelah imputasi!", remaining_na)
    else:
        logger.info("Tidak ada missing values setelah imputasi.")

    # 4. Outlier treatment: biaya_bulanan via IQR capping 
    q1, q3 = df["biaya_bulanan"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = max(0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr

    n_outliers = (
        (df["biaya_bulanan"] < lower_bound) | (df["biaya_bulanan"] > upper_bound)
    ).sum()
    df["biaya_bulanan"] = df["biaya_bulanan"].clip(lower=lower_bound, upper=upper_bound)
    logger.info(
        "Outlier biaya_bulanan di-cap: %d baris (bounds=[%.2f, %.2f])",
        n_outliers,
        lower_bound,
        upper_bound,
    )

    # 5. Cast tipe data
    df["churn"] = df["churn"].astype(int)

    base_cat_cols = [
        "gender",
        "kota",
        "paket",
        "jenis_jaringan",
        "metode_bayar",
        "punya_streaming_bundle",
        "punya_ewallet_linked",
        "pernah_upgrade_paket",
    ]
    for col in base_cat_cols:
        df[col] = df[col].astype("category")

    df["usia"] = df["usia"].astype(int)
    df["tenure_bulan"] = df["tenure_bulan"].astype(int)
    df["jumlah_komplain_6bln"] = df["jumlah_komplain_6bln"].astype(int)
    df["skor_kepuasan_csat"] = df["skor_kepuasan_csat"].astype(int)
    df["frekuensi_login_app"] = df["frekuensi_login_app"].astype(int)
    df["kuota_gb"] = df["kuota_gb"].astype(int)

    logger.info("Cleaning selesai. Final shape=%s", df.shape)
    return df


def save_clean_data(df: pd.DataFrame, path=CLEAN_DATA_PATH) -> None:
    """Simpan DataFrame hasil cleaning ke CSV."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Clean data saved to %s (shape=%s)", path, df.shape)


# CLI ENTRY POINT
def main() -> None:
    """Jalankan end-to-end: generate raw data > clean > save."""
    raw_df = generate_synthetic_data()
    save_raw_data(raw_df)

    clean_df = clean_data(raw_df)
    save_clean_data(clean_df)

    print("\nSUMMARY")
    print(f"Raw shape    : {raw_df.shape}")
    print(f"Clean shape  : {clean_df.shape}")
    print(f"Churn rate   : {clean_df['churn'].mean() * 100:.2f}%")
    print(f"Missing total: {clean_df.isna().sum().sum()}")


if __name__ == "__main__":
    main()
