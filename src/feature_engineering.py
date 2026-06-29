"""
Modul untuk transformasi fitur dari data clean > data ready formodeling.

Tahapan:
1. Feature creation: binning tenure, rasio usage/cost, flag risiko, engagement score.
2. Encoding kategorikal (LabelEncoder per kolom, disimpan sebagai dict).
3. Scaling fitur numerik (StandardScaler).

Encoder & scaler disimpan ke disk (models/encoder.pkl, models/scaler.pkl) agar
proses transformasi yang sama bisa dipakai ulang saat inference di Streamlit app
(menghindari training-serving skew).

"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    CATEGORICAL_COLS,
    ENCODER_PATH,
    FEATURES_DATA_PATH,
    NUMERIC_COLS_TO_SCALE,
    PROCESSED_DATA_DIR,
    SCALER_PATH,
    CLEAN_DATA_PATH,
    get_logger,
    save_pickle,
    load_pickle,
)

logger = get_logger(__name__)

TENURE_BINS = [0, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "6-12", "12-24", "24-48", "48-72"]


# 1. FEATURE CREATION
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur derivatif dari data yang sudah clean.

    Fitur baru yang dibuat:
    - tenure_group       : binning tenure_bulan jadi kategori (0-6, 6-12, dst).
    - usage_per_cost     : rasio pemakaian data terhadap biaya bulanan (proxy "value perception" pelanggan).
    - high_complain_flag : 1 jika komplain >= 2 kali dalam 6 bulan.
    - low_csat_flag      : 1 jika skor kepuasan <= 2.
    - engagement_score   : skor gabungan dari frekuensi login, pemakaian data, dan CSAT (semakin tinggi = semakin engaged).

    Parameters:
    df : pd.DataFrame
        DataFrame hasil `clean_data`.

    Returns:
    pd.DataFrame
        DataFrame dengan kolom tambahan di atas.
    """
    df = df.copy()

    # tenure_group: binning
    df["tenure_group"] = pd.cut(
        df["tenure_bulan"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    )
    # pd.cut bisa menghasilkan NaN jika ada nilai di luar bins > fallback
    n_na_tenure = df["tenure_group"].isna().sum()
    if n_na_tenure > 0:
        logger.warning(
            "%d baris tenure_group NaN setelah binning, fallback ke '0-6'",
            n_na_tenure,
        )
        df["tenure_group"] = df["tenure_group"].cat.add_categories(["0-6"])
        df["tenure_group"] = df["tenure_group"].fillna("0-6")

    # usage_per_cost: value perception 
    df["usage_per_cost"] = df["total_pemakaian_data_gb"] / (df["biaya_bulanan"] + 1)

    # Flag risiko
    df["high_complain_flag"] = (df["jumlah_komplain_6bln"] >= 2).astype(int)
    df["low_csat_flag"] = (df["skor_kepuasan_csat"] <= 2).astype(int)

    # Engagement score: kombinasi tertimbang
    df["engagement_score"] = (
        df["frekuensi_login_app"] * 0.4
        + df["total_pemakaian_data_gb"] * 0.01
        + df["skor_kepuasan_csat"] * 2
    )

    logger.info(
        "Feature creation selesai. Kolom baru: tenure_group, usage_per_cost, "
        "high_complain_flag, low_csat_flag, engagement_score"
    )
    return df


# 2. ENCODING
def fit_encoders(df: pd.DataFrame, cat_cols=None) -> Dict[str, LabelEncoder]:
    """
    Fit satu LabelEncoder per kolom kategorikal.

    Parameters:
    df : pd.DataFrame
        DataFrame yang sudah memiliki kolom-kolom kategorikal (termasuk
        `tenure_group` hasil `create_features`).
    cat_cols : list[str], optional
        Daftar kolom kategorikal. Default: `CATEGORICAL_COLS` dari utils.

    Returns:
    dict[str, LabelEncoder]
        Mapping nama kolom > LabelEncoder yang sudah di-fit.
    """
    cat_cols = cat_cols or CATEGORICAL_COLS
    encoders: Dict[str, LabelEncoder] = {}

    for col in cat_cols:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
        logger.info(
            "Encoder fit untuk kolom '%s': %d kelas -> %s",
            col,
            len(le.classes_),
            list(le.classes_),
        )

    return encoders


def apply_encoders(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder], cat_cols=None
) -> pd.DataFrame:
    """
    Terapkan encoder yang sudah di-fit ke DataFrame.

    Menangani unseen categories dengan aman: kategori yang tidak dikenal
    di-mapping ke kelas pertama (index 0) dan dicatat sebagai warning,
    sehingga inference tidak crash saat ada nilai baru yang belum pernah
    terlihat saat training.

    Parameters:
    df : pd.DataFrame
        DataFrame yang akan di-encode.
    encoders : dict[str, LabelEncoder]
        Hasil dari `fit_encoders` atau di-load dari `models/encoder.pkl`.
    cat_cols : list[str], optional
        Daftar kolom yang akan di-encode. Default: keys dari `encoders`.

    Returns:
    pd.DataFrame
        DataFrame dengan kolom kategorikal sudah dalam bentuk numerik.
    """
    df = df.copy()
    cat_cols = cat_cols or list(encoders.keys())

    for col in cat_cols:
        le = encoders[col]
        values = df[col].astype(str)

        known_classes = set(le.classes_)
        unseen_mask = ~values.isin(known_classes)
        n_unseen = int(unseen_mask.sum())

        if n_unseen > 0:
            logger.warning(
                "Kolom '%s': %d nilai unseen, di-map ke kelas default '%s'",
                col,
                n_unseen,
                le.classes_[0],
            )
            values = values.where(~unseen_mask, le.classes_[0])

        df[col] = le.transform(values)

    return df


# 3. SCALING
def fit_scaler(df: pd.DataFrame, num_cols=None) -> StandardScaler:
    """Fit StandardScaler pada kolom numerik yang ditentukan."""
    num_cols = num_cols or NUMERIC_COLS_TO_SCALE
    scaler = StandardScaler()
    scaler.fit(df[num_cols])
    logger.info(
        "Scaler fit pada %d kolom numerik: %s", len(num_cols), num_cols
    )
    return scaler


def apply_scaler(
    df: pd.DataFrame, scaler: StandardScaler, num_cols=None
) -> pd.DataFrame:
    """Terapkan StandardScaler yang sudah di-fit ke kolom numerik."""
    df = df.copy()
    num_cols = num_cols or NUMERIC_COLS_TO_SCALE
    df[num_cols] = scaler.transform(df[num_cols])
    return df


# 4. PIPELINE END-TO-END (FIT MODE - untuk training)
def build_features_pipeline(
    df: pd.DataFrame, fit: bool = True
) -> tuple[pd.DataFrame, Dict[str, LabelEncoder], StandardScaler]:
    """
    Pipeline lengkap feature engineering dalam mode FIT (training).

    Parameters:
    df : pd.DataFrame
        DataFrame hasil `clean_data`.
    fit : bool
        Jika True, fit encoder & scaler baru dari data ini (mode training).
        Jika False, gunakan encoder & scaler yang sudah ada di disk (mode
        inference) -- lihat `transform_new_data` untuk kasus ini.

    Returns
    tuple
        (df_transformed, encoders, scaler)
    """
    df_feat = create_features(df)

    if fit:
        encoders = fit_encoders(df_feat)
        scaler = fit_scaler(df_feat)
    else:
        encoders = load_pickle(ENCODER_PATH)
        scaler = load_pickle(SCALER_PATH)

    df_encoded = apply_encoders(df_feat, encoders)
    df_scaled = apply_scaler(df_encoded, scaler)

    return df_scaled, encoders, scaler


def transform_new_data(raw_input: pd.DataFrame) -> pd.DataFrame:
    """
    Transform data BARU (misal dari input form Streamlit) menggunakan
    encoder & scaler yang sudah di-fit saat training (load dari disk).

    Ini adalah fungsi yang dipanggil saat INFERENCE memastikan tidak ada
    training-serving skew karena memakai object yang identik dengan training.

    Parameters:
    raw_input : pd.DataFrame
        DataFrame dengan kolom-kolom RAW (sebelum feature engineering),
        sama seperti format `data/processed/telco_clean.csv` tanpa kolom churn.

    Returns:
    pd.DataFrame
        DataFrame siap untuk `model.predict_proba()`.
    """
    df_feat = create_features(raw_input)

    encoders = load_pickle(ENCODER_PATH)
    scaler = load_pickle(SCALER_PATH)

    df_encoded = apply_encoders(df_feat, encoders)
    df_scaled = apply_scaler(df_encoded, scaler)

    return df_scaled

# CLI ENTRY POINT
def main() -> None:
    """Load data clean > build features > save dataset siap modeling + artifacts."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{CLEAN_DATA_PATH} tidak ditemukan. "
            f"Jalankan `python -m src.data_processing` terlebih dahulu."
        )

    df_clean = pd.read_csv(CLEAN_DATA_PATH)
    logger.info("Loaded clean data: shape=%s", df_clean.shape)

    df_final, encoders, scaler = build_features_pipeline(df_clean, fit=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(FEATURES_DATA_PATH, index=False)
    save_pickle(encoders, ENCODER_PATH)
    save_pickle(scaler, SCALER_PATH)

    print("\nFEATURE ENGINEERING SUMMARY")
    print(f"Output shape : {df_final.shape}")
    print(f"Saved to     : {FEATURES_DATA_PATH}")
    print(f"Encoders     : {ENCODER_PATH} ({len(encoders)} kolom)")
    print(f"Scaler       : {SCALER_PATH}")
    print(f"\nKolom akhir:\n{list(df_final.columns)}")


if __name__ == "__main__":
    main()
