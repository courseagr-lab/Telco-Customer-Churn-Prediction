"""
Modul utility umum yang dipakai di seluruh project:
- Konfigurasi path (data, model, reports)
- Setup logging konsisten
- Helper untuk save/load object (pickle) dengan validasi
- Konstanta nama kolom & kategori risiko
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any


# PATH CONFIGURATION

# Semua path didefinisikan relatif terhadap root project, supaya script bisa
# dijalankan dari folder manapun (notebooks/, src/, app/) tanpa error path.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "telco_customer_churn.csv"
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "telco_clean.csv"
FEATURES_DATA_PATH = PROCESSED_DATA_DIR / "telco_features.csv"

MODEL_PATH = MODELS_DIR / "churn_model_xgb.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.pkl"
METRICS_PATH = MODELS_DIR / "metrics.pkl"


# LOGGING SETUP
def get_logger(name: str) -> logging.Logger:
    """
    Mengembalikan logger dengan format konsisten di seluruh project.

    Parameters:
    name : str
        Nama logger, biasanya `__name__` dari module pemanggil.

    Returns:
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # hindari duplikasi handler saat reload
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# PICKLE HELPERS
def save_pickle(obj: Any, path: Path) -> None:
    """Simpan object Python ke file pickle, otomatis membuat parent dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    """Load object Python dari file pickle. Raise FileNotFoundError jika tidak ada."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}. "
            f"Pastikan pipeline training sudah dijalankan (lihat src/model.py)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# KONSTANTA BISNIS
TARGET_COL = "churn"
ID_COL = "customer_id"

# Kolom kategorikal asli (sebelum encoding)
CATEGORICAL_COLS = [
    "gender",
    "kota",
    "paket",
    "jenis_jaringan",
    "metode_bayar",
    "punya_streaming_bundle",
    "punya_ewallet_linked",
    "pernah_upgrade_paket",
    "tenure_group",
]

# Kolom numerik yang akan di-scale
NUMERIC_COLS_TO_SCALE = [
    "usia",
    "tenure_bulan",
    "biaya_bulanan",
    "total_pemakaian_data_gb",
    "frekuensi_login_app",
    "engagement_score",
    "usage_per_cost",
]

# Threshold segmentasi risiko churn (dipakai di app & reporting)
RISK_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.40,
}


def segment_risk(prob: float) -> str:
    """
    Mapping probabilitas churn -> label segmen risiko.

    Parameters:
    prob : float
        Probabilitas churn dari model (0-1).

    Returns:
    str
        'High Risk', 'Medium Risk', atau 'Low Risk'.
    """
    if prob >= RISK_THRESHOLDS["high"]:
        return "High Risk"
    elif prob >= RISK_THRESHOLDS["medium"]:
        return "Medium Risk"
    return "Low Risk"


def segment_color(segment: str) -> str:
    """Mapping label segmen -> kode warna hex (dipakai konsisten di seluruh app)."""
    mapping = {
        "High Risk": "#e74c3c",
        "Medium Risk": "#f39c12",
        "Low Risk": "#2ecc71",
    }
    return mapping.get(segment, "#95a5a6")


RECOMMENDATION_MAP = {
    "High Risk": (
        "Hubungi pelanggan dalam 48 jam melalui CS prioritas. "
        "Tawarkan voucher data 10GB + diskon 20% untuk 1 bulan berikutnya. "
        "Eskalasi ke tim retensi senior jika tenure < 6 bulan."
    ),
    "Medium Risk": (
        "Kirim push notification personalisasi berisi penawaran upgrade "
        "ke paket bundle streaming. Pantau skor CSAT pada interaksi berikutnya."
    ),
    "Low Risk": (
        "Masukkan ke program loyalty points & cross-sell e-wallet linking. "
        "Tidak perlu intervensi mendesak, cukup monitoring rutin."
    ),
}
