"""
Entry point dashboard Telco Customer Churn Prediction 2026.

Halaman ini berfungsi sebagai:
1. Landing page / welcome screen dengan ringkasan project.
2. KPI cards tingkat tinggi (total pelanggan, churn rate, dsb).
3. Navigasi ke 4 halaman analisis di `app/pages/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# PATH SETUP — agar `from src...` bisa di-import dari folder app/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import FEATURES_DATA_PATH, MODEL_PATH  # noqa: E402

# PAGE CONFIG
st.set_page_config(
    page_title="Telco Churn Dashboard 2026",
    layout="wide",
    initial_sidebar_state="expanded",
)


# DATA LOADING (cached)
@st.cache_data(show_spinner="Memuat data pelanggan...")
def load_data() -> pd.DataFrame:
    """Load dataset hasil feature engineering untuk KPI overview."""
    if not FEATURES_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_DATA_PATH)


# MAIN
def main() -> None:
    st.title("Telco Customer Churn Prediction Dashboard")
    st.markdown(
        "Dashboard analisis & prediksi churn pelanggan telekomunikasi — "
        "**Indonesia 2026**"
    )

    if not MODEL_PATH.exists() or not FEATURES_DATA_PATH.exists():
        st.error(
            "**Artifact model/data belum tersedia.**\n\n"
            "Jalankan pipeline berikut dari root project sebelum menjalankan "
            "dashboard ini:\n"
            "```bash\n"
            "python -m src.data_processing\n"
            "python -m src.feature_engineering\n"
            "python -m src.model\n"
            "```"
        )
        st.stop()

    df = load_data()

    # KPI CARDS
    st.markdown("### Ringkasan Pelanggan")
    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df)
    churn_rate = df["churn"].mean() * 100
    avg_biaya = df["biaya_bulanan"].mean()
    avg_tenure = df["tenure_bulan"].mean()

    col1.metric("Total Pelanggan", f"{total_customers:,}")
    col2.metric("Churn Rate", f"{churn_rate:.1f}%", delta=None)
    col3.metric("Avg Biaya Bulanan", f"Rp {avg_biaya * 1000:,.0f}")
    col4.metric("Avg Tenure", f"{avg_tenure:.1f} bulan")

    st.markdown("---")

    # QUICK CHARTS
    c1, c2 = st.columns(2)

    with c1:
        churn_counts = df["churn"].value_counts().rename(
            index={0: "Tidak Churn", 1: "Churn"}
        )
        fig_pie = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Distribusi Status Churn",
            color=churn_counts.index,
            color_discrete_map={"Tidak Churn": "#2ecc71", "Churn": "#e74c3c"},
            hole=0.4,
        )
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        # tenure_group sudah ter-encode jadi numerik di telco_features.csv,
        # jadi re-bin dari tenure_bulan asli untuk tampilan yang readable.
        df_chart = df.copy()
        df_chart["tenure_group_label"] = pd.cut(
            df_chart["tenure_bulan"],
            bins=[0, 6, 12, 24, 48, 72],
            labels=["0-6", "6-12", "12-24", "24-48", "48-72"],
            include_lowest=True,
        )
        churn_by_tenure = (
            df_chart.groupby("tenure_group_label", observed=True)["churn"]
            .mean()
            .reset_index()
        )
        churn_by_tenure["churn"] = churn_by_tenure["churn"] * 100

        fig_bar = px.bar(
            churn_by_tenure,
            x="tenure_group_label",
            y="churn",
            title="Churn Rate (%) per Kelompok Tenure (bulan)",
            labels={"tenure_group_label": "Tenure (bulan)", "churn": "Churn Rate (%)"},
            color="churn",
            color_continuous_scale="Reds",
        )
        fig_bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_bar, width='stretch')

    st.markdown("---")

    # NAVIGATION GUIDE
    st.markdown("###Navigasi Dashboard")
    st.markdown(
        "Gunakan menu di **sidebar kiri** untuk membuka halaman berikut:"
    )

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        st.info(
            "**Overview**\n\n"
            "Statistik agregat, distribusi fitur, dan tren churn "
            "berdasarkan kota, paket, dan jaringan."
        )
        st.info(
            "**Customer Detail**\n\n"
            "Cari & lihat profil lengkap pelanggan individual, termasuk "
            "skor risiko churn-nya."
        )
    with nav_col2:
        st.info(
            "**Prediksi Churn**\n\n"
            "Form interaktif untuk memprediksi probabilitas churn "
            "pelanggan baru secara real-time."
        )
        st.info(
            "**Rekomendasi**\n\n"
            "Insight model (feature importance), segmentasi risiko, "
            "dan rekomendasi aksi retensi per segmen."
        )

    st.markdown("---")
    st.caption(
        "2026 — Telco Churn Prediction Project | Model: XGBoost Classifier | "
        "Dataset: Sintetis (7,000 pelanggan)"
    )


if __name__ == "__main__":
    main()
