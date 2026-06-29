"""
Halaman Overview: statistik agregat, distribusi fitur, dan analisis churn
berdasarkan berbagai dimensi (kota, paket, jaringan, dll).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import CLEAN_DATA_PATH, FEATURES_DATA_PATH  # noqa: E402

st.set_page_config(page_title="Overview — Telco Churn", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Memuat data...")
def load_clean_data() -> pd.DataFrame:
    """Load data clean (kategorikal masih dalam label asli, bukan encoded)."""
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_data(show_spinner="Memuat data...")
def load_features_data() -> pd.DataFrame:
    """Load data hasil feature engineering (untuk fitur derivatif numerik)."""
    return pd.read_csv(FEATURES_DATA_PATH)


def main() -> None:
    st.title("Overview — Analisis Pelanggan & Churn")
    st.markdown(
        "Halaman ini menampilkan statistik agregat dan eksplorasi distribusi "
        "dari **data pelanggan yang sudah dibersihkan** (`telco_clean.csv`)."
    )

    if not CLEAN_DATA_PATH.exists():
        st.error(
            f"File `{CLEAN_DATA_PATH}` tidak ditemukan. Jalankan "
            "`python -m src.data_processing` terlebih dahulu."
        )
        st.stop()

    df = load_clean_data()
    df_feat = load_features_data()

    # FILTER SIDEBAR
    st.sidebar.header("Filter Data")

    kota_options = ["Semua"] + sorted(df["kota"].unique().tolist())
    selected_kota = st.sidebar.selectbox("Kota", kota_options)

    paket_options = ["Semua"] + sorted(df["paket"].unique().tolist())
    selected_paket = st.sidebar.selectbox("Paket", paket_options)

    jaringan_options = ["Semua"] + sorted(df["jenis_jaringan"].unique().tolist())
    selected_jaringan = st.sidebar.selectbox("Jenis Jaringan", jaringan_options)

    df_filtered = df.copy()
    if selected_kota != "Semua":
        df_filtered = df_filtered[df_filtered["kota"] == selected_kota]
    if selected_paket != "Semua":
        df_filtered = df_filtered[df_filtered["paket"] == selected_paket]
    if selected_jaringan != "Semua":
        df_filtered = df_filtered[df_filtered["jenis_jaringan"] == selected_jaringan]

    if len(df_filtered) == 0:
        st.warning("Tidak ada data yang sesuai dengan filter yang dipilih.")
        st.stop()

    st.caption(
        f"Menampilkan **{len(df_filtered):,}** dari **{len(df):,}** pelanggan "
        f"berdasarkan filter aktif."
    )

    # KPI ROW
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Pelanggan", f"{len(df_filtered):,}")
    col2.metric("Churn Rate", f"{df_filtered['churn'].mean()*100:.1f}%")
    col3.metric(
        "Avg Komplain (6 bln)", f"{df_filtered['jumlah_komplain_6bln'].mean():.2f}"
    )
    col4.metric(
        "Avg CSAT Score", f"{df_filtered['skor_kepuasan_csat'].mean():.2f} / 5"
    )

    st.markdown("---")

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Demografi & Geografi", "Paket & Layanan", "Perilaku Pelanggan", "Korelasi"]
    )

    # TAB 1: Demografi & Geografi
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            churn_by_kota = (
                df_filtered.groupby("kota")["churn"].mean().sort_values(ascending=False) * 100
            ).reset_index()
            fig = px.bar(
                churn_by_kota,
                x="kota",
                y="churn",
                title="Churn Rate (%) per Kota",
                color="churn",
                color_continuous_scale="Oranges",
                labels={"churn": "Churn Rate (%)", "kota": "Kota"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        with c2:
            gender_churn = (
                df_filtered.groupby(["gender", "churn"]).size().reset_index(name="count")
            )
            gender_churn["churn_label"] = gender_churn["churn"].map(
                {0: "Tidak Churn", 1: "Churn"}
            )
            fig = px.bar(
                gender_churn,
                x="gender",
                y="count",
                color="churn_label",
                title="Distribusi Churn per Gender",
                barmode="group",
                color_discrete_map={"Tidak Churn": "#2ecc71", "Churn": "#e74c3c"},
                labels={"count": "Jumlah Pelanggan", "gender": "Gender", "churn_label": "Status"},
            )
            st.plotly_chart(fig, width='stretch')

        # Distribusi usia
        fig = px.histogram(
            df_filtered,
            x="usia",
            color="churn",
            nbins=20,
            barmode="overlay",
            title="Distribusi Usia Pelanggan berdasarkan Status Churn",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"usia": "Usia", "churn": "Churn"},
            opacity=0.6,
        )
        fig.for_each_trace(
            lambda t: t.update(name="Churn" if t.name == "1" else "Tidak Churn")
        )
        st.plotly_chart(fig, width='stretch')

    # TAB 2: Paket & Layanan
    with tab2:
        c1, c2 = st.columns(2)

        with c1:
            churn_by_paket = (
                df_filtered.groupby("paket")["churn"].mean() * 100
            ).reset_index().sort_values("churn", ascending=False)
            fig = px.bar(
                churn_by_paket,
                x="paket",
                y="churn",
                title="Churn Rate (%) per Jenis Paket",
                color="churn",
                color_continuous_scale="Reds",
                labels={"churn": "Churn Rate (%)", "paket": "Paket"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        with c2:
            churn_by_jaringan = (
                df_filtered.groupby("jenis_jaringan")["churn"].mean() * 100
            ).reset_index().sort_values("churn", ascending=False)
            fig = px.bar(
                churn_by_jaringan,
                x="jenis_jaringan",
                y="churn",
                title="Churn Rate (%) per Jenis Jaringan",
                color="churn",
                color_continuous_scale="Blues",
                labels={"churn": "Churn Rate (%)", "jenis_jaringan": "Jenis Jaringan"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

        c3, c4 = st.columns(2)
        with c3:
            bundle_data = []
            for col, label in [
                ("punya_streaming_bundle", "Streaming Bundle"),
                ("punya_ewallet_linked", "E-Wallet Linked"),
                ("pernah_upgrade_paket", "Pernah Upgrade"),
            ]:
                rate_yes = df_filtered[df_filtered[col] == "Yes"]["churn"].mean() * 100
                rate_no = df_filtered[df_filtered[col] == "No"]["churn"].mean() * 100
                bundle_data.append({"Fitur": label, "Status": "Punya/Ya", "Churn Rate": rate_yes})
                bundle_data.append({"Fitur": label, "Status": "Tidak", "Churn Rate": rate_no})

            bundle_df = pd.DataFrame(bundle_data)
            fig = px.bar(
                bundle_df,
                x="Fitur",
                y="Churn Rate",
                color="Status",
                barmode="group",
                title="Churn Rate (%): Pengaruh Bundle & Upgrade",
                color_discrete_map={"Punya/Ya": "#2ecc71", "Tidak": "#e74c3c"},
                labels={"Churn Rate": "Churn Rate (%)"},
            )
            st.plotly_chart(fig, width='stretch')

        with c4:
            churn_by_payment = (
                df_filtered.groupby("metode_bayar")["churn"].mean() * 100
            ).reset_index().sort_values("churn", ascending=False)
            fig = px.bar(
                churn_by_payment,
                x="metode_bayar",
                y="churn",
                title="Churn Rate (%) per Metode Pembayaran",
                color="churn",
                color_continuous_scale="Purples",
                labels={"churn": "Churn Rate (%)", "metode_bayar": "Metode Bayar"},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

    # TAB 3: Perilaku Pelanggan
    with tab3:
        c1, c2 = st.columns(2)

        with c1:
            complain_churn = (
                df_filtered.groupby("jumlah_komplain_6bln")["churn"].mean() * 100
            ).reset_index()
            fig = px.line(
                complain_churn,
                x="jumlah_komplain_6bln",
                y="churn",
                title="Churn Rate (%) vs Jumlah Komplain (6 bulan)",
                markers=True,
                labels={"jumlah_komplain_6bln": "Jumlah Komplain", "churn": "Churn Rate (%)"},
            )
            fig.update_traces(line_color="#e74c3c")
            st.plotly_chart(fig, width='stretch')

        with c2:
            csat_churn = (
                df_filtered.groupby("skor_kepuasan_csat")["churn"].mean() * 100
            ).reset_index()
            fig = px.line(
                csat_churn,
                x="skor_kepuasan_csat",
                y="churn",
                title="Churn Rate (%) vs Skor Kepuasan (CSAT)",
                markers=True,
                labels={"skor_kepuasan_csat": "Skor CSAT", "churn": "Churn Rate (%)"},
            )
            fig.update_traces(line_color="#3498db")
            st.plotly_chart(fig, width='stretch')

        # Heatmap interaksi komplain x csat
        df_heat = df_filtered.copy()
        df_heat["komplain_cat"] = pd.cut(
            df_heat["jumlah_komplain_6bln"],
            bins=[-1, 0, 1, 2, 100],
            labels=["0", "1", "2", "3+"],
        )
        pivot = (
            df_heat.pivot_table(
                values="churn", index="komplain_cat", columns="skor_kepuasan_csat", aggfunc="mean"
            )
            * 100
        )
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            color_continuous_scale="YlOrRd",
            title="Churn Rate (%): Jumlah Komplain × Skor CSAT",
            labels={"x": "Skor CSAT", "y": "Jumlah Komplain", "color": "Churn Rate (%)"},
            aspect="auto",
        )
        st.plotly_chart(fig, width='stretch')

        st.info(
            "💡 **Insight**: Kombinasi komplain tinggi (≥2) DAN CSAT rendah (≤2) "
            "menghasilkan churn rate tertinggi — segmen ini adalah prioritas "
            "utama tim retensi."
        )

    # TAB 4: Korelasi
    with tab4:
        st.markdown(
            "Heatmap korelasi Pearson antar variabel numerik "
            "(menggunakan dataset hasil feature engineering)."
        )

        numeric_cols = [
            "usia", "tenure_bulan", "biaya_bulanan", "jumlah_komplain_6bln",
            "skor_kepuasan_csat", "total_pemakaian_data_gb", "frekuensi_login_app",
            "engagement_score", "usage_per_cost", "churn",
        ]
        # Filter df_feat sesuai customer_id yang lolos filter
        df_feat_filtered = df_feat[df_feat["customer_id"].isin(df_filtered["customer_id"])]

        corr = df_feat_filtered[numeric_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Heatmap Korelasi Antar Variabel Numerik",
            aspect="auto",
        )
        st.plotly_chart(fig, width='stretch')

        churn_corr = corr["churn"].drop("churn").sort_values(key=abs, ascending=False)
        st.markdown("**Korelasi setiap variabel dengan `churn` (diurutkan):**")
        st.dataframe(
            churn_corr.reset_index().rename(
                columns={"index": "Variabel", "churn": "Korelasi dengan Churn"}
            ),
            width='stretch',
            hide_index=True,
        )

    st.markdown("---")
    with st.expander("Lihat Data Mentah (Sample)"):
        st.dataframe(df_filtered.head(100), width='stretch')


if __name__ == "__main__":
    main()
