"""
Halaman Customer Detail: cari pelanggan berdasarkan ID, lihat profil lengkap,
dan skor risiko churn-nya (dihitung on-the-fly menggunakan model terlatih).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_feature_list, load_model, predict_proba_df  # noqa: E402
from src.utils import (  # noqa: E402
    CLEAN_DATA_PATH,
    FEATURES_DATA_PATH,
    ID_COL,
    RECOMMENDATION_MAP,
    TARGET_COL,
    segment_color,
    segment_risk,
)

st.set_page_config(page_title="Customer Detail — Telco Churn", page_icon="🔍", layout="wide")


@st.cache_data(show_spinner="Memuat data...")
def load_clean_data() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA_PATH)


@st.cache_data(show_spinner="Memuat data...")
def load_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH)


@st.cache_resource(show_spinner="Memuat model...")
def get_model_artifacts():
    model = load_model()
    feature_names = load_feature_list()
    return model, feature_names


@st.cache_data(show_spinner="Menghitung skor risiko untuk semua pelanggan...")
def compute_all_risk_scores(df_features: pd.DataFrame, _model, feature_names) -> pd.DataFrame:
    """
    Hitung probabilitas churn untuk SEMUA pelanggan sekali saja (cached),
    supaya halaman ini responsif saat user mencari pelanggan berbeda-beda.

    Parameter `_model` diberi underscore prefix agar Streamlit tidak mencoba
    hash object model (yang bisa gagal/lambat untuk object XGBoost).
    """
    X = df_features.drop(columns=[ID_COL, TARGET_COL])
    probas = predict_proba_df(_model, X, feature_names)

    result = df_features[[ID_COL, TARGET_COL]].copy()
    result["churn_probability"] = probas
    result["risk_segment"] = result["churn_probability"].apply(segment_risk)
    return result


def render_gauge(probability: float) -> go.Figure:
    """Render gauge chart untuk probabilitas churn."""
    segment = segment_risk(probability)
    color = segment_color(segment)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40}},
            title={"text": f"Probabilitas Churn — {segment}", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#d4f4dd"},
                    {"range": [40, 70], "color": "#fde9c8"},
                    {"range": [70, 100], "color": "#fbd5d0"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def main() -> None:
    st.title("Customer Detail")
    st.markdown(
        "Cari pelanggan berdasarkan **Customer ID** untuk melihat profil "
        "lengkap dan skor risiko churn yang dihitung oleh model."
    )

    required_files = [CLEAN_DATA_PATH, FEATURES_DATA_PATH]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            f"File berikut belum tersedia: {[str(f) for f in missing]}. "
            "Jalankan pipeline `src/` terlebih dahulu."
        )
        st.stop()

    df_clean = load_clean_data()
    df_features = load_features_data()
    model, feature_names = get_model_artifacts()

    risk_scores = compute_all_risk_scores(df_features, model, feature_names)
    df_merged = df_clean.merge(risk_scores, on=[ID_COL, TARGET_COL], how="left")

    # SEARCH
    st.markdown("###Cari Pelanggan")

    col_search, col_random = st.columns([3, 1])

    with col_search:
        customer_ids = sorted(df_merged[ID_COL].unique().tolist())
        selected_id = st.selectbox(
            "Pilih atau cari Customer ID",
            options=customer_ids,
            index=0,
            help="Ketik untuk mencari, contoh: CUST-00001",
        )

    with col_random:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Pelanggan Acak", width='stretch'):
            selected_id = df_merged[ID_COL].sample(1).values[0]
            st.session_state["random_id"] = selected_id

    if "random_id" in st.session_state:
        selected_id = st.session_state["random_id"]

    customer = df_merged[df_merged[ID_COL] == selected_id]
    if customer.empty:
        st.warning(f"Customer ID `{selected_id}` tidak ditemukan.")
        st.stop()

    customer = customer.iloc[0]

    st.markdown("---")

    # CUSTOMER PROFILE
    col_profile, col_gauge = st.columns([2, 1])

    with col_profile:
        st.markdown(f"### Profil Pelanggan: `{customer[ID_COL]}`")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("**Demografi**")
            st.write(f"Gender: **{customer['gender']}**")
            st.write(f"Usia: **{customer['usia']} tahun**")
            st.write(f"Kota: **{customer['kota']}**")

        with p2:
            st.markdown("**Layanan**")
            st.write(f"Paket: **{customer['paket']}**")
            st.write(f"Jaringan: **{customer['jenis_jaringan']}**")
            st.write(f"Kuota: **{customer['kuota_gb']} GB**" + (" (Unlimited)" if customer['is_unlimited'] == 1 else ""))
            st.write(f"Tenure: **{customer['tenure_bulan']} bulan**")

        with p3:
            st.markdown("**Pembayaran & Engagement**")
            st.write(f"Biaya bulanan: **Rp {customer['biaya_bulanan']*1000:,.0f}**")
            st.write(f"Metode bayar: **{customer['metode_bayar']}**")
            st.write(f"Frekuensi login app: **{customer['frekuensi_login_app']}x/minggu**")
            st.write(f"Pemakaian data: **{customer['total_pemakaian_data_gb']:.1f} GB**")

        st.markdown("**Indikator Kepuasan & Risiko**")
        i1, i2, i3, i4 = st.columns(4)
        i1.metric("Skor CSAT", f"{customer['skor_kepuasan_csat']} / 5")
        i2.metric("Komplain (6 bln)", f"{int(customer['jumlah_komplain_6bln'])}")
        i3.metric("Streaming Bundle", customer["punya_streaming_bundle"])
        i4.metric("E-Wallet Linked", customer["punya_ewallet_linked"])

        actual_status = "CHURN" if customer["churn"] == 1 else "AKTIF"
        st.markdown(f"**Status Aktual (historis)**: {actual_status}")

    with col_gauge:
        st.plotly_chart(render_gauge(customer["churn_probability"]), width='stretch')

        segment = customer["risk_segment"]
        color = segment_color(segment)
        st.markdown(
            f"""
            <div style="background-color:{color}22; border-left: 5px solid {color};
                        padding: 12px; border-radius: 6px;">
                <b>Segmen Risiko: {segment}</b><br>
                Probabilitas churn: <b>{customer['churn_probability']*100:.1f}%</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # RECOMMENDATION
    st.markdown("### Rekomendasi Aksi untuk Pelanggan Ini")
    segment = customer["risk_segment"]
    color = segment_color(segment)

    st.markdown(
        f"""
        <div style="background-color:{color}15; border: 1px solid {color};
                    padding: 16px; border-radius: 8px;">
            <h4 style="margin-top:0; color:{color};">{segment}</h4>
            <p>{RECOMMENDATION_MAP[segment]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Highlight faktor risiko spesifik pelanggan ini
    risk_factors = []
    if customer["jumlah_komplain_6bln"] >= 2:
        risk_factors.append(
            f"Komplain tinggi: **{int(customer['jumlah_komplain_6bln'])}x** dalam 6 bulan terakhir."
        )
    if customer["skor_kepuasan_csat"] <= 2:
        risk_factors.append(
            f"Skor kepuasan rendah: **{customer['skor_kepuasan_csat']}/5**."
        )
    if customer["tenure_bulan"] < 6:
        risk_factors.append(
            f"Pelanggan baru: tenure hanya **{customer['tenure_bulan']} bulan** "
            "(masuk window kritis churn)."
        )
    if customer["paket"] == "Prabayar" and customer["punya_streaming_bundle"] == "No":
        risk_factors.append(
            "Paket Prabayar tanpa streaming bundle — kurang 'stickiness'."
        )
    if customer["frekuensi_login_app"] < 2:
        risk_factors.append(
            f"Engagement rendah: hanya **{int(customer['frekuensi_login_app'])}x/minggu** login app."
        )

    if risk_factors:
        st.markdown("**Faktor risiko spesifik pelanggan ini:**")
        for f in risk_factors:
            st.markdown(f"- {f}")
    else:
        st.markdown(
            "Tidak ada faktor risiko signifikan terdeteksi pada pelanggan ini."
        )

    st.markdown("---")

    # SIMILAR CUSTOMERS
    with st.expander("Lihat Pelanggan Lain dengan Segmen Risiko Sama"):
        similar = df_merged[
            (df_merged["risk_segment"] == segment) & (df_merged[ID_COL] != customer[ID_COL])
        ].sample(min(10, len(df_merged) - 1), random_state=42)

        display_cols = [
            ID_COL, "kota", "paket", "tenure_bulan", "jumlah_komplain_6bln",
            "skor_kepuasan_csat", "churn_probability", "risk_segment",
        ]
        display_df = similar[display_cols].copy()
        display_df["churn_probability"] = (display_df["churn_probability"] * 100).round(1)
        display_df = display_df.rename(
            columns={"churn_probability": "Churn Prob (%)", "risk_segment": "Segmen"}
        )
        st.dataframe(display_df, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
