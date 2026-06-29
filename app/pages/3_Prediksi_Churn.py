"""
Halaman Prediksi Churn: form interaktif untuk memprediksi probabilitas churn
pelanggan BARU secara real-time menggunakan model XGBoost terlatih.

Menggunakan `transform_new_data()` dari `src.feature_engineering` agar
preprocessing IDENTIK dengan yang dipakai saat training (menghindari
training-serving skew).
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

from src.feature_engineering import transform_new_data  # noqa: E402
from src.model import load_feature_list, load_model, predict_single  # noqa: E402
from src.utils import (  # noqa: E402
    ENCODER_PATH,
    MODEL_PATH,
    SCALER_PATH,
    RECOMMENDATION_MAP,
    segment_color,
)

st.set_page_config(page_title="Prediksi Churn — Telco", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Memuat model...")
def get_model_artifacts():
    model = load_model()
    feature_names = load_feature_list()
    return model, feature_names


def render_gauge(probability: float, segment: str) -> go.Figure:
    color = segment_color(segment)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": f"Probabilitas Churn", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 40], "color": "#d4f4dd"},
                    {"range": [40, 70], "color": "#fde9c8"},
                    {"range": [70, 100], "color": "#fbd5d0"},
                ],
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=70, b=10))
    return fig


def main() -> None:
    st.title("Prediksi Churn Pelanggan Baru")
    st.markdown(
        "Masukkan data pelanggan di bawah ini untuk memprediksi "
        "**probabilitas churn** menggunakan model XGBoost terlatih. "
        "Hasil prediksi disertai segmen risiko dan rekomendasi aksi."
    )

    required_files = [MODEL_PATH, ENCODER_PATH, SCALER_PATH]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            f"Artifact model belum tersedia: {[str(f) for f in missing]}. "
            "Jalankan `python -m src.feature_engineering` dan "
            "`python -m src.model` terlebih dahulu."
        )
        st.stop()

    model, feature_names = get_model_artifacts()

    st.markdown("---")
    st.markdown("### Form Input Data Pelanggan")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demografi**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            usia = st.number_input("Usia", min_value=18, max_value=65, value=30, step=1)
            kota = st.selectbox(
                "Kota",
                ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang",
                 "Makassar", "Yogyakarta", "Lainnya"],
            )

        with col2:
            st.markdown("**Layanan & Paket**")
            tenure_bulan = st.number_input(
                "Tenure (bulan)", min_value=0, max_value=72, value=12, step=1,
                help="Lama berlangganan dalam bulan",
            )
            paket = st.selectbox("Jenis Paket", ["Prabayar", "Pascabayar", "Hybrid"])
            jenis_jaringan = st.selectbox("Jenis Jaringan", ["4G", "5G", "Fiber"])
            kuota_option = st.selectbox(
                "Kuota Data", ["5 GB", "10 GB", "25 GB", "50 GB", "100 GB", "Unlimited"]
            )
            biaya_bulanan = st.number_input(
                "Biaya Bulanan (Rp ribu)", min_value=10, max_value=500, value=100, step=5,
                help="Dalam satuan ribu Rupiah, contoh: 100 = Rp 100.000",
            )

        with col3:
            st.markdown("**Engagement & Kepuasan**")
            jumlah_komplain = st.number_input(
                "Jumlah Komplain (6 bulan)", min_value=0, max_value=10, value=0, step=1
            )
            skor_csat = st.slider(
                "Skor Kepuasan (CSAT)", min_value=1, max_value=5, value=3,
                help="1 = Sangat Tidak Puas, 5 = Sangat Puas",
            )
            total_pemakaian = st.number_input(
                "Total Pemakaian Data (GB)", min_value=0.0, max_value=300.0, value=15.0, step=0.5
            )
            frekuensi_login = st.number_input(
                "Frekuensi Login App (per minggu)", min_value=0, max_value=30, value=8, step=1
            )

        st.markdown("**Fitur Tambahan & Layanan**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            punya_streaming = st.selectbox("Punya Streaming Bundle?", ["Yes", "No"])
        with col5:
            punya_ewallet = st.selectbox("E-Wallet Linked?", ["Yes", "No"])
        with col6:
            pernah_upgrade = st.selectbox("Pernah Upgrade Paket?", ["Yes", "No"])
        with col7:
            metode_bayar = st.selectbox(
                "Metode Pembayaran",
                ["E-Wallet", "Transfer Bank", "Auto-debit", "Kartu Kredit"],
            )

        submitted = st.form_submit_button("Prediksi Churn", type="primary", width='stretch')

    # PREDICTION
    if submitted:
        # Parse kuota
        if kuota_option == "Unlimited":
            kuota_gb = 999
            is_unlimited = 1
        else:
            kuota_gb = int(kuota_option.split()[0])
            is_unlimited = 0

        raw_input = pd.DataFrame(
            [
                {
                    "customer_id": "CUST-PREVIEW",
                    "gender": gender,
                    "usia": usia,
                    "kota": kota,
                    "tenure_bulan": tenure_bulan,
                    "paket": paket,
                    "jenis_jaringan": jenis_jaringan,
                    "kuota_gb": kuota_gb,
                    "biaya_bulanan": float(biaya_bulanan),
                    "punya_streaming_bundle": punya_streaming,
                    "punya_ewallet_linked": punya_ewallet,
                    "metode_bayar": metode_bayar,
                    "jumlah_komplain_6bln": jumlah_komplain,
                    "skor_kepuasan_csat": skor_csat,
                    "pernah_upgrade_paket": pernah_upgrade,
                    "total_pemakaian_data_gb": float(total_pemakaian),
                    "frekuensi_login_app": frekuensi_login,
                    "is_unlimited": is_unlimited,
                }
            ]
        )

        transformed = transform_new_data(raw_input)
        result = predict_single(model, feature_names, transformed)

        proba = result["churn_probability"]
        segment = result["risk_segment"]
        color = segment_color(segment)

        st.markdown("---")
        st.markdown("### Hasil Prediksi")

        col_result, col_gauge = st.columns([1, 1])

        with col_gauge:
            st.plotly_chart(render_gauge(proba, segment), width='stretch')

        with col_result:
            if segment == "High Risk":
                st.error(f" **{segment}** — Probabilitas Churn: **{proba*100:.1f}%**")
            elif segment == "Medium Risk":
                st.warning(f" **{segment}** — Probabilitas Churn: **{proba*100:.1f}%**")
            else:
                st.success(f" **{segment}** — Probabilitas Churn: **{proba*100:.1f}%**")

            st.markdown("**Rekomendasi Aksi:**")
            st.markdown(
                f"""
                <div style="background-color:{color}15; border-left: 4px solid {color};
                            padding: 12px; border-radius: 4px;">
                    {RECOMMENDATION_MAP[segment]}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Faktor risiko spesifik
            st.markdown("**Faktor yang Mempengaruhi:**")
            factors = []
            if jumlah_komplain >= 2:
                factors.append(f" Komplain tinggi ({jumlah_komplain}x dalam 6 bulan)")
            if skor_csat <= 2:
                factors.append(f" Skor kepuasan rendah ({skor_csat}/5)")
            if tenure_bulan < 6:
                factors.append(f" Pelanggan baru (tenure {tenure_bulan} bulan)")
            if paket == "Prabayar":
                factors.append(" Paket Prabayar (churn rate struktural lebih tinggi)")
            if punya_streaming == "No" and punya_ewallet == "No":
                factors.append(" Tidak punya streaming bundle maupun e-wallet linked")
            if frekuensi_login < 2:
                factors.append(f" Engagement rendah ({frekuensi_login}x/minggu login)")

            if factors:
                for f in factors:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- Tidak ada faktor risiko signifikan terdeteksi")

        with st.expander("Lihat Detail Fitur Hasil Transformasi (untuk debugging)"):
            st.dataframe(transformed.drop(columns=["customer_id"]), width='stretch')

    else:
        st.info(
            " Isi form di atas dan klik **'Prediksi Churn'** untuk melihat hasil."
        )


if __name__ == "__main__":
    main()
