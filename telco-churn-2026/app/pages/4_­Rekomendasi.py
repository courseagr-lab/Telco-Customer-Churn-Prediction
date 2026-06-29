"""
Halaman Rekomendasi: insight model (feature importance), performa model,
segmentasi risiko populasi, dan rekomendasi aksi retensi per segmen
beserta estimasi business impact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import load_feature_list, load_metrics, load_model, predict_proba_df  # noqa: E402
from src.utils import (  # noqa: E402
    FEATURES_DATA_PATH,
    ID_COL,
    METRICS_PATH,
    MODEL_PATH,
    RECOMMENDATION_MAP,
    RISK_THRESHOLDS,
    TARGET_COL,
    segment_color,
    segment_risk,
)

st.set_page_config(page_title="Rekomendasi — Telco Churn", page_icon="💡", layout="wide")


@st.cache_data(show_spinner="Memuat data...")
def load_features_data() -> pd.DataFrame:
    return pd.read_csv(FEATURES_DATA_PATH)


@st.cache_resource(show_spinner="Memuat model & metrics...")
def get_model_artifacts():
    model = load_model()
    feature_names = load_feature_list()
    metrics = load_metrics()
    return model, feature_names, metrics


@st.cache_data(show_spinner="Menghitung skor risiko populasi...")
def compute_population_risk(df_features: pd.DataFrame, _model, feature_names) -> pd.DataFrame:
    X = df_features.drop(columns=[ID_COL, TARGET_COL])
    probas = predict_proba_df(_model, X, feature_names)

    result = df_features[[ID_COL, TARGET_COL]].copy()
    result["churn_probability"] = probas
    result["risk_segment"] = result["churn_probability"].apply(segment_risk)
    return result


def main() -> None:
    st.title("Rekomendasi & Insight Model")
    st.markdown(
        "Halaman ini merangkum **performa model**, **fitur paling berpengaruh**, "
        "**segmentasi risiko populasi**, dan **rekomendasi aksi retensi** "
        "yang dapat ditindaklanjuti tim bisnis."
    )

    required_files = [MODEL_PATH, METRICS_PATH, FEATURES_DATA_PATH]
    missing = [f for f in required_files if not f.exists()]
    if missing:
        st.error(
            f"Artifact belum tersedia: {[str(f) for f in missing]}. "
            "Jalankan `python -m src.model` terlebih dahulu."
        )
        st.stop()

    df_features = load_features_data()
    model, feature_names, metrics = get_model_artifacts()
    population_risk = compute_population_risk(df_features, model, feature_names)

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Performa Model", "Feature Importance", "Segmentasi Risiko", "Business Impact"]
    )

    # TAB 1: Performa Model
    with tab1:
        st.markdown("### Ringkasan Performa Model")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ROC-AUC (Test)", f"{metrics['roc_auc']:.3f}")
        col2.metric(
            "ROC-AUC (5-Fold CV)",
            f"{metrics['cv_roc_auc_mean']:.3f}",
            delta=f"±{metrics['cv_roc_auc_std']:.3f}",
            delta_color="off",
        )
        report = metrics["classification_report"]
        col3.metric("Recall (Churn)", f"{report['1']['recall']:.3f}")
        col4.metric("Precision (Churn)", f"{report['1']['precision']:.3f}")

        st.markdown("---")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Confusion Matrix (threshold = 0.5)")
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Pred: Tidak Churn", "Pred: Churn"],
                    y=["Actual: Tidak Churn", "Actual: Churn"],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    colorscale="Blues",
                    showscale=False,
                )
            )
            fig.update_layout(height=350, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width='stretch')

            tn, fp, fn, tp = cm.ravel()
            st.caption(
                f"**TP={tp}** (benar deteksi churn) | **FN={fn}** (gagal deteksi — "
                f"prioritas turunkan!) | **FP={fp}** (salah alarm) | **TN={tn}** (benar non-churn)"
            )

        with c2:
            st.markdown("#### ROC Curve")
            roc = metrics["roc_curve"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=roc["fpr"], y=roc["tpr"], mode="lines",
                    name=f"XGBoost (AUC={metrics['roc_auc']:.3f})",
                    line=dict(color="#3498db", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(color="gray", dash="dash"),
                )
            )
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=350,
                legend=dict(x=0.4, y=0.1),
            )
            st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("#### Classification Report Lengkap")
        report_df = pd.DataFrame(report).T.round(3)
        report_df.index = report_df.index.map(
            lambda x: {"0": "Tidak Churn (0)", "1": "Churn (1)"}.get(x, x)
        )
        st.dataframe(report_df, width='stretch')

        st.markdown("#### Cross-Validation Scores (5-Fold)")
        cv_df = pd.DataFrame(
            {"Fold": [f"Fold {i+1}" for i in range(len(metrics["cv_scores"]))],
             "ROC-AUC": metrics["cv_scores"]}
        )
        fig = px.bar(
            cv_df, x="Fold", y="ROC-AUC", title="ROC-AUC per Fold",
            color="ROC-AUC", color_continuous_scale="Blues", range_y=[0.5, 1.0],
        )
        fig.add_hline(
            y=metrics["cv_roc_auc_mean"], line_dash="dash", line_color="red",
            annotation_text=f"Mean = {metrics['cv_roc_auc_mean']:.3f}",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

    # TAB 2: Feature Importance
    with tab2:
        st.markdown("### Fitur Paling Berpengaruh terhadap Prediksi Churn")
        st.markdown(
            "Diurutkan berdasarkan *importance score* dari model XGBoost "
            "(berapa sering & seberapa signifikan fitur ini dipakai untuk split)."
        )

        feat_imp_df = pd.DataFrame(metrics["feature_importance"])

        fig = px.bar(
            feat_imp_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top Feature Importance (XGBoost)",
            color="importance",
            color_continuous_scale="Viridis",
            labels={"importance": "Importance Score", "feature": "Fitur"},
        )
        fig.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch')

        st.markdown("---")
        st.markdown("### 🔑 Interpretasi 3 Fitur Teratas")

        top3 = feat_imp_df.head(3)

        interpretations = {
            "skor_kepuasan_csat": (
                "**Skor Kepuasan (CSAT)** adalah predictor TERKUAT churn. "
                "Pelanggan dengan CSAT ≤2 memiliki churn rate jauh lebih tinggi "
                "dibanding CSAT ≥4. **Implikasi**: bangun *early warning system* "
                "yang memicu intervensi otomatis saat CSAT pelanggan turun."
            ),
            "jumlah_komplain_6bln": (
                "**Jumlah Komplain** adalah predictor kedua terkuat. Setiap "
                "komplain tambahan meningkatkan risiko churn secara signifikan, "
                "terutama setelah komplain ke-2. **Implikasi**: prioritaskan "
                "*proactive outreach* setelah komplain pertama, jangan tunggu "
                "komplain berulang."
            ),
            "paket": (
                "**Jenis Paket** (terutama Prabayar) berkontribusi signifikan. "
                "Pelanggan Prabayar tidak terikat kontrak, sehingga lebih mudah "
                "berpindah provider. **Implikasi**: tawarkan insentif migrasi "
                "ke Pascabayar/Hybrid dengan benefit tambahan."
            ),
            "punya_streaming_bundle": (
                "**Streaming Bundle** menciptakan *stickiness* — pelanggan dengan "
                "bundle ini cenderung lebih loyal. **Implikasi**: kampanye trial "
                "gratis bundle untuk pelanggan high-risk tanpa bundle."
            ),
            "punya_ewallet_linked": (
                "**E-Wallet Linked** juga menciptakan *stickiness* melalui "
                "integrasi ekosistem digital. **Implikasi**: dorong adopsi "
                "linking e-wallet sebagai bagian onboarding pelanggan baru."
            ),
        }

        for _, row in top3.iterrows():
            feat = row["feature"]
            with st.container(border=True):
                st.markdown(f"**#{list(top3['feature']).index(feat)+1}. `{feat}`** (importance: {row['importance']:.3f})")
                st.markdown(
                    interpretations.get(
                        feat,
                        "Fitur ini berkontribusi signifikan terhadap prediksi model. "
                        "Analisis lebih lanjut direkomendasikan untuk fitur ini.",
                    )
                )

    # TAB 3: Segmentasi Risiko
    with tab3:
        st.markdown("### Segmentasi Risiko Populasi Pelanggan")
        st.markdown(
            f"Threshold segmentasi: **High Risk** (≥{RISK_THRESHOLDS['high']*100:.0f}%), "
            f"**Medium Risk** ({RISK_THRESHOLDS['medium']*100:.0f}%–"
            f"{RISK_THRESHOLDS['high']*100:.0f}%), **Low Risk** "
            f"(<{RISK_THRESHOLDS['medium']*100:.0f}%)."
        )

        segment_summary = (
            population_risk.groupby("risk_segment")
            .agg(
                jumlah_pelanggan=(ID_COL, "count"),
                actual_churn_rate=(TARGET_COL, "mean"),
                avg_predicted_proba=("churn_probability", "mean"),
            )
            .reindex(["High Risk", "Medium Risk", "Low Risk"])
        )
        segment_summary["actual_churn_rate"] = (segment_summary["actual_churn_rate"] * 100).round(1)
        segment_summary["avg_predicted_proba"] = (segment_summary["avg_predicted_proba"] * 100).round(1)
        segment_summary["persentase_populasi"] = (
            segment_summary["jumlah_pelanggan"] / len(population_risk) * 100
        ).round(1)

        c1, c2 = st.columns([1, 1])

        with c1:
            colors_map = {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"}
            fig = px.pie(
                segment_summary.reset_index(),
                values="jumlah_pelanggan",
                names="risk_segment",
                title="Distribusi Segmen Risiko",
                color="risk_segment",
                color_discrete_map=colors_map,
                hole=0.4,
            )
            st.plotly_chart(fig, width='stretch')

        with c2:
            fig = px.bar(
                segment_summary.reset_index(),
                x="risk_segment",
                y="actual_churn_rate",
                title="Validasi: Actual Churn Rate per Segmen",
                color="risk_segment",
                color_discrete_map=colors_map,
                labels={"actual_churn_rate": "Actual Churn Rate (%)", "risk_segment": "Segmen"},
                text="actual_churn_rate",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')

            is_monotonic = (
                segment_summary.loc["Low Risk", "actual_churn_rate"]
                <= segment_summary.loc["Medium Risk", "actual_churn_rate"]
                <= segment_summary.loc["High Risk", "actual_churn_rate"]
            )
            if is_monotonic:
                st.success(
                    "**Segmentasi VALID**: actual churn rate meningkat monoton "
                    "dari Low → Medium → High Risk."
                )
            else:
                st.warning("egmentasi tidak monoton — perlu investigasi lebih lanjut.")

        st.markdown("---")
        st.markdown("### Detail per Segmen & Rekomendasi Aksi")

        display_summary = segment_summary.rename(
            columns={
                "jumlah_pelanggan": "Jumlah Pelanggan",
                "actual_churn_rate": "Actual Churn Rate (%)",
                "avg_predicted_proba": "Avg Predicted Probability (%)",
                "persentase_populasi": "% Populasi",
            }
        )
        st.dataframe(display_summary, width='stretch')

        for segment in ["High Risk", "Medium Risk", "Low Risk"]:
            color = segment_color(segment)
            row = segment_summary.loc[segment]
            with st.container(border=True):
                st.markdown(
                    f"<h4 style='color:{color}; margin-top:0;'>{segment} "
                    f"({int(row['jumlah_pelanggan']):,} pelanggan, "
                    f"{row['persentase_populasi']}% populasi)</h4>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Aksi**: {RECOMMENDATION_MAP[segment]}")

    # TAB 4: Business Impact
    with tab4:
        st.markdown("### Simulasi Estimasi Business Impact")
        st.markdown(
            "Simulasi ini mengilustrasikan potensi dampak finansial program "
            "retensi berbasis model. **Parameter di bawah dapat disesuaikan** "
            "untuk skenario yang berbeda — gunakan data historis aktual "
            "perusahaan untuk keputusan investasi nyata."
        )

        st.markdown("#### Parameter Simulasi")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            arpu = st.number_input(
                "ARPU Bulanan (Rp)", min_value=10_000, max_value=1_000_000,
                value=100_000, step=10_000,
            )
        with p2:
            cost_promo = st.number_input(
                "Biaya Promo per Pelanggan (Rp, one-time)", min_value=0,
                max_value=500_000, value=50_000, step=5_000,
            )
        with p3:
            effectiveness = st.slider(
                "Efektivitas Retensi (%)", min_value=10, max_value=80, value=40, step=5,
            ) / 100
        with p4:
            horizon_months = st.number_input(
                "Horizon Revenue (bulan)", min_value=1, max_value=24, value=6, step=1,
            )
        with st.expander("Skala Populasi"):
            total_base = st.number_input(
                "Total Basis Pelanggan", min_value=1_000, max_value=10_000_000,
                value=100_000, step=10_000,
            )

        # Hitung berdasarkan population_risk yang sudah dihitung
        scale_factor = total_base / len(population_risk)

        high_risk = population_risk[population_risk["risk_segment"] == "High Risk"]
        high_risk_scaled = int(len(high_risk) * scale_factor)

        actual_churners_high_risk = high_risk[high_risk[TARGET_COL] == 1]
        actual_churners_scaled = int(len(actual_churners_high_risk) * scale_factor)

        saved_customers = int(actual_churners_scaled * effectiveness)
        monthly_revenue_saved = saved_customers * arpu
        revenue_saved_horizon = monthly_revenue_saved * horizon_months
        promo_cost_total = high_risk_scaled * cost_promo
        net_impact = revenue_saved_horizon - promo_cost_total
        payback_months = (
            promo_cost_total / monthly_revenue_saved if monthly_revenue_saved > 0 else float("inf")
        )

        st.markdown("---")
        st.markdown(f"#### Hasil Simulasi (Basis: {total_base:,} Pelanggan)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pelanggan High Risk", f"{high_risk_scaled:,}")
        c2.metric("Akan Churn (Actual)", f"{actual_churners_scaled:,}")
        c3.metric("Berhasil Diretensi", f"{saved_customers:,}")
        c4.metric("Biaya Promo (one-time)", f"Rp {promo_cost_total:,.0f}")

        c5, c6, c7 = st.columns(3)
        c5.metric(f"Revenue Saved ({horizon_months} bln)", f"Rp {revenue_saved_horizon:,.0f}")
        c6.metric(
            "Net Impact",
            f"Rp {net_impact:,.0f}",
            delta="Positive" if net_impact > 0 else "Negative",
            delta_color="normal" if net_impact > 0 else "inverse",
        )
        if payback_months != float("inf"):
            c7.metric("Payback Period", f"{payback_months:.1f} bulan")
        else:
            c7.metric("Payback Period", "N/A")

        if net_impact > 0:
            st.success(
                f" Dengan parameter ini, program retensi pada segmen High Risk "
                f"**layak dijalankan**: payback dalam **{payback_months:.1f} bulan**, "
                f"setelahnya menghasilkan net positive **Rp {monthly_revenue_saved:,.0f}/bulan** "
                f"secara recurring."
            )
        else:
            st.warning(
                "Dengan parameter ini, program retensi belum *cost-effective* "
                "dalam horizon yang dipilih. Coba naikkan horizon, efektivitas "
                "retensi, atau turunkan biaya promo."
            )

        # Waterfall chart
        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative", "relative", "total"],
                x=["Revenue Saved", "Biaya Promo", "Net Impact"],
                y=[revenue_saved_horizon, -promo_cost_total, net_impact],
                text=[
                    f"+Rp {revenue_saved_horizon:,.0f}",
                    f"-Rp {promo_cost_total:,.0f}",
                    f"Rp {net_impact:,.0f}",
                ],
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#e74c3c"}},
                increasing={"marker": {"color": "#2ecc71"}},
                totals={"marker": {"color": "#3498db"}},
            )
        )
        fig.update_layout(
            title=f"Waterfall: Net Impact Program Retensi ({horizon_months} bulan horizon)",
            height=400,
        )
        st.plotly_chart(fig, width='stretch')


if __name__ == "__main__":
    main()
