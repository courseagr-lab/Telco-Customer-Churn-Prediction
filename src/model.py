"""
Modul untuk training, evaluasi, dan prediksi model churn (XGBoost Classifier).

Tahapan:
1. Split train/test (stratified).
2. Handle class imbalance via `scale_pos_weight`.
3. Train XGBClassifier.
4. Evaluasi: classification report, ROC-AUC, confusion matrix.
5. Cross-validation (StratifiedKFold).
6. Feature importance.
7. Save model + metrics ke disk.
8. Fungsi `predict_single` untuk inference satu pelanggan (dipakai di Streamlit).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from src.utils import (
    FEATURES_DATA_PATH,
    FEATURE_LIST_PATH,
    ID_COL,
    MODEL_PATH,
    METRICS_PATH,
    TARGET_COL,
    get_logger,
    load_pickle,
    save_pickle,
    segment_risk,
)

logger = get_logger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 5

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "eval_metric": "auc",
    "random_state": RANDOM_STATE,
}

# 1. DATA SPLIT
def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Split DataFrame menjadi train/test set (stratified by target).

    Kolom `customer_id` dan `churn` di-drop dari fitur X.

    Parameters:
    df : pd.DataFrame
        DataFrame hasil feature engineering (telco_features.csv).
    test_size : float
        Proporsi data test.
    random_state : int
        Seed reproducibility.

    Returns:
    tuple
        (X_train, X_test, y_train, y_test, feature_names)
    """
    drop_cols = [c for c in [ID_COL, TARGET_COL] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[TARGET_COL]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(
        "Split data: train=%s, test=%s, churn_rate_train=%.2f%%, churn_rate_test=%.2f%%",
        X_train.shape,
        X_test.shape,
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_names


# 2. TRAINING
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any] = None) -> XGBClassifier:
    """
    Train XGBClassifier dengan `scale_pos_weight` otomatis untuk menangani
    class imbalance (jumlah non-churn jauh lebih banyak dari churn).

    Parameters:
    X_train, y_train : training set.
    params : dict, optional
        Override hyperparameter XGBoost. Default: `XGB_PARAMS`.

    Returns:
    XGBClassifier
        Model yang sudah di-fit.
    """
    params = {**XGB_PARAMS, **(params or {})}

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    params["scale_pos_weight"] = scale_pos_weight

    logger.info(
        "Training XGBClassifier: n_pos=%d, n_neg=%d, scale_pos_weight=%.3f, params=%s",
        n_pos,
        n_neg,
        scale_pos_weight,
        params,
    )

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Training selesai.")
    return model


# ---------------------------------------------------------------------------
# 3. EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluasi model pada test set.

    Returns
    -------
    dict
        Berisi: classification_report (dict), roc_auc, confusion_matrix (list),
        roc_curve (fpr, tpr, thresholds), y_pred, y_proba.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    logger.info("ROC-AUC test set: %.4f", auc)
    logger.info(
        "Recall churn (kelas 1): %.4f | Precision churn: %.4f",
        report["1"]["recall"],
        report["1"]["precision"],
    )
    logger.info("Confusion matrix:\n%s", cm)

    return {
        "classification_report": report,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        },
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def cross_validate_model(
    X: pd.DataFrame, y: pd.Series, params: Dict[str, Any] = None, n_folds: int = N_CV_FOLDS
) -> Dict[str, float]:
    """
    Stratified K-Fold cross-validation menggunakan ROC-AUC sebagai metrik.

    Parameter:
    X, y : seluruh dataset (bukan hanya train).
    params : dict, optional
        Hyperparameter XGBoost (scale_pos_weight dihitung dari keseluruhan y).
    n_folds : int
        Jumlah fold.

    Returns:
    dict
        {'mean': float, 'std': float, 'scores': list[float]}
    """
    params = {**XGB_PARAMS, **(params or {})}
    n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
    params["scale_pos_weight"] = n_neg / n_pos if n_pos > 0 else 1.0

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    model = XGBClassifier(**params)

    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    logger.info(
        "Cross-validation (%d-fold) ROC-AUC: mean=%.4f, std=%.4f, scores=%s",
        n_folds,
        scores.mean(),
        scores.std(),
        np.round(scores, 4).tolist(),
    )

    return {"mean": float(scores.mean()), "std": float(scores.std()), "scores": scores.tolist()}


def get_feature_importance(
    model: XGBClassifier, feature_names: list[str], top_n: int = 15
) -> pd.DataFrame:
    """
    Ekstrak feature importance dari model XGBoost, diurutkan descending.

    Returns:
    pd.DataFrame
        Kolom: feature, importance. Diurutkan descending, top_n baris.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)
    df_imp = df_imp.reset_index(drop=True)
    return df_imp


# 4. SAVE / LOAD
def save_model_artifacts(
    model: XGBClassifier, feature_names: list[str], metrics: Dict[str, Any]
) -> None:
    """Simpan model, daftar fitur, dan metrics ke disk (models/)."""
    save_pickle(model, MODEL_PATH)
    save_pickle(feature_names, FEATURE_LIST_PATH)

    # Hilangkan array besar (y_pred, y_proba) dari metrics sebelum disimpan,
    # supaya file metrics.pkl ringkas dan hanya berisi summary.
    metrics_to_save = {
        k: v for k, v in metrics.items() if k not in ("y_pred", "y_proba")
    }
    save_pickle(metrics_to_save, METRICS_PATH)

    logger.info("Model artifacts saved: %s, %s, %s", MODEL_PATH, FEATURE_LIST_PATH, METRICS_PATH)


def load_model() -> XGBClassifier:
    """Load model XGBoost dari `models/churn_model_xgb.pkl`."""
    return load_pickle(MODEL_PATH)


def load_feature_list() -> list[str]:
    """Load daftar nama fitur (urutan kolom saat training) dari disk."""
    return load_pickle(FEATURE_LIST_PATH)


def load_metrics() -> Dict[str, Any]:
    """Load summary metrics evaluasi model dari disk."""
    return load_pickle(METRICS_PATH)


# 5. INFERENCE
def predict_proba_df(model: XGBClassifier, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Prediksi probabilitas churn untuk sebuah DataFrame.

    Memastikan urutan & kelengkapan kolom sesuai `feature_names` (urutan saat
    training) sebelum dilempar ke model -- mencegah error "feature mismatch".

    Parameters:
    model : XGBClassifier
        Model terlatih.
    X : pd.DataFrame
        Data yang sudah melalui `transform_new_data` (encoded + scaled).
    feature_names : list[str]
        Urutan kolom fitur saat training (dari `load_feature_list()`).

    Returns:
    np.ndarray
        Array probabilitas churn (kelas 1), shape (n_samples,).
    """
    missing_cols = set(feature_names) - set(X.columns)
    if missing_cols:
        raise ValueError(f"Kolom fitur hilang pada input: {missing_cols}")

    X_ordered = X[feature_names]
    return model.predict_proba(X_ordered)[:, 1]


def predict_single(
    model: XGBClassifier, feature_names: list[str], features_row: pd.DataFrame
) -> Dict[str, Any]:
    """
    Prediksi untuk SATU pelanggan (dipakai di halaman Prediksi Churn Streamlit).

    Parameters:
    model : XGBClassifier
    feature_names : list[str]
    features_row : pd.DataFrame
        DataFrame 1 baris, hasil `transform_new_data`.

    Returns
    dict
        {'churn_probability': float, 'risk_segment': str, 'prediction': int}
    """
    proba = predict_proba_df(model, features_row, feature_names)[0]
    segment = segment_risk(proba)
    prediction = int(proba >= 0.5)

    return {
        "churn_probability": float(proba),
        "risk_segment": segment,
        "prediction": prediction,
    }


# CLI ENTRY POINT
def main() -> None:
    """Pipeline lengkap: load features -> split -> train -> evaluate -> CV -> save."""
    if not FEATURES_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURES_DATA_PATH} tidak ditemukan. "
            f"Jalankan `python -m src.feature_engineering` terlebih dahulu."
        )

    df = pd.read_csv(FEATURES_DATA_PATH)
    logger.info("Loaded features data: shape=%s", df.shape)

    X_train, X_test, y_train, y_test, feature_names = split_data(df)

    model = train_model(X_train, y_train)
    eval_results = evaluate_model(model, X_test, y_test)

    X_all = df.drop(columns=[ID_COL, TARGET_COL])
    y_all = df[TARGET_COL]
    cv_results = cross_validate_model(X_all, y_all)

    feat_imp = get_feature_importance(model, feature_names)

    metrics = {
        "classification_report": eval_results["classification_report"],
        "roc_auc": eval_results["roc_auc"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "roc_curve": eval_results["roc_curve"],
        "cv_roc_auc_mean": cv_results["mean"],
        "cv_roc_auc_std": cv_results["std"],
        "cv_scores": cv_results["scores"],
        "feature_importance": feat_imp.to_dict(orient="records"),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    save_model_artifacts(model, feature_names, metrics)

    print("\n=== MODEL TRAINING SUMMARY ===")
    print(f"ROC-AUC (test)      : {eval_results['roc_auc']:.4f}")
    print(f"CV ROC-AUC (5-fold) : {cv_results['mean']:.4f} +/- {cv_results['std']:.4f}")
    print(f"Recall (churn)      : {eval_results['classification_report']['1']['recall']:.4f}")
    print(f"Precision (churn)   : {eval_results['classification_report']['1']['precision']:.4f}")
    print(f"F1-score (churn)    : {eval_results['classification_report']['1']['f1-score']:.4f}")
    print("\nTop 5 Feature Importance:")
    print(feat_imp.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
