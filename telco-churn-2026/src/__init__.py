"""
src package

Modul-modul inti project Telco Churn Prediction 2026:

- `utils`              : konfigurasi path, logging, helper pickle, konstanta bisnis.
- `data_processing`    : generate dataset sintetis + data cleaning.
- `feature_engineering`: feature creation, encoding, scaling.
- `model`              : training, evaluasi, dan inference model XGBoost.

Contoh penggunaan end-to-end:

    from src.data_processing import generate_synthetic_data, clean_data
    from src.feature_engineering import build_features_pipeline
    from src.model import split_data, train_model, evaluate_model

    raw = generate_synthetic_data()
    clean = clean_data(raw)
    features, encoders, scaler = build_features_pipeline(clean, fit=True)

    X_train, X_test, y_train, y_test, feat_names = split_data(features)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
"""

__version__ = "1.0.0"
